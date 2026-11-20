"""Continuous tracking with start/stop recording and an RCM point.

The Vega is polled in a background thread at the tracking rate. Recording
sessions and the RCM point can be changed at any time from any thread
(keyboard, UDP control, ROS link).
"""
import re
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from . import rcm
from .calib import ToolCalibration
from .session import Session
from .vega import Vega

RECORDINGS = Path(__file__).resolve().parent.parent / 'recordings'
RCM_LATCH_TIMEOUT = 0.5     # s: an RCM request waits this long for a visible frame
RCM_LATCH_AVERAGE = 0.1     # s: tip averaged over this window when latching


class Tracker:
    def __init__(self, host, tool='8700340', rate=60, log=print, calib=None):
        self.log = log
        self.tool = tool
        self.rate = rate
        # calib: a different calibration for the same ROM, e.g. the robot tool's tip
        # through a strapped-on wand (robot-tcp --save-calib)
        self.calib = ToolCalibration.load(calib or tool)
        self.vega = Vega(host)
        self.vega.init()
        info = self.vega.info().splitlines()
        self.serial = info[1].split(':')[-1].strip() if len(info) > 1 else '?'
        self.vega.load_rom(tool)
        self.vega.start(rate)

        self.lock = threading.RLock()
        self.session = None
        self.rcm_point = None
        self._rcm_request = None        # (deadline, source)
        self._recent = deque()          # (unix, tip) visible samples for latching
        self.latest = None              # (unix, pose, tip, axis, rcm_err, depth)
        self.fps = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # -- control ---------------------------------------------------------
    def start_recording(self, run_dir=None, start_unix=None, source='manual', name=None):
        """run_dir: given by the teleop (its NEW_RUN). Otherwise (terminal) always
        recordings/<date_time>[_<name>] in this package."""
        with self.lock:
            if self.session:
                if self.session.dir == Path(run_dir or '') and self.session.source == source:
                    return f'already recording to {self.session.dir}'
                self.stop_recording(source=f'superseded by {source}')
            if not run_dir:
                stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', Path(name).name).strip('._') if name else ''
                run_dir = RECORDINGS / (f'{stamp}_{safe}' if safe else stamp)
            run_dir = Path(run_dir)
            self.session = Session(run_dir, start_unix, source, meta={
                'tool': self.tool, 'vega_serial': self.serial, 'rate_hz': self.rate,
                'calibration': vars(self.calib).copy() | {'status': self.calib.calibrated},
                'date': datetime.now().isoformat(timespec='seconds')})
            if self.rcm_point is not None:
                self.session.event('rcm_set', 'already set', self.rcm_point)
            return f'recording ({source}) -> {run_dir}'

    def stop_recording(self, source='manual'):
        with self.lock:
            s, self.session = self.session, None
        if s is None:
            return 'not recording'
        s.meta['stopped_by'] = source
        metrics = s.close()
        msg = f'saved {s.rows} rows ({s.visible} visible) -> {s.dir}'
        for m in metrics:
            if 'error_rms_mm' in m:
                msg += (f'\n    RCM {m["t_start"]:.1f}-{m["t_end"]:.1f} s: error rms {m["error_rms_mm"]:.2f} '
                        f'max {m["error_max_mm"]:.2f} mm, swing {m["shaft_swing_deg"]:.1f} deg')
        return msg

    def request_rcm(self, source='manual'):
        """Latch the tool tip (averaged over the last ~0.1 s) as the RCM point."""
        with self.lock:
            self._rcm_request = (time.time() + RCM_LATCH_TIMEOUT, source)
        return 'latching RCM at the tool tip...'

    def set_rcm(self, point, source='manual'):
        with self.lock:
            self.rcm_point = None if point is None else np.asarray(point, float)
            if self.session:
                self.session.event('rcm_set' if point is not None else 'rcm_clear', source, point)
        if point is None:
            return f'RCM cleared ({source})'
        return f'RCM set ({source}) at [{point[0]:.2f} {point[1]:.2f} {point[2]:.2f}] mm'

    def close(self):
        self._running = False
        self._thread.join(timeout=2)
        msg = self.stop_recording() if self.session else None
        self.vega.close()
        return msg

    # -- loop ------------------------------------------------------------
    def _loop(self):
        last_frame = None
        n, t_rate = 0, time.time()
        while self._running:
            try:
                f = self.vega.read()
            except Exception as e:      # keep going through a glitch, but say so
                self.log(f'Vega read failed: {e}')
                time.sleep(0.5)
                continue
            pose = f.tools[0]
            if pose.frame == last_frame:
                time.sleep(0.002)
                continue
            last_frame = pose.frame
            n += 1
            if f.t - t_rate >= 1.0:
                self.fps, n, t_rate = n / (f.t - t_rate), 0, f.t
            tip = axis = None
            err = depth = float('nan')
            with self.lock:
                if pose.visible:
                    R = pose.rotation()
                    tip = R @ self.calib.tip + pose.pos
                    axis = R @ self.calib.axis
                    self._recent.append((f.t, tip))
                while self._recent and self._recent[0][0] < f.t - RCM_LATCH_AVERAGE:
                    self._recent.popleft()
                self._resolve_rcm_request(f.t)
                if pose.visible and self.rcm_point is not None:
                    e, dp = rcm.line_metrics(self.rcm_point, tip[None], axis[None])
                    err, depth = float(e[0]), float(dp[0])
                if self.session:
                    self.session.write(f.t, pose.frame, pose, tip, axis, self.rcm_point, err, depth)
                self.latest = (f.t, pose, tip, axis, err, depth)

    def _resolve_rcm_request(self, now):
        if self._rcm_request is None:
            return
        deadline, source = self._rcm_request
        if self._recent and self._recent[-1][0] > now - 0.05:
            self._rcm_request = None
            point = np.mean([p for _, p in self._recent], axis=0)
            self.log(self.set_rcm(point, source))
        elif now > deadline:
            self._rcm_request = None
            self.log(f'RCM NOT latched ({source}): tool not visible')

    def status(self):
        with self.lock:
            if self.latest is None:
                return 'waiting for first frame'
            _, pose, tip, _, err, depth = self.latest
            rec = f'REC {self.session.t(time.time()):6.1f}s' if self.session else 'idle      '
            if not pose.visible:
                return f'{rec} | {self.fps:4.0f} Hz | tool {pose.status}'
            s = f'{rec} | {self.fps:4.0f} Hz | tip [{tip[0]:8.2f} {tip[1]:8.2f} {tip[2]:8.2f}] fit {pose.error:.2f}'
            if self.rcm_point is not None:
                s += f' | RCM err {err:5.2f} depth {depth:6.1f}'
            return s
