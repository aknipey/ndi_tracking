"""One recording: NDI_PATH.csv + NDI_EVENTS.csv while running, NDI_META.yaml
(with RCM metrics) and NDI_PLOT.png when it stops.

NDI_PATH.csv columns (positions mm, camera frame):
  t           s since recording start (the MacroMicro recorder's start when it
              started the run, so it lines up with its `t`)
  unix_time   host receive time
  frame       Vega frame number (60 Hz)
  visible     1 if the tool was tracked
  tip_*       tool tip;  axis_*  unit shaft direction (tip -> handle)
  x,y,z,q*    raw ROM-frame pose (q0 scalar);  fit_error  RMS marker fit (mm)
  rcm_active  1 while an RCM point is set;  rcm_*  that point
  rcm_error   shaft line -> RCM point (mm);  rcm_depth  tip past the point (mm)
"""
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from . import rcm

PATH_COLUMNS = ['t', 'unix_time', 'frame', 'visible', 'tip_x', 'tip_y', 'tip_z',
                'axis_x', 'axis_y', 'axis_z', 'x', 'y', 'z', 'q0', 'qx', 'qy', 'qz',
                'fit_error', 'rcm_active', 'rcm_x', 'rcm_y', 'rcm_z', 'rcm_error', 'rcm_depth']


class Session:
    def __init__(self, run_dir, start_unix=None, source='manual', meta=None):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.start_unix = start_unix or time.time()
        self.source = source
        self.meta = dict(meta or {})
        self.rows = 0
        self.visible = 0
        self._path_f = open(self.dir / 'NDI_PATH.csv', 'w', newline='')
        self._path = csv.writer(self._path_f)
        self._path.writerow(PATH_COLUMNS)
        self._events_f = open(self.dir / 'NDI_EVENTS.csv', 'w', newline='')
        self._events = csv.writer(self._events_f)
        self._events.writerow(['t', 'unix_time', 'event', 'source', 'x', 'y', 'z'])
        self.event('start', source)

    def t(self, unix):
        return unix - self.start_unix

    def write(self, unix, frame, pose, tip, axis, rcm_point, rcm_err, depth):
        f6 = lambda a: [f'{v:.4f}' for v in a]  # noqa: E731
        row = [f'{self.t(unix):.4f}', f'{unix:.4f}', frame, int(pose.visible)]
        if pose.visible:
            row += f6(tip) + [f'{v:.6f}' for v in axis] + f6(pose.pos) + list(pose.quat) + [pose.error]
            self.visible += 1
        else:
            row += [''] * 14
        if rcm_point is None:
            row += [0, '', '', '', '', '']
        else:
            row += [1] + f6(rcm_point) + ([f'{rcm_err:.4f}', f'{depth:.4f}'] if pose.visible else ['', ''])
        self._path.writerow(row)
        self.rows += 1

    def event(self, name, source, point=None, unix=None):
        unix = unix or time.time()
        self._events.writerow([f'{self.t(unix):.4f}', f'{unix:.4f}', name, source]
                              + (list(np.round(point, 4)) if point is not None else ['', '', '']))
        self._events_f.flush()

    def close(self, plot=True):
        self.event('stop', self.source)
        self._path_f.close()
        self._events_f.close()
        metrics = compute_metrics(self.dir / 'NDI_PATH.csv')
        self.meta.update({
            'start_unix': self.start_unix,
            'source': self.source,
            'rows': self.rows,
            'visible_rows': self.visible,
            'rcm_segments': metrics,
            'notes': {'columns': __doc__.split('NDI_PATH.csv columns')[1].strip()},
        })
        with open(self.dir / 'NDI_META.yaml', 'w') as f:
            yaml.safe_dump(self.meta, f, sort_keys=False, default_flow_style=None)
        if plot and self.rows:
            # Separate process: keeps matplotlib out of the tracking loop
            subprocess.Popen([sys.executable, '-m', 'ndi_tracking', 'view', str(self.dir), '--no-show'],
                             cwd=Path(__file__).resolve().parent.parent,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return metrics


def load_path(path):
    """NDI_PATH.csv -> dict of float arrays (NaN for blanks)."""
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    return {k: np.array([float(r[k]) if r[k] != '' else np.nan for r in rows]) for k in rows[0]}


def compute_metrics(path):
    d = load_path(path)
    if not d:
        return []
    out = []
    active = d['rcm_active'] == 1
    P = np.c_[d['rcm_x'], d['rcm_y'], d['rcm_z']]
    # a new segment also starts whenever the RCM point changes
    change = np.r_[False, np.any(np.abs(np.diff(P, axis=0)) > 1e-6, axis=1)]
    for a, b in rcm.segments(active):
        cuts = [a] + [i for i in np.flatnonzero(change[a:b]) + a if i > a] + [b]
        for s, e in zip(cuts[:-1], cuts[1:]):
            vis = d['visible'][s:e] == 1
            tips = np.c_[d['tip_x'], d['tip_y'], d['tip_z']][s:e][vis]
            axes = np.c_[d['axis_x'], d['axis_y'], d['axis_z']][s:e][vis]
            m = {'t_start': float(d['t'][s]), 't_end': float(d['t'][e - 1]),
                 'visible_pct': float(100 * vis.mean())}
            m.update(rcm.summarise(P[s], tips, axes, d['t'][s:e][vis]))
            out.append(m)
    return out
