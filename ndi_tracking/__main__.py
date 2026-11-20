"""NDI Polaris Vega tool tracking: python3 -m ndi_tracking <command> ...

  run     track continuously; start/stop recordings and set the RCM point from
          the keyboard, `ctl`, or the MacroMicro teleop (GREY button, RCM mode)
  ctl     send start / stop / rcm / clear / status to a running `run`
  view    show a recording (default: the latest one)
  pivot   calibrate the tool tip (tip in a divot, swivel the tool)
  axis    calibrate the shaft axis (roll the tool about its shaft)
  mount   check the probe on the robot lines up with tool0 z (rotate wrist 3)
  robot-tcp  find the robot tool tip in tool0 for the teleop's RCM (freedrive pivot)
  check   load every bundled ROM and report what the camera sees
"""
import argparse
import select
import socket
import sys
import threading
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from .vega import DEFAULT_HOST, ROM_DIR, Vega

CONTROL_PORT = 47595        # UDP on localhost, used by `ctl`
KEYS = "keys: [s] start/stop recording  [r] set RCM at tool tip  [c] clear RCM  [q] quit"


# ---------------------------------------------------------------- run
class Console:
    """Status line at the bottom, log messages scrolling above it."""
    def __init__(self):
        self.lock = threading.Lock()
        self.tty = sys.stdout.isatty()

    def log(self, msg):
        with self.lock:
            sys.stdout.write(('\r\033[K' if self.tty else '') + f'{msg}\n')
            sys.stdout.flush()

    def status(self, line):
        if self.tty:
            with self.lock:
                sys.stdout.write('\r\033[K' + line)
                sys.stdout.flush()


def handle(tracker, cmd):
    """One control command (from a key, `ctl` or a script) -> reply text."""
    parts = cmd.strip().split(maxsplit=1)
    if not parts:
        return ''
    c, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else None)
    if c == 'start':
        return tracker.start_recording(name=arg)
    if c == 'stop':
        return tracker.stop_recording()
    if c == 'toggle':
        return tracker.stop_recording() if tracker.session else tracker.start_recording()
    if c == 'rcm':
        if arg:
            return tracker.set_rcm([float(v) for v in arg.replace(',', ' ').split()])
        return tracker.request_rcm()
    if c == 'clear':
        return tracker.set_rcm(None)
    if c == 'status':
        return tracker.status()
    return f'unknown command {c!r} (start [name] | stop | toggle | rcm [x y z] | clear | status)'


def control_server(tracker, console, stop_event):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(('127.0.0.1', CONTROL_PORT))
    except OSError:
        console.log(f'control port {CONTROL_PORT} busy: is another `run` going? (`ctl` disabled)')
        return
    sock.settimeout(0.2)
    while not stop_event.is_set():
        try:
            data, addr = sock.recvfrom(4096)
        except socket.timeout:
            continue
        cmd = data.decode(errors='replace')
        try:
            reply = handle(tracker, cmd)
        except Exception as e:
            reply = f'error: {e}'
        if cmd.strip() != 'status':
            console.log(f'[ctl] {cmd.strip()}: {reply}')
        sock.sendto(reply.encode(), addr)


def cmd_run(args):
    from .tracker import Tracker
    console = Console()
    tracker = Tracker(args.host, args.tool, args.rate, log=console.log, calib=args.calib)
    console.log(f'tracking {args.tool} at {args.rate} Hz on Vega {tracker.serial}, calibration '
                f'{args.calib or args.tool} ({tracker.calib.calibrated})')
    if tracker.calib.pivot_rms_mm is None:
        console.log(f'WARNING: tip not calibrated, so "tip" is the ROM origin (first marker). '
                    f'Run: python3 -m ndi_tracking pivot -t {args.tool}')
    stop_event = threading.Event()
    threading.Thread(target=control_server, args=(tracker, console, stop_event), daemon=True).start()

    ros_shutdown = None
    if args.ros != 'off':
        try:
            from . import ros_link
            ros_shutdown = ros_link.start(tracker, args.recording_topic, args.rcm_topic, log=console.log)
        except ImportError as e:
            msg = f'ROS link off ({e}); source ROS 2 to link to the teleop'
            if args.ros == 'on':
                tracker.close()
                raise SystemExit(msg)
            console.log(msg)
    if args.rcm:
        console.log(handle(tracker, f'rcm {args.rcm}'))
    if args.record:
        console.log(tracker.start_recording())

    keys = sys.stdin.isatty()
    old_term = None
    if keys:
        import termios
        import tty
        old_term = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        console.log(KEYS + '   (or: python3 -m ndi_tracking ctl ...)')
    t0 = time.time()
    try:
        while args.duration <= 0 or time.time() - t0 < args.duration:
            console.status(tracker.status())
            if keys and select.select([sys.stdin], [], [], 0.2)[0]:
                k = sys.stdin.read(1).lower()
                if k == 'q':
                    break
                cmd = {'s': 'toggle', ' ': 'toggle', 'r': 'rcm', 'c': 'clear'}.get(k)
                if cmd:
                    console.log(handle(tracker, cmd))
                elif k in 'h?':
                    console.log(KEYS)
            elif not keys:
                time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if old_term is not None:
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
        stop_event.set()
        msg = tracker.close()
        if msg:
            console.log(msg)
        if ros_shutdown:
            ros_shutdown()
        console.log('stopped')


def cmd_ctl(args):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(15)     # stop waits for the metrics
    sock.sendto(' '.join(args.command).encode(), ('127.0.0.1', CONTROL_PORT))
    try:
        print(sock.recv(65536).decode())
    except socket.timeout:
        raise SystemExit('no reply: is `python3 -m ndi_tracking run` running?')


# ---------------------------------------------------------------- calibration
def load_raw(path):
    """Poses saved by collect(): (R, t, fit error)."""
    from .vega import quat_to_rot
    a = np.loadtxt(path, delimiter=',', skiprows=1)
    return [quat_to_rot(q) for q in a[:, 2:6]], list(a[:, 6:9]), a[:, 9]


def collect(args, what):
    """Poses (R, t, fit error) of the tool for args.duration seconds, also saved to
    calibrations/raw/ (re-analyse with --file)."""
    if args.file:
        return load_raw(args.file)
    print(f'{what} for {args.duration:.0f} s, starting in 3 s...')
    R, t, err, rows = [], [], [], []
    with Vega(args.host) as v:
        v.init()
        v.load_rom(args.tool)
        v.start(60)
        time.sleep(3)
        print('recording')
        t0 = time.time()
        last = None
        while time.time() - t0 < args.duration:
            p = v.read().tools[0]
            if p.visible and p.frame != last:
                last = p.frame
                R.append(p.rotation())
                t.append(p.pos)
                err.append(p.error)
                rows.append([time.time(), p.frame, *p.quat, *p.pos, p.error])
            time.sleep(0.01)
    print(f'{len(R)} visible poses')
    from .calib import CALIBRATIONS
    raw = CALIBRATIONS / 'raw' / f'{args.cmd}_{args.tool}_{time.strftime("%Y%m%d_%H%M%S")}.csv'
    raw.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(raw, rows, delimiter=',', fmt='%.6f',
               header='unix_time,frame,q0,qx,qy,qz,x,y,z,fit_error', comments='')
    print(f'raw poses saved to {raw}')
    if len(R) < 50:
        raise SystemExit('too few visible poses; keep the tool in view and retry')
    return R, t, np.array(err)


def cmd_pivot_combine(args):
    """Fit one tip to several saved pivot attempts (each with its own pivot point)."""
    from .calib import CALIBRATIONS, ToolCalibration, pivot_calibrate, pivot_calibrate_multi
    files = args.combine or sorted(str(f) for f in (CALIBRATIONS / 'raw').glob(f'pivot_{args.tool}_*.csv'))
    if len(files) < 2:
        raise SystemExit(f'need at least 2 pivot attempts to combine, found {len(files)}')
    attempts = [load_raw(f)[:2] for f in files]
    tip, pivots, rms, total = pivot_calibrate_multi(attempts)
    print(f'{len(files)} attempts, {sum(len(R) for R, _ in attempts)} poses')
    for f, (R, t), r in zip(files, attempts, rms):
        own = pivot_calibrate(R, t)[0]
        print(f'  {Path(f).name}: own tip {own.round(2).tolist()} ({np.linalg.norm(own - tip):.2f} mm from combined), '
              f'RMS with combined tip {r:.2f} mm')
    # leave one out: how much does the tip depend on any single attempt?
    loo = [np.linalg.norm(pivot_calibrate_multi(attempts[:i] + attempts[i + 1:])[0] - tip)
           for i in range(len(attempts))] if len(attempts) > 2 else []
    print(f'combined tip {tip.round(2).tolist()} mm, RMS {total:.3f} mm'
          + (f', leave-one-out change max {max(loo):.2f} mm' if loo else ''))
    c = ToolCalibration.load(args.tool)
    c.tip_offset_mm, c.pivot_rms_mm = tip.tolist(), total
    print(f'saved {c.save()}')


def cmd_pivot(args):
    if args.combine is not None:
        return cmd_pivot_combine(args)
    from .calib import ToolCalibration, pivot_calibrate, pivot_diagnostics
    R, t, err = collect(args, f'Put the {args.tool} tip in a divot and swivel it (cone ~30 deg, all directions)')
    tip, pivot, rms = pivot_calibrate(R, t)
    tilt = np.degrees(max(np.arccos(np.clip((np.trace(R[0].T @ r) - 1) / 2, -1, 1)) for r in R))
    print(f'tip offset {tip.round(2).tolist()} mm ({np.linalg.norm(tip):.1f} mm from ROM origin), '
          f'RMS {rms:.3f} mm, swivel {tilt:.0f} deg')
    print(pivot_diagnostics(R, t, err, tip, pivot))
    if rms > 1.0:
        print('RMS > 1 mm: check the diagnostics above (slip, or tracking error at large angles)')
    if tilt < 20:
        print('swivel < 20 deg: tip poorly constrained; swivel further and record again')
    c = ToolCalibration.load(args.tool)
    c.tip_offset_mm, c.pivot_rms_mm = tip.tolist(), rms
    print(f'saved {c.save()}')


def cmd_axis(args):
    from .calib import ToolCalibration, axis_calibrate
    c = ToolCalibration.load(args.tool)
    if c.pivot_rms_mm is None:
        raise SystemExit('calibrate the tip first (pivot)')
    R, t, _ = collect(args, 'Tip in the divot, shaft as still as you can; twist the handle about the shaft '
                            '+-60 deg or more (balls still visible). Or rotate UR wrist 3')
    try:
        axis, info = axis_calibrate(R, t, c.tip)
    except ValueError as e:
        raise SystemExit(str(e))
    old = c.axis
    print(f'shaft axis {axis.round(4).tolist()} (tool frame, tip -> handle), '
          f'{np.degrees(np.arccos(np.clip(axis @ old, -1, 1))):.2f} deg from the previous/assumed axis, '
          f'{info["tip_axis_angle_from_rom_z_deg"]:.1f} deg from the marker line (ROM z)')
    print(f'twist {info["twist_range_deg"]:.0f} deg, shaft wobble {info["shaft_wobble_deg_rms"]:.1f} deg rms '
          f'(max {info["shaft_wobble_deg_max"]:.1f})')
    if info['shaft_wobble_deg_rms'] > 5:
        print('wobble > 5 deg rms: hold the shaft steadier (e.g. lightly through a ring of finger and thumb) '
              'and record again')
    c.shaft_axis, c.axis_info = axis.tolist(), info
    print(f'saved {c.save()}')



def cmd_mount(args):
    from .calib import ToolCalibration, mount_check
    c = ToolCalibration.load(args.tool)
    if c.pivot_rms_mm is None or c.shaft_axis is None:
        raise SystemExit('calibrate the tip (pivot) and shaft (axis) first')
    R, t, _ = collect(args, 'Probe mounted on the robot: rotate UR wrist 3 slowly, +-60 deg or more, '
                            'nothing else moving')
    m = mount_check(R, t, c.tip, c.axis)
    print(f'wrist 3 turned {m["twist_deg"]:.0f} deg (axis steady to {m["wrist3_axis_steadiness_deg_rms"]:.2f} deg rms)')
    print(f'probe shaft vs wrist-3 axis (tool0 z): {m["axis_angle_deg"]:.2f} deg')
    print(f'probe tip off the wrist-3 axis line:   {m["tip_off_wrist3_axis_mm"]:.2f} mm')
    if m['twist_deg'] < 40:
        print('turn wrist 3 further (+-60 deg) and repeat')
    if m['axis_angle_deg'] > 0.5 or m['tip_off_wrist3_axis_mm'] > 0.5:
        print('-> the teleop takes the shaft as tool0 z through the flange centre: the probe is not quite there, '
              'so expect that much RCM error from the mount alone (fix the holder, or set '
              'TOUCH_RCM_SHAFT_START / _DIRECTION to match)')
    else:
        print('-> mount matches the teleop\'s shaft (tool0 z) well')
    print('Shaft LENGTH (where along tool0 z the tip is) cannot be checked this way: measure flange -> tip '
          'and compare with TOUCH_RCM_SHAFT_START + TOUCH_RCM_SHAFT_LENGTH')


# ---------------------------------------------------------------- other
def cmd_view(args):
    from .viewer import find_recordings, resolve, view
    if args.list:
        for r in find_recordings():
            print(r)
        return
    for p in (args.paths or [None]):
        view(resolve(p), show=not args.no_show, save=args.save or args.no_show)


def cmd_check(args):
    with Vega(args.host) as v:
        v.init()
        print(v.info().splitlines()[1])
        for rom in sorted(ROM_DIR.glob('*.rom')):
            v.load_rom(rom)
        v.start(60)
        t0 = time.time()
        last = 0
        while time.time() - t0 < args.duration:
            f = v.read(strays=True)
            if f.strays is None:
                print(f'could not parse stray markers, raw reply: {f.raw!r}')
                f.strays = np.zeros((0, 3))
            if f.t - last >= 0.5:
                last = f.t
                vis = [f'{p.name} pos={p.pos.round(1).tolist()} err={p.error:.3f}'
                       for p in f.tools if p.visible]
                d = sorted(np.linalg.norm(a - b) for a, b in combinations(f.strays, 2))
                print(f'{f.t - t0:5.1f}s  strays={len(f.strays)}  '
                      f'marker spacings(mm)={np.round(d, 1).tolist()}  tracked={vis or "none"}')
            time.sleep(0.01)


def main():
    ap = argparse.ArgumentParser(prog='ndi_tracking', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host', default=DEFAULT_HOST)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('run', help='track; record on key / ctl / GREY button')
    p.add_argument('-t', '--tool', default='8700340', help='ROM name in roms/ or path to a .rom')
    p.add_argument('-r', '--rate', type=int, default=60, choices=(20, 30, 60), help='tracking rate, Hz')
    p.add_argument('--ros', choices=('auto', 'on', 'off'), default='auto',
                   help='link to the MacroMicro teleop (auto: if ROS 2 is sourced)')
    p.add_argument('--recording-topic', default='/trajectory_recorder/recording')
    p.add_argument('--rcm-topic', default='/ur16e_touch_teleop/rcm_active')
    p.add_argument('--calib', help='calibration name in calibrations/ (default: the ROM name); '
                   'e.g. robot_tool from robot-tcp to track the robot tool through a strapped-on wand')
    p.add_argument('--record', action='store_true', help='start a standalone recording straight away')
    p.add_argument('--rcm', help='fixed RCM point "x,y,z" mm (camera frame)')
    p.add_argument('-d', '--duration', type=float, default=0, help='quit after this many s (0 = never)')
    p.set_defaults(func=cmd_run)

    p = sub.add_parser('ctl', help='control a running `run`')
    p.add_argument('command', nargs='+', help='start [name] | stop | toggle | rcm [x y z] | clear | status')
    p.set_defaults(func=cmd_ctl)

    p = sub.add_parser('view', help='show a recording (default: latest)')
    p.add_argument('paths', nargs='*', help='recording folders (or their NDI_PATH.csv)')
    p.add_argument('--list', action='store_true', help='list all recordings')
    p.add_argument('--save', action='store_true', help='also save NDI_PLOT.png in the folder')
    p.add_argument('--no-show', action='store_true', help='only save the PNG')
    p.set_defaults(func=cmd_view)

    for name, func, dur, hlp in (('pivot', cmd_pivot, 20, 'calibrate the tool tip'),
                                 ('axis', cmd_axis, 20, 'calibrate the shaft axis'),
                                 ('mount', cmd_mount, 20, 'check the probe on the robot is along tool0 z')):
        p = sub.add_parser(name, help=hlp + ' (stop `run` first)')
        p.add_argument('-t', '--tool', default='8700340')
        p.add_argument('-d', '--duration', type=float, default=dur)
        p.add_argument('--file', help='re-analyse saved raw poses (calibrations/raw/*.csv) instead of recording')
        if name == 'pivot':
            p.add_argument('--combine', nargs='*', metavar='CSV',
                           help='fit one tip to several saved attempts (default: all in calibrations/raw/)')
        p.set_defaults(func=func)

    p = sub.add_parser('robot-tcp', help='find the robot tool tip in tool0 (freedrive pivot; + wand hand-eye)')
    p.add_argument('-t', '--tool', default='8700340', help='wand ROM')
    p.add_argument('-d', '--duration', type=float, default=40)
    p.add_argument('--no-ndi', action='store_true', help='robot only (no wand on the tool)')
    p.add_argument('--wand-is-tool', action='store_true',
                   help='the wand IS the tool: shaft direction from its calibrated axis + tip cross-check')
    p.add_argument('--holder-z', type=float, default=0.05,
                   help='m, where the shaft leaves the holder on tool0 z (direction when not --wand-is-tool)')
    p.add_argument('--shaft-length', type=float, default=0.262, help='m, TOUCH_RCM_SHAFT_LENGTH to use')
    p.add_argument('--save-calib', default='robot_tool', help='name of the tracking calibration written')
    p.add_argument('--base-frame', default='base_link')
    p.add_argument('--ee-frame', default='tool0')
    p.add_argument('--file', help='re-analyse a saved calibrations/raw/robot_tcp_*.npz')
    p.add_argument('--tip-ball', action='store_true',
                   help='a reflective ball on the tool tip, tracked as a stray marker (no divot; freedrive anywhere)')
    p.add_argument('--ball-radius', type=float, default=5.75,
                   help='mm; the tip is this far from the ball centre, towards the wand markers (NDI sphere: 11.5 mm dia)')
    p.set_defaults(func=lambda a: __import__('ndi_tracking.robot_tcp', fromlist=['main']).main(a))

    p = sub.add_parser('check', help='report visible markers/tools')
    p.add_argument('-d', '--duration', type=float, default=15)
    p.set_defaults(func=cmd_check)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
