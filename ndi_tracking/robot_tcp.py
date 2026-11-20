"""Robot tool tip (TCP) calibration in tool0, for the teleop's RCM tool (TOUCH_RCM_SHAFT_*).

Put the tool tip in a divot and hand-guide the arm (UR freedrive) through as many
orientations as you can while the tip stays in the divot. base_link -> tool0 is
recorded from TF (the kinematics the teleop itself uses) and the point that stays
still is the tip: the same pivot fit as the wand, done by the robot.

If the wand is on the tool and visible, its pose is recorded too and a hand-eye fit
gives tool0 -> wand. That
  - with --wand-is-tool: gives the shaft direction in tool0 from the wand's calibrated
    axis, and an independent check of the tip (wand tip mapped into tool0);
  - always: writes a tracking calibration (calibrations/<--save-calib>.json) so
    `run --calib <name>` tracks the ROBOT TOOL's tip and shaft through the wand.

--tip-ball: a reflective ball is stuck on the tool tip instead, so no divot is needed:
hand-guide the arm anywhere in view, with as much rotation as possible. The Vega sees
the ball as a stray marker; the stray that stays fixed in the wand frame is the ball.
Its centre in tool0 comes from the robot kinematics alone (c_i = Y A_i p, camera pose Y
and p fitted together) and, as a check, from the hand-eye fit. The tip is the side of
the ball nearest the wand markers (back up the shaft): centre - radius * direction.
The wand's own calibration (calibrations/<--tool>.json) is only read, never written.
"""
import threading
import time
from datetime import datetime

import numpy as np

from . import handeye as he
from .calib import CALIBRATIONS, ToolCalibration, pivot_calibrate


# ------------------------------------------------------------------ recording
def record(args):
    """Returns (robot_t, robot_T [mm], ndi_t, ndi_T [mm], strays) recorded for args.duration s.
    strays: (N, 4) [t, x, y, z] camera-frame stray markers (mm), only with --tip-ball."""
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformException, TransformListener

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node('ndi_robot_tcp')
    buf = Buffer()
    TransformListener(buf, node)
    robot_t, robot_T = [], []
    last_stamp = [None]

    def poll():
        try:
            tf = buf.lookup_transform(args.base_frame, args.ee_frame, Time())
        except TransformException:
            return
        st = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
        if st == last_stamp[0]:
            return
        last_stamp[0] = st
        q, p = tf.transform.rotation, tf.transform.translation
        from .vega import quat_to_rot
        robot_t.append(st)
        robot_T.append(he.T(quat_to_rot([q.w, q.x, q.y, q.z]), 1000 * np.array([p.x, p.y, p.z])))

    node.create_timer(1 / 250, poll)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            try:
                ex.spin_once(timeout_sec=0.05)
            except Exception:
                break
    th = threading.Thread(target=spin, daemon=True)
    th.start()

    ndi_t, ndi_T, strays = [], [], []
    vega = None
    if not args.no_ndi:
        from .vega import Vega
        try:
            vega = Vega(args.host)
            vega.init()
            vega.load_rom(args.tool)
            vega.start(60)
        except OSError as e:
            print(f'NDI not available ({e}): robot-only calibration')
            vega = None

    try:
        print('waiting for TF ...', end='', flush=True)
        t_wait = time.time()
        while not robot_t and time.time() - t_wait < 10:
            time.sleep(0.1)
        if not robot_t:
            raise SystemExit(f'\nno TF {args.base_frame} -> {args.ee_frame}: is the robot driver running '
                             '(and ROS_DOMAIN_ID the same as the robot)?')
        print(f' ok. Tip in the divot; recording {args.duration:.0f} s in 3 s ...')
        time.sleep(3)
        n0 = len(robot_t)
        robot_t0 = robot_t[-1]
        t0 = time.time()
        last_frame = None
        while time.time() - t0 < args.duration:
            if vega:
                f = vega.read(strays=args.tip_ball)
                p = f.tools[0]
                if p.frame != last_frame:
                    last_frame = p.frame
                    if p.visible:
                        ndi_t.append(f.t)
                        ndi_T.append(he.T(p.rotation(), p.pos))
                    if f.strays is not None:
                        strays.extend([f.t, *m] for m in f.strays)
                time.sleep(0.004)
            else:
                time.sleep(0.05)
            el = time.time() - t0
            print(f'\r  {el:5.1f} s  robot {len(robot_t) - n0:6d}  ndi {len(ndi_t):5d}'
                  + (f'  strays {len(strays):6d}' if args.tip_ball else ''), end='', flush=True)
        print()
    finally:
        stop.set()
        th.join(timeout=2)
        ex.shutdown(timeout_sec=1)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if vega:
            vega.close()
    keep = [i for i, t in enumerate(robot_t) if t >= robot_t0]
    return (np.array([robot_t[i] for i in keep]), np.array([robot_T[i] for i in keep]),
            np.array(ndi_t), np.array(ndi_T).reshape(-1, 4, 4), np.array(strays).reshape(-1, 4))


# ------------------------------------------------------------------ analysis
def analyse(robot_t, robot_T, ndi_t, ndi_T, args, strays=None):
    if args.tip_ball:
        return analyse_ball(robot_t, robot_T, ndi_t, ndi_T, strays, args)
    R, t = robot_T[:, :3, :3], robot_T[:, :3, 3]
    swing = max(np.degrees(np.linalg.norm(he.rot_log(R[0].T @ r))) for r in R[::10])
    tip, pivot, rms = pivot_calibrate(R, t)
    res = np.linalg.norm(np.einsum('nij,j->ni', R, tip) + t - pivot, axis=1)
    quarters = [pivot_calibrate(R[q], t[q])[1] for q in np.array_split(np.arange(len(R)), 4)]
    print(f'\nROBOT PIVOT  ({len(R)} poses, swing {swing:.0f} deg, frames {args.base_frame} -> {args.ee_frame})')
    print(f'  tip in tool0: {tip.round(2).tolist()} mm   ({np.linalg.norm(tip):.1f} mm from the flange centre, '
          f'{np.hypot(tip[0], tip[1]):.2f} mm off the tool0 z axis)')
    print(f'  residual rms {rms:.2f}  p95 {np.percentile(res, 95):.2f}  max {res.max():.2f} mm')
    print('  pivot per quarter, mm from the 1st: '
          + '  '.join(f'{np.linalg.norm(q - quarters[0]):.2f}' for q in quarters) + '   (jump = tip slipped)')
    if swing < 25:
        print('  WARNING: swing < 25 deg, the tip is poorly constrained: tilt the tool further')
    if rms > 1.0:
        print('  WARNING: rms > 1 mm: tip moved in the divot, or the TF kinematics are not this arm\'s '
              '(ur_type / calibration)')

    result = {'date': datetime.now().isoformat(timespec='seconds'), 'base_frame': args.base_frame,
              'ee_frame': args.ee_frame, 'tip_tool0_mm': tip.tolist(), 'pivot_rms_mm': rms,
              'swing_deg': swing, 'robot_poses': int(len(R))}

    direction = None
    X = None
    if len(ndi_t) >= 100:
        try:
            X, info = he.solve_with_delay(robot_t, robot_T, ndi_t, ndi_T)
        except ValueError as e:
            print(f'\nHAND-EYE failed: {e}')
        if X is not None:
            print(f'\nHAND-EYE  tool0 -> wand  ({info["samples"]} paired samples, NDI delay '
                  f'{1000 * info["ndi_delay_s"]:.0f} ms)')
            print(f'  wand origin in tool0: {X[:3, 3].round(2).tolist()} mm')
            print(f'  consistency: camera pose spread {info["camera_pose_spread_mm_rms"]:.2f} mm / '
                  f'{info["camera_pose_spread_deg_rms"]:.3f} deg rms, rotation-angle mismatch '
                  f'{info["angle_mismatch_deg_rms"]:.3f} deg rms')
            if info['camera_pose_spread_mm_rms'] > 2:
                print('  WARNING: spread > 2 mm: the wand may have moved on the tool, or too little rotation')
            result['handeye'] = {'tool0_to_wand': X.tolist(), **info}
            wand = ToolCalibration.load(args.tool)
            if args.wand_is_tool:
                if wand.pivot_rms_mm is None or wand.shaft_axis is None:
                    print(f'  --wand-is-tool: calibrate the wand first (pivot + axis for {args.tool})')
                else:
                    wtip = X[:3, :3] @ wand.tip + X[:3, 3]
                    print(f'  wand tip in tool0 (NDI): {wtip.round(2).tolist()} mm -> '
                          f'{np.linalg.norm(wtip - tip):.2f} mm from the robot-pivot tip')
                    direction = -(X[:3, :3] @ wand.axis)         # handle -> tip, in tool0
                    result['wand_tip_tool0_mm'] = wtip.tolist()

    return finish(tip, direction, X, rms, result, args)


def finish(tip, direction, X, rms, result, args, tip_wand=None, axis_wand=None):
    """Print the teleop settings for this tip (mm, tool0) and save robot_tcp.json and,
    with a hand-eye X, the tracking calibration. direction None = from the holder point.
    tip_wand / axis_wand: tip and shaft axis (tip -> handle) measured in the wand frame
    directly; the tracking calibration uses them instead of mapping tip through X."""
    if direction is None:
        holder = np.array([0.0, 0.0, 1000 * args.holder_z])
        direction = (tip - holder) / np.linalg.norm(tip - holder)
        how = f'from the holder point (0, 0, {args.holder_z} m) on tool0 z to the tip'
    else:
        how = 'from the wand\'s calibrated shaft axis (hand-eye)'
    L = args.shaft_length
    start = tip / 1000 - L * direction
    print(f'\nFOR teleop_settings.py  (direction {how}; LENGTH = your measured shaft length)')
    print(f'TOUCH_RCM_SHAFT_START = [{start[0]:.4f}, {start[1]:.4f}, {start[2]:.4f}]     # m')
    print(f'TOUCH_RCM_SHAFT_DIRECTION = [{direction[0]:.4f}, {direction[1]:.4f}, {direction[2]:.4f}]')
    print(f'TOUCH_RCM_SHAFT_LENGTH = {L}               # m')
    print(f'  (tip = START + LENGTH * DIRECTION = {(tip / 1000).round(4).tolist()} m; '
          f'{np.degrees(np.arccos(np.clip(direction[2], -1, 1))):.2f} deg from tool0 z)')
    result.update({'direction_tool0': direction.tolist(), 'direction_source': how,
                   'teleop_settings': {'TOUCH_RCM_SHAFT_START': start.round(5).tolist(),
                                       'TOUCH_RCM_SHAFT_DIRECTION': direction.round(5).tolist(),
                                       'TOUCH_RCM_SHAFT_LENGTH': L}})

    current = teleop_tip()
    if current is not None:
        print(f'  current teleop_settings tip: {(current / 1000).round(4).tolist()} m -> this tip is '
              f'{np.linalg.norm(tip - current):.2f} mm from it  (diff {(tip - current).round(2).tolist()} mm)')
        result['teleop_settings_tip_before_mm'] = current.tolist()

    if tip_wand is not None:
        c = ToolCalibration(args.save_calib)
        c.tip_offset_mm = np.asarray(tip_wand).tolist()
        c.shaft_axis = np.asarray(axis_wand).tolist()
        c.pivot_rms_mm = result.get('ball_scatter_mm')
        c.axis_info = {'source': f'tip ball seen in the wand frame, ROM {args.tool}',
                       'ball_radius_mm': args.ball_radius}
        print(f'\nsaved tracking calibration {c.save()}  ->  python3 -m ndi_tracking run --calib {args.save_calib}'
              '  (tip from the ball in the wand frame: no robot kinematics or hand-eye involved)')
    elif X is not None:
        # tracking calibration: the robot tool's tip and shaft in the wand frame
        Xi = he.inv(X)
        c = ToolCalibration(args.save_calib)
        c.tip_offset_mm = (Xi[:3, :3] @ tip + Xi[:3, 3]).tolist()
        c.shaft_axis = (-(Xi[:3, :3] @ direction)).tolist()       # tip -> handle
        c.pivot_rms_mm = rms
        c.axis_info = {'source': f'robot_tcp hand-eye, ROM {args.tool}', 'direction': how}
        print(f'\nsaved tracking calibration {c.save()}  ->  python3 -m ndi_tracking run --calib {args.save_calib}')

    import json
    out = CALIBRATIONS / 'robot_tcp.json'
    CALIBRATIONS.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f'saved {out}')
    return result


def teleop_tip():
    """Tip (mm, tool0) in MacroMicro's teleop_settings.py, if that repo is next to this one."""
    import runpy
    from pathlib import Path
    f = Path(__file__).resolve().parents[2] / 'MacroMicroSurgicalRobot' / 'macro_micro_robot' / 'teleop_settings.py'
    try:
        cfg = runpy.run_path(str(f))
        return 1000 * (np.array(cfg['TOUCH_RCM_SHAFT_START'])
                       + cfg['TOUCH_RCM_SHAFT_LENGTH'] * np.array(cfg['TOUCH_RCM_SHAFT_DIRECTION']))
    except Exception:
        return None


def find_ball(ndi_t, ndi_T, strays, tol=3.0):
    """The stray marker fixed in the wand frame. Returns (centre in the wand frame mm,
    times, camera positions (N, 3) mm of that marker, rms scatter mm)."""
    pose = {t: T for t, T in zip(ndi_t, ndi_T)}
    t_s, cam, w = [], [], []
    for t, *m in strays:
        T = pose.get(t)
        if T is not None:
            t_s.append(t)
            cam.append(m)
            w.append(T[:3, :3].T @ (np.array(m) - T[:3, 3]))
    if len(w) < 100:
        raise SystemExit(f'only {len(w)} stray markers seen with the wand visible: is the tip ball in view?')
    t_s, cam, w = np.array(t_s), np.array(cam), np.array(w)
    cand = w[::max(1, len(w) // 500)]
    c = cand[np.argmax([(np.linalg.norm(w - p, axis=1) < tol).sum() for p in cand])]
    for _ in range(5):
        near = np.linalg.norm(w - c, axis=1) < tol
        c = np.median(w[near], 0)
    near = np.linalg.norm(w - c, axis=1) < tol
    rms = float(np.sqrt(np.mean(np.sum((w[near] - c) ** 2, 1))))
    return c, t_s[near], cam[near], rms


def kabsch(P, Q):
    """R, t minimising |R P + t - Q|."""
    pm, qm = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pm).T @ (Q - qm))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, qm - R @ pm


def fit_robot_point(A, cam, p0, iters=3000):
    """A point p fixed in tool0 seen by the camera at cam_i = Y (A_i p), Y = base -> camera
    unknown too. Alternates Y (Kabsch) and p (least squares). Returns (p, Y, residuals)."""
    R, t = A[:, :3, :3], A[:, :3, 3]
    p = np.asarray(p0, float)
    for _ in range(iters):
        RY, tY = kabsch(np.einsum('nij,j->ni', R, p) + t, cam)
        b = (cam - tY) @ RY                                  # camera -> base
        p_new = np.einsum('nji,nj->ni', R, b - t).mean(0)    # base -> tool0
        done = np.linalg.norm(p_new - p) < 1e-6
        p = p_new
        if done:
            break
    RY, tY = kabsch(np.einsum('nij,j->ni', R, p) + t, cam)
    res = np.linalg.norm((np.einsum('nij,j->ni', R, p) + t) @ RY.T + tY - cam, axis=1)
    return p, he.T(RY, tY), res


def analyse_ball(robot_t, robot_T, ndi_t, ndi_T, strays, args):
    if strays is None or len(ndi_t) < 100:
        raise SystemExit('--tip-ball needs the wand on the tool and in view (and a recording made with --tip-ball)')
    r = args.ball_radius
    c_w, t_b, cam, scatter = find_ball(ndi_t, ndi_T, strays)
    wand = ToolCalibration.load(args.tool)
    print(f'\nTIP BALL  {len(cam)} sightings ({100 * len(cam) / len(ndi_t):.0f}% of wand frames), '
          f'scatter in the wand frame {scatter:.2f} mm rms')
    print(f'  ball centre in the wand frame: {c_w.round(2).tolist()} mm  '
          f'({np.linalg.norm(c_w - wand.tip):.2f} mm from the wand\'s calibrated tip)')
    if scatter > 1.0:
        print('  WARNING: scatter > 1 mm: the ball or the wand is loose, or another stray was picked')
    result = {'date': datetime.now().isoformat(timespec='seconds'), 'method': 'tip_ball',
              'base_frame': args.base_frame, 'ee_frame': args.ee_frame, 'ball_radius_mm': r,
              'ball_wand_mm': c_w.tolist(), 'ball_scatter_mm': scatter, 'ball_sightings': int(len(cam))}

    X, delay = None, 0.0
    try:
        X, info = he.solve_with_delay(robot_t, robot_T, ndi_t, ndi_T)
        delay = info['ndi_delay_s']
        print(f'\nHAND-EYE  tool0 -> wand  ({info["samples"]} paired samples, NDI delay {1000 * delay:.0f} ms, '
              f'camera pose spread {info["camera_pose_spread_mm_rms"]:.2f} mm / '
              f'{info["camera_pose_spread_deg_rms"]:.3f} deg rms)')
        if info['camera_pose_spread_mm_rms'] > 2:
            print('  WARNING: spread > 2 mm: the wand may have moved on the tool, or too little rotation')
        result['handeye'] = {'tool0_to_wand': X.tolist(), **info}
    except ValueError as e:
        print(f'\nHAND-EYE failed: {e}')
    c_he = X[:3, :3] @ c_w + X[:3, 3] if X is not None else None

    A, C = he.match(robot_t, robot_T, t_b, cam, delay)
    A, C = np.array(A), np.array(C)
    if len(A) < 100:
        raise SystemExit(f'only {len(A)} ball sightings paired with robot poses')
    swing = max(np.degrees(np.linalg.norm(he.rot_log(A[0][:3, :3].T @ a[:3, :3]))) for a in A[::10])
    p0 = c_he if c_he is not None else (teleop_tip() if teleop_tip() is not None else np.array([0, 0, 300.0]))
    c, Y, res = fit_robot_point(A, C, p0)
    print(f'\nROBOT-ONLY  ball centre in {args.ee_frame}: {c.round(2).tolist()} mm  ({len(A)} poses, swing '
          f'{swing:.0f} deg, residual rms {np.sqrt(np.mean(res ** 2)):.2f}  p95 {np.percentile(res, 95):.2f} mm)')
    if c_he is not None:
        print(f'  hand-eye gives {c_he.round(2).tolist()} mm -> {np.linalg.norm(c - c_he):.2f} mm apart')
    if swing < 25:
        print('  WARNING: swing < 25 deg, the centre is poorly constrained: tilt the tool further')
    if np.sqrt(np.mean(res ** 2)) > 1.0:
        print('  WARNING: rms > 1 mm: TF kinematics may not be this arm\'s (ur_type / calibration)')
    result.update({'ball_tool0_mm': c.tolist(), 'ball_tool0_handeye_mm': None if c_he is None else c_he.tolist(),
                   'robot_fit_rms_mm': float(np.sqrt(np.mean(res ** 2))), 'swing_deg': swing,
                   'base_to_camera': Y.tolist(), 'robot_poses': int(len(A))})

    if args.wand_is_tool and X is not None:
        direction = -(X[:3, :3] @ wand.axis)
    else:
        holder = np.array([0.0, 0.0, 1000 * args.holder_z])
        direction = (c - holder) / np.linalg.norm(c - holder)
    tip = c - r * direction                                  # side of the ball facing the markers
    print(f'\nTIP = ball centre - {r} mm along the shaft: {tip.round(2).tolist()} mm in {args.ee_frame}')
    if X is not None:
        wtip = X[:3, :3] @ wand.tip + X[:3, 3]
        print(f'  wand\'s calibrated tip in tool0: {wtip.round(2).tolist()} mm -> {np.linalg.norm(wtip - tip):.2f} mm away')
    # Tracking calibration straight from the wand frame: the side of the ball facing the
    # markers. Shaft axis: the wand's own with --wand-is-tool, else the tool0 direction mapped
    # through the hand-eye (only the axis then depends on X; without X, the wand's axis)
    if args.wand_is_tool or X is None:
        axis_w = wand.axis
    else:
        axis_w = -(X[:3, :3].T @ direction)
    tip_w = c_w + r * axis_w
    result['tip_wand_mm'] = tip_w.tolist()
    # direction None: finish() takes it from the holder point, the same line as above
    return finish(tip, direction if args.wand_is_tool and X is not None else None, X,
                  result['robot_fit_rms_mm'], result, args, tip_wand=tip_w, axis_wand=axis_w)


def main(args):
    if args.save_calib == args.tool:
        raise SystemExit(f'--save-calib {args.save_calib} would overwrite the wand calibration: pick another name')
    if args.file:
        d = np.load(args.file)
        data = d['robot_t'], d['robot_T'], d['ndi_t'], d['ndi_T'], d['strays'] if 'strays' in d else None
    else:
        data = record(args)
        raw = CALIBRATIONS / 'raw' / f'robot_tcp_{time.strftime("%Y%m%d_%H%M%S")}.npz'
        raw.parent.mkdir(parents=True, exist_ok=True)
        np.savez(raw, robot_t=data[0], robot_T=data[1], ndi_t=data[2], ndi_T=data[3], strays=data[4])
        print(f'raw data saved to {raw}')
    if len(data[0]) < 200:
        raise SystemExit(f'only {len(data[0])} robot poses: did the arm move?')
    analyse(*data[:4], args, strays=data[4])
