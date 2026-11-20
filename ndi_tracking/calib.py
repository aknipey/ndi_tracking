"""Tool calibration: tip offset (pivot) and shaft axis (roll), in the tool's ROM frame.

8700340 passive probe ROM frame (from the .rom): markers at (0,0,0), (0,0,50),
(0,25,100), (0,-25,135) mm, all facing +x. The reported position is the first
marker (nearest the shaft), NOT the tip, so the tip must be pivot-calibrated.
"""
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

CALIBRATIONS = Path(__file__).resolve().parent.parent / 'calibrations'


@dataclass
class ToolCalibration:
    tool: str
    tip_offset_mm: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # Unit vector along the shaft, pointing from the tip back towards the handle
    shaft_axis: Optional[list] = None
    pivot_rms_mm: Optional[float] = None
    axis_info: Optional[dict] = None
    date: str = ''

    @property
    def tip(self) -> np.ndarray:
        return np.asarray(self.tip_offset_mm, float)

    @property
    def axis(self) -> np.ndarray:
        """Shaft axis; if never calibrated, assume it runs from the tip to the ROM origin."""
        if self.shaft_axis is not None:
            return np.asarray(self.shaft_axis, float)
        n = np.linalg.norm(self.tip)
        return -self.tip / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])

    @property
    def calibrated(self) -> str:
        parts = ['tip' if self.pivot_rms_mm is not None else 'tip NOT calibrated',
                 'axis' if self.shaft_axis is not None else 'axis assumed (tip->ROM origin)']
        return ', '.join(parts)

    @classmethod
    def load(cls, tool) -> 'ToolCalibration':
        f = CALIBRATIONS / f'{tool}.json'
        if not f.exists():
            return cls(tool)
        d = json.loads(f.read_text())
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def save(self) -> Path:
        CALIBRATIONS.mkdir(exist_ok=True)
        self.date = datetime.now().isoformat(timespec='seconds')
        f = CALIBRATIONS / f'{self.tool}.json'
        f.write_text(json.dumps(asdict(self), indent=2))
        return f


def pivot_calibrate(rotations, positions):
    """Tip fixed in a divot, tool swivelled: R_i @ tip + t_i = pivot for all i.

    Returns (tip in tool frame, pivot in camera frame, RMS residual mm)."""
    R = np.asarray(rotations)
    t = np.asarray(positions)
    n = len(R)
    A = np.zeros((3 * n, 6))
    A[:, :3] = R.reshape(3 * n, 3)
    A[:, 3:] = -np.tile(np.eye(3), (n, 1))
    x, *_ = np.linalg.lstsq(A, -t.reshape(-1), rcond=None)
    tip, pivot = x[:3], x[3:]
    residuals = np.linalg.norm(np.einsum('nij,j->ni', R, tip) + t - pivot, axis=1)
    return tip, pivot, float(np.sqrt(np.mean(residuals ** 2)))



def pivot_calibrate_multi(attempts):
    """Several pivot recordings [(R, t), ...], each with its own pivot point (the divot
    may move between them), one shared tip. Averages out the tracker's position- and
    angle-dependent error. Returns (tip, [pivot per attempt], [rms per attempt], rms)."""
    n = [len(R) for R, _ in attempts]
    N, k = sum(n), len(attempts)
    A = np.zeros((3 * N, 3 + 3 * k))
    b = np.zeros(3 * N)
    row = 0
    for j, (R, t) in enumerate(attempts):
        m = 3 * len(R)
        A[row:row + m, :3] = np.asarray(R).reshape(m, 3)
        A[row:row + m, 3 + 3 * j:6 + 3 * j] = -np.tile(np.eye(3), (len(R), 1))
        b[row:row + m] = -np.asarray(t).reshape(-1)
        row += m
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    tip = x[:3]
    pivots = [x[3 + 3 * j:6 + 3 * j] for j in range(k)]
    rms = []
    for (R, t), p in zip(attempts, pivots):
        r = np.linalg.norm(np.einsum('nij,j->ni', np.asarray(R), tip) + np.asarray(t) - p, axis=1)
        rms.append(float(np.sqrt(np.mean(r ** 2))))
    total = float(np.sqrt(sum(r ** 2 * m for r, m in zip(rms, n)) / N))
    return tip, pivots, rms, total


def axis_calibrate(rotations, positions, tip):
    """Tool twisted about its own shaft (by hand with the tip in a divot, or UR wrist 3).

    The shaft is the tool-frame direction that keeps pointing the same way in the
    camera frame while everything else swings round with the twist: the direction a
    maximising |mean(R_i) a|. Hand wobble only tilts it slightly (+-3 deg wobble while
    twisting +-60 deg -> ~0.4 deg axis error in simulation).

    Returns (unit axis pointing tip -> handle, info dict)."""
    R = np.asarray(rotations)
    rel = np.einsum('ji,njk->nik', R[0], R)            # R0^T R_i, tool frame
    ang = np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1))
    twist = float(np.degrees(ang.max()))
    if twist < 40:
        raise ValueError(f'only twisted {twist:.0f} deg; twist further (+-60 deg or more)')
    _, sv, Vt = np.linalg.svd(R.mean(0))
    axis = Vt[0]
    if axis @ -np.asarray(tip) < 0:                    # point from the tip towards the handle
        axis = -axis
    dirs = R @ axis                                    # camera-frame shaft direction per pose
    mean = dirs.mean(0) / np.linalg.norm(dirs.mean(0))
    wobble = np.degrees(np.arccos(np.clip(dirs @ mean, -1, 1)))
    info = {'twist_range_deg': twist,
            'shaft_wobble_deg_rms': float(np.sqrt(np.mean(wobble ** 2))),
            'shaft_wobble_deg_max': float(wobble.max()),
            'tip_axis_angle_from_rom_z_deg': float(np.degrees(np.arccos(abs(axis[2])))),
            'n_poses': int(len(R))}
    return axis, info


def pivot_diagnostics(R, t, fit_err, tip, pivot):
    """Why is the pivot RMS what it is? Slip shows as the pivot point moving over time;
    tracking error shows as residuals growing with fit error / viewing angle."""
    R, t, fit_err = np.asarray(R), np.asarray(t), np.asarray(fit_err)
    res = np.linalg.norm(np.einsum('nij,j->ni', R, tip) + t - pivot, axis=1)
    # angle between the markers' facing direction (tool +x) and the line to the camera
    view = -t / np.linalg.norm(t, axis=1, keepdims=True)
    facing = np.degrees(np.arccos(np.clip(np.einsum('ni,ni->n', R[:, :, 0], view), -1, 1)))
    lines = [f'residual  median {np.median(res):.2f}  p95 {np.percentile(res, 95):.2f}  '
             f'max {res.max():.2f} mm   marker fit error median {np.median(fit_err):.3f} mm']
    # pivot point fitted separately on each quarter of the recording: slipping moves it
    quarters = [pivot_calibrate(R[q], t[q]) for q in np.array_split(np.arange(len(R)), 4)]
    piv = np.array([q[1] for q in quarters])
    lines.append('pivot point per quarter, mm from the 1st quarter\'s: '
                 + '  '.join(f'{np.linalg.norm(p - piv[0]):.2f}' for p in piv)
                 + '   (a slip shows as a jump; all < ~0.5 = no slip)')
    if fit_err.std() > 0:
        lines.append(f'residual vs marker fit error: correlation {np.corrcoef(res, fit_err)[0, 1]:+.2f}')
    for lo, hi in ((0, 20), (20, 35), (35, 50), (50, 180)):
        m = (facing >= lo) & (facing < hi)
        if m.sum() >= 10:
            lines.append(f'  markers {lo:2d}-{hi:3d} deg from facing camera: {m.sum():4d} poses, '
                         f'residual rms {np.sqrt(np.mean(res[m] ** 2)):.2f} mm, '
                         f'fit error {fit_err[m].mean():.3f} mm')
    good = facing < 35
    if good.sum() >= 100:
        tip2, _, rms2 = pivot_calibrate(R[good], t[good])
        lines.append(f'refit on poses < 35 deg from facing: RMS {rms2:.2f} mm, '
                     f'tip moves {np.linalg.norm(tip2 - tip):.2f} mm')
    return '\n'.join(lines)


def mount_check(rotations, positions, tip, shaft_axis):
    """Probe mounted on the robot, UR wrist 3 rotated: compare the wrist-3 axis (tool0 z,
    which the teleop's RCM mode takes as the shaft) with the probe's calibrated shaft.

    Returns dict: angle between them, the probe tip's distance from the wrist-3 axis line,
    and how far along that line the tip sits from the point nearest the camera-fixed
    rotation centre (no robot data, so only the relative geometry)."""
    R = np.asarray(rotations)
    t = np.asarray(positions)
    _, _, Vt = np.linalg.svd(R.mean(0))
    w3 = Vt[0]
    if w3 @ np.asarray(shaft_axis) < 0:
        w3 = -w3
    rel = np.einsum('ji,njk->nik', R[0], R)
    twist = float(np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)).max()))
    # tool-frame points on the wrist-3 axis do not move: R_i x + t_i = c for all i
    n = len(R)
    A = np.zeros((3 * n, 6))
    A[:, :3] = R.reshape(3 * n, 3)
    A[:, 3:] = -np.tile(np.eye(3), (n, 1))
    x, *_ = np.linalg.lstsq(A, -t.reshape(-1), rcond=None)
    p = x[:3]                                  # a point on the wrist-3 axis, tool frame
    d = np.asarray(tip) - p
    off = d - (d @ w3) * w3
    dirs = R @ w3
    m = dirs.mean(0) / np.linalg.norm(dirs.mean(0))
    wob = np.degrees(np.arccos(np.clip(dirs @ m, -1, 1)))
    return {'twist_deg': twist,
            'axis_angle_deg': float(np.degrees(np.arccos(np.clip(w3 @ shaft_axis, -1, 1)))),
            'tip_off_wrist3_axis_mm': float(np.linalg.norm(off)),
            'wrist3_axis_steadiness_deg_rms': float(np.sqrt(np.mean(wob ** 2)))}
