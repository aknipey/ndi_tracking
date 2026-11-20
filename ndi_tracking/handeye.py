"""Hand-eye calibration: where the wand sits on the robot tool.

A_i = base_link -> tool0 (robot), B_i = camera -> wand (NDI), recorded together.
With X = tool0 -> wand and Y = base_link -> camera both fixed:  A_i X = Y B_i,
so for any pair  (A_i^-1 A_j) X = X (B_i^-1 B_j)   (AX = XB, Park & Martin).
"""
import numpy as np


def rot_log(R):
    """Rotation matrix -> rotation vector (rad)."""
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    a = np.arccos(c)
    if a < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - a < 1e-6:                       # 180 deg: axis from R + I
        w, V = np.linalg.eigh(R + np.eye(3))
        return V[:, -1] * a
    return v * a / (2 * np.sin(a))


def T(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def inv(M):
    R, t = M[:3, :3], M[:3, 3]
    return T(R.T, -R.T @ t)


def pick_pairs(A, min_angle_deg=10, n_pairs=600, seed=0):
    """Random pose pairs whose robot rotation differs by more than min_angle."""
    rng = np.random.default_rng(seed)
    n = len(A)
    pairs = []
    for _ in range(n_pairs * 20):
        i, j = rng.integers(0, n, 2)
        if i == j:
            continue
        ang = np.degrees(np.linalg.norm(rot_log(A[i][:3, :3].T @ A[j][:3, :3])))
        if ang > min_angle_deg:
            pairs.append((i, j))
            if len(pairs) >= n_pairs:
                break
    return pairs


def solve(A, B, pairs=None):
    """A, B: lists of 4x4. Returns (X = tool0 -> wand, info)."""
    pairs = pairs or pick_pairs(A)
    if len(pairs) < 10:
        raise ValueError('not enough rotation between poses (need many pairs > 10 deg apart)')
    rel = [(inv(A[i]) @ A[j], inv(B[i]) @ B[j]) for i, j in pairs]
    # rotation: alpha = R_X beta for the rotation vectors of each relative motion
    M = np.zeros((3, 3))
    angle_mismatch = []
    for Aij, Bij in rel:
        a, b = rot_log(Aij[:3, :3]), rot_log(Bij[:3, :3])
        M += np.outer(b, a)
        angle_mismatch.append(np.degrees(abs(np.linalg.norm(a) - np.linalg.norm(b))))
    w, V = np.linalg.eigh(M.T @ M)
    RX = V @ np.diag(w ** -0.5) @ V.T @ M.T
    U, _, Vt = np.linalg.svd(RX)               # clean up to an exact rotation
    RX = U @ Vt
    # translation: (R_Aij - I) t_X = R_X t_Bij - t_Aij
    C = np.vstack([Aij[:3, :3] - np.eye(3) for Aij, _ in rel])
    d = np.concatenate([RX @ Bij[:3, 3] - Aij[:3, 3] for Aij, Bij in rel])
    tX, *_ = np.linalg.lstsq(C, d, rcond=None)
    X = T(RX, tX)
    # consistency: Y_i = A_i X B_i^-1 should be the same camera pose for every sample
    Y = np.array([A[i] @ X @ inv(B[i]) for i in range(len(A))])
    Yt = Y[:, :3, 3]
    Ym = Y.mean(0)
    U, _, Vt = np.linalg.svd(Ym[:3, :3])
    Rm = U @ Vt
    rot_spread = [np.degrees(np.linalg.norm(rot_log(Rm.T @ y[:3, :3]))) for y in Y]
    info = {'pairs': len(pairs),
            'angle_mismatch_deg_rms': float(np.sqrt(np.mean(np.square(angle_mismatch)))),
            'camera_pose_spread_mm_rms': float(np.sqrt(np.mean(np.sum((Yt - Yt.mean(0)) ** 2, 1)))),
            'camera_pose_spread_deg_rms': float(np.sqrt(np.mean(np.square(rot_spread))))}
    return X, info


def match(robot_t, robot_T, ndi_t, ndi_T, offset=0.0, max_gap=0.01):
    """Pair each NDI sample with the nearest robot sample at ndi_t - offset."""
    robot_t = np.asarray(robot_t)
    A, B = [], []
    for t, Tb in zip(ndi_t, ndi_T):
        k = np.searchsorted(robot_t, t - offset)
        cands = [c for c in (k - 1, k) if 0 <= c < len(robot_t)]
        if not cands:
            continue
        c = min(cands, key=lambda c: abs(robot_t[c] - (t - offset)))
        if abs(robot_t[c] - (t - offset)) <= max_gap:
            A.append(robot_T[c])
            B.append(Tb)
    return A, B


def solve_with_delay(robot_t, robot_T, ndi_t, ndi_T, delays=np.arange(-0.10, 0.1001, 0.005)):
    """Hand-eye at each NDI delay (s, NDI receive time - robot stamp); keep the most
    consistent one. Returns (X, info) with info['ndi_delay_s']."""
    best = None
    pairs, n_prev = None, None
    for dly in delays:
        A, B = match(robot_t, robot_T, ndi_t, ndi_T, dly)
        if len(A) < 50:
            continue
        if len(A) != n_prev:            # same pairs across delays where possible: fairer score
            pairs, n_prev = pick_pairs(A), len(A)
        try:
            X, info = solve(A, B, pairs)
        except ValueError:
            continue
        score = info['camera_pose_spread_mm_rms']
        if best is None or score < best[0]:
            best = (score, X, dict(info, ndi_delay_s=float(dly), samples=len(A)))
    if best is None:
        raise ValueError('could not pair robot and NDI samples (no overlap in time?)')
    return best[1], best[2]
