"""Remote-centre-of-motion metrics: how far the tool shaft stays from the RCM point.

Shaft line at each sample: through the tip T along the unit axis d (tip -> handle).
  rcm_error  distance from the RCM point P to that line (mm): the number that matters
  rcm_depth  (T - P) . (-d): how far the tip is past P along the shaft (mm, + = inserted)
"""
import numpy as np


def line_metrics(P, tips, axes):
    """P (3,), tips (N,3), axes (N,3) unit -> (rcm_error (N,), depth (N,))."""
    v = np.asarray(P) - tips
    along = np.einsum('ij,ij->i', v, axes)
    perp = v - along[:, None] * axes
    return np.linalg.norm(perp, axis=1), along


def best_fit_pivot(tips, axes):
    """Point closest (least squares) to all shaft lines: where the tool really pivoted.

    Returns (point or None if the shaft barely rotated, condition number)."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for a, d in zip(tips, axes):
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ a
    cond = np.linalg.cond(A)
    if not np.isfinite(cond) or cond > 1e4:
        return None, float(cond)
    return np.linalg.solve(A, b), float(cond)


def summarise(P, tips, axes, t):
    """Summary of one RCM segment (all arrays visible-only)."""
    if len(tips) < 2:
        return {'samples': int(len(tips))}
    err, depth = line_metrics(P, tips, axes)
    ang = np.degrees(np.arccos(np.clip(axes @ axes.mean(0) / np.linalg.norm(axes.mean(0)), -1, 1)))
    s = {
        'samples': int(len(tips)),
        'duration_s': float(t[-1] - t[0]),
        'rcm_point_mm': np.round(P, 3).tolist(),
        'error_mean_mm': float(err.mean()),
        'error_rms_mm': float(np.sqrt(np.mean(err ** 2))),
        'error_p95_mm': float(np.percentile(err, 95)),
        'error_max_mm': float(err.max()),
        'within_1mm_pct': float(100 * np.mean(err <= 1.0)),
        'within_2mm_pct': float(100 * np.mean(err <= 2.0)),
        'depth_min_mm': float(depth.min()),
        'depth_max_mm': float(depth.max()),
        'shaft_swing_deg': float(2 * ang.max()),
    }
    fit, cond = best_fit_pivot(tips, axes)
    if fit is not None:
        ferr, _ = line_metrics(fit, tips, axes)
        s.update({'fitted_pivot_mm': np.round(fit, 3).tolist(),
                  'fitted_pivot_offset_mm': float(np.linalg.norm(fit - P)),
                  'error_rms_about_fitted_pivot_mm': float(np.sqrt(np.mean(ferr ** 2)))})
    return s


def segments(active):
    """[(start, stop)] index ranges where active is true."""
    a = np.r_[False, np.asarray(active, bool), False].astype(int)
    edges = np.flatnonzero(np.diff(a))
    return list(zip(edges[::2], edges[1::2]))
