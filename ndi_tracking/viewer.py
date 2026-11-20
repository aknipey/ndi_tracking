"""View NDI recordings: tip time series, RCM error / depth, 3D path with shaft lines.

A recording is any folder with NDI_PATH.csv: ndi_tracking/recordings/<date_time>/
(standalone) or a MacroMicro run folder (logs/NEW_RUN, logs/V<n>/<run>).
"""
import os
from pathlib import Path

import numpy as np
import yaml

from .session import load_path

RECORDINGS = Path(__file__).resolve().parent.parent / 'recordings'
MACROMICRO_LOGS = Path(os.path.expanduser(
    os.environ.get('MACROMICRO_LOGS', '~/ros2_ws/src/MacroMicroSurgicalRobot/logs')))
SHAFT_DRAW_MM = 150         # shaft lines drawn in 3D: to 20 mm past the RCM point, else this long
SHAFT_LINES = 40            # how many shaft lines to draw per RCM segment


def find_recordings():
    """All recording folders, oldest first."""
    found = set(p.parent for p in RECORDINGS.glob('*/NDI_PATH.csv'))
    if MACROMICRO_LOGS.is_dir():
        found |= set(p.parent for p in MACROMICRO_LOGS.glob('**/NDI_PATH.csv'))
    return sorted(found, key=lambda p: (p / 'NDI_PATH.csv').stat().st_mtime)


def resolve(arg):
    if arg is None:
        recs = find_recordings()
        if not recs:
            raise SystemExit('no recordings found')
        return recs[-1]
    p = Path(arg).expanduser()
    if p.is_file():
        p = p.parent
    if not (p / 'NDI_PATH.csv').exists():
        raise SystemExit(f'{p} has no NDI_PATH.csv')
    return p


def scroll_zoom_3d(fig, step=1.2):
    '''Mouse wheel zooms every 3D axes in fig about its centre (all axes equally).'''
    def on_scroll(event):
        ax = event.inaxes
        if ax is None or ax.name != '3d':
            return
        scale = 1 / step if event.button == 'up' else step
        for get, set_ in ((ax.get_xlim3d, ax.set_xlim3d), (ax.get_ylim3d, ax.set_ylim3d),
                          (ax.get_zlim3d, ax.set_zlim3d)):
            lo, hi = get()
            mid, half = (lo + hi) / 2, (hi - lo) / 2 * scale
            set_(mid - half, mid + half)
        fig.canvas.draw_idle()
    fig._scroll_zoom_cid = fig.canvas.mpl_connect('scroll_event', on_scroll)


def _spans(ax, t, active):
    from .rcm import segments
    for a, b in segments(active):
        ax.axvspan(t[a], t[b - 1], color='C3', alpha=0.08, lw=0)


def view(run_dir, show=True, save=False):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    d = load_path(run_dir / 'NDI_PATH.csv')
    if not d:
        raise SystemExit(f'{run_dir}/NDI_PATH.csv is empty')
    meta_f = run_dir / 'NDI_META.yaml'
    meta = yaml.safe_load(meta_f.read_text()) if meta_f.exists() else {}
    t = d['t']
    vis = d['visible'] == 1
    active = d['rcm_active'] == 1
    has_rcm = active.any()

    fig = plt.figure(figsize=(15, 9))
    rate = len(t) / max(t[-1] - t[0], 1e-9)
    calib = (meta.get('calibration') or {}).get('status', '')
    fig.suptitle(f'{run_dir}   {vis.sum()}/{len(t)} frames visible, {rate:.0f} Hz   {calib}', fontsize=10)
    gs = fig.add_gridspec(5, 2, width_ratios=[1.3, 1], hspace=0.15)

    rows = [('tip_x', 'tip x (mm)'), ('tip_y', 'tip y (mm)'), ('tip_z', 'tip z (mm)')]
    rows += [('rcm_error', 'RCM error (mm)'), ('rcm_depth', 'depth (mm)')] if has_rcm \
        else [('fit_error', 'marker fit (mm)')]
    axes = []
    for i, (k, label) in enumerate(rows):
        ax = fig.add_subplot(gs[i, 0], sharex=axes[0] if axes else None)
        ax.plot(t, d[k], lw=1, color=f'C{i}')          # NaN gaps = tool not visible
        ax.set_ylabel(label)
        ax.ticklabel_format(axis='y', useOffset=False)
        ax.grid(alpha=0.3)
        _spans(ax, t, active)
        if k == 'rcm_error':
            for lim, ls in ((1, '--'), (2, ':')):
                ax.axhline(lim, color='k', lw=0.8, ls=ls)
            ax.set_ylim(bottom=0)
        if i < len(rows) - 1:
            ax.tick_params(labelbottom=False)
        axes.append(ax)
    axes[-1].set_xlabel('time (s)' + ('   (red = RCM set)' if has_rcm else ''))

    ax3 = fig.add_subplot(gs[:3, 1], projection='3d')
    scroll_zoom_3d(fig)
    tips = np.c_[d['tip_x'], d['tip_y'], d['tip_z']]
    shaft = np.c_[d['axis_x'], d['axis_y'], d['axis_z']]
    pts = [tips[vis]]
    if vis.any():
        ax3.plot(*tips[vis].T, lw=1, color='C0', label='tip')
    segs = meta.get('rcm_segments') or []
    from .rcm import segments
    for a, b in segments(active & vis):
        idx = np.arange(a, b)[::max(1, (b - a) // SHAFT_LINES)]
        for i in idx:
            depth = d['rcm_depth'][i]
            length = depth + 20 if depth == depth and depth > 0 else SHAFT_DRAW_MM
            line = np.c_[tips[i], tips[i] + length * shaft[i]]
            ax3.plot(*line, color='0.6', lw=0.5)
    for s in segs:
        if 'rcm_point_mm' in s:
            p = np.array(s['rcm_point_mm'])
            ax3.scatter(*p, color='C3', marker='*', s=120)
            pts.append(p[None])
        if 'fitted_pivot_mm' in s:
            ax3.scatter(*s['fitted_pivot_mm'], color='k', marker='x', s=60)
    if segs:
        ax3.scatter([], [], color='C3', marker='*', label='RCM point')
        ax3.scatter([], [], color='k', marker='x', label='fitted pivot')
    allp = np.vstack(pts)
    if len(allp):
        span = max((allp.max(0) - allp.min(0)).max() / 2, 5)
        mid = (allp.max(0) + allp.min(0)) / 2
        for setlim, m in zip((ax3.set_xlim, ax3.set_ylim, ax3.set_zlim), mid):
            setlim(m - span, m + span)
    ax3.set_xlabel('x'); ax3.set_ylabel('y'); ax3.set_zlabel('z')
    ax3.set_title('camera frame (mm), scroll to zoom', fontsize=9)
    ax3.legend(fontsize=8, loc='upper left')

    txt = fig.add_subplot(gs[3:, 1])
    txt.axis('off')
    lines = []
    for i, s in enumerate(segs):
        if 'error_rms_mm' not in s:
            lines.append(f'RCM {i + 1}: {s.get("samples", 0)} visible samples - no metrics')
            continue
        lines.append(f'RCM {i + 1}  {s["t_start"]:.1f}-{s["t_end"]:.1f} s  visible {s["visible_pct"]:.0f}%')
        lines.append(f'  error  rms {s["error_rms_mm"]:.2f}  mean {s["error_mean_mm"]:.2f}  '
                     f'p95 {s["error_p95_mm"]:.2f}  max {s["error_max_mm"]:.2f} mm')
        lines.append(f'  within 1 mm {s["within_1mm_pct"]:.0f}%   2 mm {s["within_2mm_pct"]:.0f}%')
        lines.append(f'  depth {s["depth_min_mm"]:.1f} to {s["depth_max_mm"]:.1f} mm   '
                     f'shaft swing {s["shaft_swing_deg"]:.1f} deg')
        if 'fitted_pivot_offset_mm' in s:
            lines.append(f'  fitted pivot {s["fitted_pivot_offset_mm"]:.2f} mm from RCM point, '
                         f'error about it rms {s["error_rms_about_fitted_pivot_mm"]:.2f} mm')
    if not segs:
        lines.append('No RCM point set in this recording.')
        lines.append("Set one with 'r' in `run`, or RCM mode on the Touch (with --ros).")
    txt.text(0, 1, '\n'.join(lines), va='top', family='monospace', fontsize=9)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.06)
    if save:
        png = run_dir / 'NDI_PLOT.png'
        fig.savefig(png, dpi=110)
        print(f'saved {png}')
    if show:
        plt.show()
    plt.close(fig)
