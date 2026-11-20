# ndi_tracking

Standalone Python (numpy, matplotlib, PyYAML; ROS 2 optional) tracker/recorder for the
**NDI Polaris Vega VT** (part 10010595, S/N P9-24159). Talks the NDI Combined API directly over
TCP port 8765. Built to check a robot's **remote centre of motion (RCM)**: how far the tool
shaft strays from the RCM point.

## Setup (each session)

```bash
./setup_vega.sh      # idempotent: network address, waits for the Vega to boot, checks the API
```

- The Vega is powered over Ethernet by its PoE power supply. Cable from the PC goes to the supply's data port.
- The Vega has a fixed IP **192.168.3.3**. The PC's built-in port `enp98s0` is 192.168.3.10, in the `vega` NetworkManager
  profile. The UR (192.168.0.100) is on the USB Ethernet adapter `enx6c6e0740fc7e` as 192.168.0.10, in the `ur16e` profile.
  The pendant's External Control connects back to 192.168.0.10.
- Measurement volume starts about **0.95 m** from the camera; best around 1.5–2 m.

## Tool definition files

`ndi_tracking/roms/` must hold the NDI `.rom` file of each tool you track (8700340 for the passive probe used in
this work; 8700248, 8700302, 8700339 and 8700449 were also on the lab PC). They are supplied by NDI with the tools
(or from NDI Toolbox), so they are not in this repository.

## How this fits with the thesis

This is the independent measurement used in the thesis: NDI tracking of every RCM run, the wand pivot and hand-eye
calibrations, and the coupling re-seat tests. The teleoperation stack is
[MacroMicroSurgicalRobot](https://github.com/aknipey/MacroMicroSurgicalRobot), whose recordings carry the NDI files
this tool wrote (`NDI_META.yaml`, `NDI_EVENTS.csv`). Overview of all the thesis repositories:
[Honours-Thesis](https://github.com/aknipey/Honours-Thesis).

## Tool: 8700340 passive probe

From the ROM, the markers are at (0,0,0), (0,0,50), (0,25,100) and (0,−25,135) mm, facing +x.
**The reported pose is the first marker, not the tip**, so calibrate once (stop `run` first: one
client at a time):

```bash
python3 -m ndi_tracking pivot   # tip in a divot, swivel the probe ~30 deg all round, 20 s
python3 -m ndi_tracking axis    # tip in the divot, twist the handle about the shaft (+-60 deg), 20 s
```

The pivot residual (~0.9 mm here) is mostly the camera's depth error magnified by the 160 mm tip lever. It
changes with position and tilt, so a single careful attempt can't remove it. To average it out, do 3–5 `pivot`
attempts with the divot block moved around the working area and different starting leans, then
`python3 -m ndi_tracking pivot --combine`. That fits one tip to all the attempts in `calibrations/raw/` (or the
CSVs you list), each with its own pivot point. Delete bad attempts from `calibrations/raw/` first.

Once the probe is mounted on the robot, `python3 -m ndi_tracking mount` (rotate UR wrist 3 ±60° while it records)
reports the angle between the probe shaft and tool0 z, and how far the probe tip is from that axis line. The
teleop's RCM mode assumes the shaft *is* tool0 z, so any misalignment shows up as RCM error. Nothing is saved.

Results are saved to `calibrations/8700340.json` and used by every later run. Without `axis`, the shaft is
assumed to run from the tip to the first marker. The RCM error is very sensitive to the axis: 1° off at
60 mm depth is about 1 mm of error. By hand is fine: the fit uses the direction that stays steady, so a
few degrees of wobble gives well under 1 deg of axis error. On the robot, rotate UR wrist 3 instead.

## Robot tool tip for the teleop's RCM (`robot-tcp`)

Finds the robot tool's tip in **tool0** (the kinematics the teleop uses, TF `base_link -> tool0`) and prints
`TOUCH_RCM_SHAFT_START / _DIRECTION / _LENGTH` for `teleop_settings.py`. Needs the robot driver running and ROS
sourced (same `ROS_DOMAIN_ID`).

1. Fix a divot where the robot can reach it, and put the **tool tip** in it.
2. Run one of these (stop `run` first: one Vega client at a time):

   ```bash
   python3 -m ndi_tracking robot-tcp --wand-is-tool    # the wand is the tool
   python3 -m ndi_tracking robot-tcp                   # wand strapped to some other tool
   python3 -m ndi_tracking robot-tcp --no-ndi          # no wand at all
   ```

3. For 40 s (`-d`), hand-guide the arm (UR freedrive) through as many tilts as you can, about 30–45° each way.
   Keep the tip in the divot and the wand visible.

What you get:

- **Tip in tool0** from the robot alone: the pivot fit on tool0 poses. The residual and per-quarter checks show slips,
  or TF kinematics that don't match the arm.
- **With the wand visible**, a hand-eye fit (tool0 → wand, NDI delay found automatically). The camera-pose spread
  shows whether the fit is consistent.
  - `--wand-is-tool`: the shaft direction in tool0 comes from the wand's calibrated axis (run `pivot` + `axis`
    first). The wand tip, mapped into tool0, is checked against the robot's tip.
  - Always: writes `calibrations/robot_tool.json` (`--save-calib`). `run --calib robot_tool` then tracks the
    **robot tool's** tip and shaft through the strapped-on wand.
- Without `--wand-is-tool`, the direction is taken from a holder point on tool0 z (`--holder-z`, default 0.05 m) to
  the tip. `--shaft-length` (default 0.262 m) sets LENGTH; START = tip − LENGTH × DIRECTION.

Results go to `calibrations/robot_tcp.json` and the raw data to `calibrations/raw/robot_tcp_*.npz` (re-analyse
with `--file`).

### With a reflective ball on the tool tip (`--tip-ball`)

No divot needed. Stick a passive marker ball on the tip, keep the wand on the tool, and run:

```bash
python3 -m ndi_tracking robot-tcp --tip-ball --wand-is-tool   # drop --wand-is-tool if the wand is strapped to another tool
```

For 40 s, freedrive the arm anywhere the camera sees both the wand and the ball, with as much rotation as you can
(30–45° each way about all three axes). The Vega sees the ball as a stray marker. The one that stays fixed in the
wand frame is taken as the ball, and its centre in tool0 is fitted from the robot kinematics alone. The hand-eye fit gives a
second estimate as a check. The **tip is the side of the ball facing the wand markers**: centre − `--ball-radius`
(default 5.75 mm, a standard 11.5 mm NDI sphere) along the shaft. The output also shows how far this tip is from the
current tip in MacroMicro's `teleop_settings.py`, and from the wand's calibrated tip.

The wand's own calibration (`calibrations/<ROM>.json`) is only read. `--save-calib` refuses the wand's name.

## Running

```bash
python3 -m ndi_tracking run          # keeps tracking until q / Ctrl+C
```

Live status line: rate, tip position, marker fit, and RCM error/depth when an RCM point is set.

| key | |
|---|---|
| `s` / space | start / stop a standalone recording → `recordings/<date_time>/` |
| `r` | set the RCM point at the current tool tip (averaged over 0.1 s) |
| `c` | clear the RCM point |
| `q` | quit |

The same from another terminal or script: `python3 -m ndi_tracking ctl start [name] | stop | toggle | rcm [x y z] | clear | status`.
Recordings started from a terminal (key or `ctl`) always go to `recordings/<date_time>[_<name>]/` in this
package; only the GREY button's recordings go to MacroMicro's `NEW_RUN`.

Other `run` options: `--record` starts a recording straight away, `--rcm x,y,z` sets a fixed RCM point (mm, camera frame), and `-d N` quits after N s.

### Linked to the MacroMicro teleop

With ROS 2 sourced, `run` also (`--ros auto`, the default):

- **GREY button recording**: follows `/trajectory_recorder/recording` and records into the same run
  folder (`logs/NEW_RUN`), with `t` on the recorder's clock. A standalone recording in progress is
  stopped first.
- **RCM mode (hold WHITE + click GREY)**: follows `/ur16e_touch_teleop/rcm_active`. Turning it on latches
  the NDI-measured tool tip as the RCM point, at the same moment the teleop latches its own. Turning it
  off clears it. If RCM is already on when `run` starts, its point is unknown; press `r` or re-enter RCM.
- **Publishes** `/ndi_tracking/tip` (PointStamped, frame `ndi`, mm) and `/ndi_tracking/rcm_error` (Float64, mm).

## Files (per recording folder)

| file | |
|---|---|
| `NDI_PATH.csv` | every frame: `t`, `unix_time`, `frame`, `visible`, `tip_*`, `axis_*` (shaft unit vector, tip → handle), raw pose, `fit_error`, `rcm_active`, `rcm_*`, `rcm_error`, `rcm_depth` |
| `NDI_EVENTS.csv` | start / stop / rcm_set / rcm_clear, with source and point |
| `NDI_META.yaml` | tool, calibration, rate, and **RCM metrics per segment** |
| `NDI_PLOT.png` | written at stop (same as `view`) |

All positions are in mm, in the camera frame (the Vega's gravity vector shows that its +x is down).

### RCM metrics

For each RCM segment (RCM point set, same point):

- **`error_*`**: distance from the RCM point to the shaft line (mean / rms / p95 / max, % within 1 and 2 mm).
  This is the main number.
- **`depth_*`**: how far the tip is past the RCM point along the shaft (+ = inserted).
- **`shaft_swing_deg`**: how much the shaft pivoted. The error only means something if it did.
- **`fitted_pivot_*`**: the point closest to all the shaft lines, i.e. where the tool actually pivoted, plus its
  distance from the set RCM point. A large offset with a small error about it means the robot pivoted
  cleanly but about the wrong point.

## Viewing

```bash
python3 -m ndi_tracking view              # latest recording (standalone or MacroMicro NEW_RUN)
python3 -m ndi_tracking view <folder>...  # specific ones, e.g. logs/V7/SomeRun
python3 -m ndi_tracking view --list       # all recordings found
```

The view shows tip x/y/z over time (RCM segments shaded), RCM error with 1/2 mm lines, depth, a 3D tip
path with shaft lines, the RCM point and fitted pivot, and the metrics.

## Other

- `python3 -m ndi_tracking check`: loads every ROM in `ndi_tracking/roms/` and prints what the camera
  sees (stray markers, their spacings, which tool tracks).
- ROMs are NDI's standard passive tools, from
  [PlusToolkit/PlusLibData](https://github.com/PlusToolkit/PlusLibData/tree/master/ConfigFiles/NdiToolDefinitions).
- Tracking rate: `run -r 20|30|60` (default 60).
