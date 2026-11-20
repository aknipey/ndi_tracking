# WAND ONLY calibration: NDI probe 8700340, markers -> tip (2026-10-20)

The wand calibrated by hand, on its own (not on the robot). It gives the tip and the shaft
axis in the wand's marker frame (first marker = origin). Valid for this physical wand on any
mount, as long as the tip isn't bent or the markers moved.

- `8700340_wand_only.json`: copy of `calibrations/8700340.json` (the file `ndi_tracking` uses), saved 2026-10-20 13:05:12
  - tip offset: (-16.59, 1.55, -158.58) mm from the first marker
  - pivot residual: 0.93 mm rms
  - shaft axis (tip -> handle): (0.0028, -0.0057, 1.0000), 0.37 deg from the ROM z
  - axis twist range 143 deg, wobble 0.31 deg rms
- `pivot_8700340_*.csv`: raw `python3 -m ndi_tracking pivot` attempts (tip in a divot, swivel)
- `axis_8700340_*.csv`: raw `python3 -m ndi_tracking axis` attempts (tip in the divot, twist about the shaft)

Copies; the originals stay in `calibrations/raw/` (where `pivot --combine` looks).
To use this calibration again: `cp 8700340_wand_only.json ../8700340.json`.
