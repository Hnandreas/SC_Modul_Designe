# Load profiles

| Folder | Content | Git |
|---|---|---|
| `public/` | Profiles from public sources, e.g. `legged_robot_LeggedRobotMPC2025.csv` (reconstructed from the SVG figure of LeggedRobotMPC2025, one leg, about 20 Hz effective resolution) | committed |
| `private/` | Own measurements (robot-arm profile `robot_arm_profile.npz`, uploads from the GUI) | **not** committed (`.gitignore`) |

Formats: CSV with time [s] and power [W] (column names are detected automatically, e.g. `zeit,leistung`,
`Time_s,Power_W`), CSV with voltage and current (P = U I factor), or NPZ with `p` and `dt` (or `t`).
Positive power = load demand, negative = regeneration.

If the private profile is missing on a machine, the presets fall back to the published load values
(`manual_load` in the preset), so the IECON 2023 numbers can still be reproduced.
