# Data sources

Every row of every CSV carries a `source` column. Summary:

| Table | Content | Source |
|---|---|---|
| `cells.csv` | 16 Vishay 235 EDLC-HVR cells (3.0 V, 5 to 100 F): C, ESR DC/AC, I_max, mass, D x L, price, part number | `Workspace_Supercaps.mat` (`HVR235_matrix`) of the IECON 2023 design scripts, datasheet values |
| `cells.csv` | 25 F cells of Eaton, Kyocera, Maxwell, Sech, Wuerth (Dataset d DUTs) | nominal max. ESR from Diss Tab. `esr_summary_25F`; datasheet fields still missing, therefore `candidate = False` |
| `lifetime_series.csv` | Arrhenius parameters (U0, T0, t0 = 2000 h, dU = 0.2 V, dT = 10 K) for 235 EDLC-HVR and 225 EDLC-R | Tatschl 2023 (`langere-lebensdauer`), lifetime block of the IECON 2023 script |
| `capacitance_fits.csv` | C0, k (model M1), C_N (model M2) per cell, dataset a to d, voltage range, current | Diss Tabs. `compact_fit_table`, `C_overview_vishay_50F`, `C_overview_vishay_all_caps`, `model_comparison_methods`/`C_overview_25F`; OJIA 2026 |
| `esr_measurements.csv` | ESR per method M1 to M4, current, dataset, plausibility flag | Diss Tabs. `esr_summary`, `esr_vs_current_validation_set`, `esr_summary_allcaps`, `esr_summary_25F`; OJIA 2026, ECCE 2026 (ESR) |
| `thermal_params.csv` | R_th = 22.1 K/W, C_th = 19.7 J/K, Vishay 50 F, sensor 1 (top), condition c (6 A, 3.0 to 0.8 V), 30 °C, still air | Diss Tab. `thermal_sim_conditions`; ECCE 2026 (R_th/C_th) |
| `thermal_deltaT.csv` | Measured temperature rise per sensor position and condition a to d | Diss Tab. `thermal_result` |
| `reference_iecon2023_table.csv` | Values printed in the IECON 2023 summary table / Diss Tab. `results` | for comparison only |

Mapping of the characterized Vishay cells: 10 F, 25 F, 50 F of Datasets a to d are the HVR235 types
`VISHAY_HVR235_10F`, `VISHAY_HVR235_25F` (16 x 25 mm, 223051006000) and `VISHAY_HVR235_50F`
(confirmed by the author). Note: Tab. `esr_summary_allcaps` lists a nominal ESR of 36 mOhm for the
25 F cell, the 16 x 25 mm datasheet value is 34 mOhm.

Known deviations of the IECON 2023 table from this database (see README):
module costs (360/182/450/183/298 EUR) cannot be reproduced with the prices in the .mat file,
and the printed volumes/densities use the legacy volume formula.
