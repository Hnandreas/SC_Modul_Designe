# SC_Modul_Designe

Auslegung von Supercap-Modulen für hybride Energiespeicher (HESS) nach dem Verfahren der Dissertation
(Sec. *Supercapacitor Module Design for Hybrid Robotic Systems*, Flussdiagramm `Skript.pdf`,
Anhang A01 *Thermal Verification*, Anhang A02 *Energy-Equivalent Capacitance*) und IECON 2023
(Hanschek et al., DOI 10.1109/IECON51785.2023.10311944).

Drei Ebenen, alle über dieselbe Rechenlogik:

1. **IECON 2023 (legacy)**: exakte Reproduktion des alten MATLAB-Ablaufs (Datenblattwerte).
2. **Erweiterung C0 und k**: energieäquivalente Kapazität C_eq,E aus der eigenen Charakterisierung.
3. **Thermik**: P_T, transiente Prüfung mit R_th und C_th, Emulation des Profils als Endlosschleife.

Jeder Schritt, für den Daten fehlen, wird übersprungen und im Status angezeigt, statt abzubrechen.

## Schnellstart (macOS)

```bash
cd SC_Modul_Designe
./setup_env.sh        # einmalig: legt .venv an und installiert requirements.txt (Python >= 3.9)
./run_gui.sh          # startet die GUI im Browser (http://localhost:8501)
```

Alternativ `run_gui.command` im Finder doppelklicken.

Ohne GUI:

```bash
source .venv/bin/activate
python -m sc_module_design.cli --list
python -m sc_module_design.cli --preset iecon2023_legacy          # rechnet und exportiert nach outputs/
python -m sc_module_design.cli --preset thesis_characterized --out outputs/thesis
python -m pytest -q                                                # Regressionstests
```

In Python oder Jupyter:

```python
from sc_module_design.config import load_preset
from sc_module_design.pipeline import run_design
res = run_design(load_preset("thesis_characterized"))
res.candidates["profile_braking"]          # pandas DataFrame aller Kandidaten
res.steps["2"].summary                     # Status und Ergebnis jedes Flussdiagramm-Schritts
```

## GUI

| Tab | Inhalt |
|---|---|
| Übersicht | Flussdiagramm mit Status je Schritt (grün gerechnet, gelb Warnung, grau übersprungen), Kennwerte, Warnungen |
| 1 Anforderungen & Profil | Profil wählen/hochladen, Skalierung, Zeitfenster, manuelle Lastwerte, Filter (Moving Average, Peak-Shaver, Tiefpass, Butterworth, Rate-Limiter, eigener Ausdruck, Plugins), Bremsenergie, Peak-Statistik, FFT, Energieexkursionen, **Filtervergleich** |
| 2 Spannungen & Lebensdauer | U_Bus, U_C0max, Lebensdauer-Derating (Arrhenius), S, U_C0N, U_C0min aus P_SC,max / I_C0max oder fest, U_C0inM, Abbildungen wie in der Diss |
| 3 Kapazität | L_EOL, Szenarien (Module 1: Profil, Module 2: mit Bremsenergie), C_req |
| 4 Zellauswahl & Trade-off | Kapazitätsquelle (Datenblatt / C_eq,E / C_N), ESR-Quelle, Volumenformel, Thermik-Modus, Tabelle aller Kandidaten (P_C, P_I, P_T, Modulwerte), Picks, Pareto, Score, Abbildung wie IECON, Vergleich mit der IECON-Tabelle, LaTeX-Tabelle |
| 5 Zelldetail | eine Zelle anklicken und mit eigenen Optionen rechnen (Fenster, Messstrom, Datensatz, ESR, Thermik inkl. eigener R_th/C_th, P mindestens), Rechenweg, C(U)-Plot, Temperaturverlauf |
| 6 Emulation | gewähltes Modul im HESS: Profil endlos wiederholt bis periodischer Beharrungszustand, SOC-Regler für das Nachladen, Spannungs- und Stromgrenzen, ESR-Verluste, Zelltemperatur, BOL/EOL, **Live-Ansicht als Endlosschleife** |
| 7 Datenbank | alle Tabellen ansehen und bearbeiten (speichert CSV, baut SQLite neu), SQL-Abfragen, Quellen |
| 8 Export | Ordner unter `outputs/` mit inputs.yaml, summary.json/.md, Kandidaten-CSV, `tab_results.tex`, Abbildungen (PDF/SVG/PNG im Diss-Stil) |

Presets (Sidebar) laden und speichern komplette Eingaben als YAML in `data/presets/`.

## Flussdiagramm und Code

| Schritt | Gleichung (Diss) | Code |
|---|---|---|
| Application Requirements | Tab. load_metrics, Eq. movave, DChmax, Chmax | `profile.py`, `splits.py`, `energy.py` |
| 1 U_C0max, U_rated, Derating, S | voltagelife, seriescells, ratedmodul | `equations.u_cell_max_lifetime`, `n_series` |
| 2 U_C0min | currentlimit | `equations.u_c0min_from_current` |
| 3 U_C0inM | Intermediate | `equations.u_c0inm` |
| 4 C_req | cap6 | `equations.c_required` |
| 5/6 Kandidat, p = max(P_C, P_I) | parallel_capacitance, parallel_current, energy_equivalent_capacitance2 | `selection.evaluate_candidate`, `equations.c_eq_energy` |
| 7 bis 10 Thermik | app_thermal_rms_limit, app_parallel_thermal, app_transient_thermal_model | `thermal.py`, `selection.py` |
| 12 Trade-off | Tab. results, Fig. modules Cells | `tradeoff.py`, `plotting.fig_tradeoff`, `report.latex_results_table` |
| 13 Balancing | (Schnittstelle) | `balancing.py` |
| Thermische Simulation | Endlosschleife | `emulation.py` |

Ablaufsteuerung mit Status je Schritt: `pipeline.run_design`.

## Optionen, die übersprungen oder erweitert werden können

- **Profil fehlt**: manuelle Werte (P_peak, P_avg, E_DCH,max, E_CH,max) im Preset oder in Tab 1.
- **Kapazität**: `datasheet` (IECON), `ceq_M1` (C0, k; Fenster 90-20 % oder 80-40 %; 90-70 % ist nur für ESR geeignet und wird nicht angeboten), `cn_M2`. Ohne Charakterisierung fällt jede Zelle automatisch auf das Datenblatt zurück (steht in der Spalte `C_source`).
- **ESR**: Datenblatt DC (IECON), Datenblatt AC oder gemessen (Methodenreihenfolge M4, M3, M2, M1, nur plausible Werte).
- **Thermik**: `skip`, `available_only` (Flussdiagramm: nur Zellen mit eigenen R_th/C_th), `proxy` (gleiche Bauform, sonst nächst größere charakterisierte Zelle, immer mit Warnung; eine größere Zelle hat meist kleineres R_th, die Temperatur wird dann eher unterschätzt). Im Zelldetail zusätzlich eigene Werte.
- **Modulstrom für die Thermik**: `energy_balance` (P_SC(t)/U(t), U aus Energiebilanz ab U_C0inM) oder `conservative_umin` (P_SC(t)/U_C0min).
- **Thermisches Profil**: `periodic` (Profil endlos, periodischer Beharrungszustand) oder `single_pass` (ein Durchlauf ab Gleichgewicht, wie Anhang A01).
- **Energieauswertung**: `zero_crossing_legacy` (MATLAB-Algorithmus IECON), `zero_crossing_exact`, `cumulative_swing` (Spannweite der kumulierten SC-Energie, erfasst aufeinanderfolgende Pulse).
- **Volumen**: `corrected` oder `legacy_iecon`.

## Erweitern

- **Neue Zelle**: Zeile in `data/database/cells.csv` (oder Tab Datenbank), `candidate = True`. Pflicht: `c_nom_F, u_rated_V, esr_dc_mOhm, i_max_A, lifetime_series_id`. Für den Trade-off: Masse, D, L, Preis. Fehlende Pflichtfelder: Zelle wird übersprungen und gemeldet.
- **Charakterisierung**: Zeilen in `capacitance_fits.csv`, `esr_measurements.csv`, `thermal_params.csv` mit derselben `cell_id`. Datenblatt-R_th/C_th gehen auch direkt in `cells.csv`.
- **Eigener Filter**: Datei in `plugins/` ablegen (Beispiel `plugins/example_split_plugin.py`), erscheint automatisch in GUI und CLI. Für schnelle Versuche: Strategie *Custom (Python expression)* in Tab 1.
- **Balancing**: Modell mit `@register_balancing` in `sc_module_design/balancing.py` registrieren.

## Daten und Git

- Alles wird committet außer `data/profiles/private/` (eigene Messprofile, Uploads), `outputs/`, `Old/` und der automatisch gebauten `sc_database.sqlite`.
- Das Roboterarm-Profil liegt als `data/profiles/private/robot_arm_profile.npz` (149 s, 0.1 ms, in W) nur lokal. Fehlt es, rechnen die Presets mit den veröffentlichten Werten weiter.
- Öffentliches Vergleichsprofil: `data/profiles/public/legged_robot_LeggedRobotMPC2025.csv`.
- Quellen jeder Tabelle: `data/database/SOURCES.md`.

## Hinweise zur Reproduktion von IECON 2023

Mit dem Preset `iecon2023_legacy` werden E_DCH,max = 1214.1 J, E_CH,max = 557.0 J, U_C0inM = 43.8 V / 41.9 V,
S = 23, C_req = 4.165 F / 5.183 F sowie alle P/S-Konfigurationen, Massen, ESR-Werte und die Abbildung
*modules Cells* exakt reproduziert. Abweichungen der gedruckten Tabelle:

1. **Volumen/Dichte**: das alte Skript rechnet `V = P pi D^2 L` (Durchmesser als Radius, Faktor S fehlt). Das Volumen ist um 23/4 = 5.75 zu klein, die Dichte von 9.4 kg/dm^3 entspricht physikalisch etwa 1.6 kg/dm^3. Standard im Tool ist die korrigierte Formel.
2. **Kosten**: mit den Preisen aus `Workspace_Supercaps.mat` ergeben sich 325.68 / 200.10 / 407.10 / 200.10 / 246.56 EUR statt 360 / 182 / 450 / 183 / 298 EUR. Für Module 1 ist mit diesen Preisen die 35-F-Zelle (185.61 EUR) günstiger als die 40-F-Zelle.
3. **U_C0min**: 2643 W / 77.7 A = 34.0 V, gerechnet wurde mit 33 V (bei 33 V fließen 80.1 A).
4. **Bremsleistung**: IECON-Folie 1650 W (t = 1.2 s), Diss 1980 W (t = 1.0 s). Im Tool über `t_brake_s` einstellbar.
5. **Moving Average**: das MATLAB-Skript nutzt `conv(...,'same')` plus `circshift(-N/2)`, also ein vorausschauendes Fenster. Gl. (movave) der Diss beschreibt dagegen den kausalen Mittelwert. Kausal ergibt E_CH,max = 972 J und C_req = 4.77 F statt 4.17 F (Option `alignment` im Filter).
6. **Peak-Grenze 1486 W** (IECON-Folie 13) ist der obere Zaun Q3 + 1.5 IQR mit MATLAB-Quartilen, nicht der Whisker.

## Struktur

```
app/                 Streamlit-GUI (app.py, widgets.py)
sc_module_design/    Rechenkern (config, database, profile, splits, energy, equations,
                     selection, thermal, tradeoff, emulation, pipeline, plotting, report, cli)
data/database/       CSV-Datenbank + SOURCES.md (SQLite wird automatisch gebaut)
data/presets/        YAML-Presets (iecon2023_legacy, thesis_datasheet, thesis_characterized, ...)
data/profiles/       public/ (in git), private/ (nicht in git)
plugins/             eigene Filter (automatisch geladen)
tests/               pytest
outputs/             Exporte (nicht in git)
```
