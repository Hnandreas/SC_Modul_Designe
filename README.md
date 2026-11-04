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

Geführter Ablauf, eine Seite je Schritt des Flussdiagramms (Seitenleiste links, Status je Schritt):

| Seite | Inhalt |
|---|---|
| Start und Überblick | Flussdiagramm mit Status, gewähltes Modul, Kennwerte, Presets laden/speichern |
| 1 Lastprofil | Profil wählen/hochladen, Skalierung, Zeitfenster, manuelle Werte, Peak-Statistik, FFT |
| 2 Leistungsaufteilung | Filter (Moving Average, Peak-Shaver, Tiefpass, Butterworth, Rate-Limiter, eigener Ausdruck, Plugins), Filtervergleich, Wandler-Wirkungsgrad Laden/Entladen, Bremsereignis |
| 3 Spannungen und Lebensdauer | U_Bus, U_C0max, Derating, S, U_C0min, U_C0inM |
| 4 Kapazität | L_EOL, Szenarien (Module 1: Profil, Module 2: mit Bremsenergie), C_req |
| 5 Zellauswahl | Kapazitäts- und ESR-Quelle, **Leiterplatte** (Packung, Ausrichtung, Stecker), **Vorgaben** (Bauraum, Masse, Kosten), Tabelle aller Kandidaten, Layout-Vergleich quadratisch/hexagonal |
| 6 Thermik | Thermik-Modus, Wärmeabfuhr, ESR am Lebensende, Aufschlüsselung ΔT, ΔT je Kandidat |
| 7 Bewertung und Auswahl | Gewichtung (ausgewogen, Kosten, kompakt, robust, eigene), Rangliste, Abbildung wie IECON mit Moduldichte, interaktiver Vergleich, Auswahl übernehmen |
| 8 Emulation und Auslegung | Profil endlos im HESS, Strom aus der Leistungsbilanz mit ESR-Abfall, Zelltemperatur aus i² ESR und thermischem Tiefpass, P erhöhen bis alles eingehalten ist, Live-Ansicht |
| Zelldetail, Datenbank, Export | eine Zelle mit eigenen Optionen, Tabellen bearbeiten, Export (YAML, CSV, LaTeX, Abbildungen PDF/SVG/PNG) |

Konfigurationen außerhalb der Vorgaben (oder der thermischen bzw. P-Grenze) bleiben in den Tabellen sichtbar
(Spalte Status) und sind in allen Abbildungen grau.

### Emulationsbasierte Auslegung

Je Zeitschritt: Leistung am SC aus der Busleistung mit Wirkungsgrad (Entladen P_Bus / η_dch, Laden P_Bus η_ch),
Strom aus der Leistungsbilanz P = U_SC I_SC mit U_SC = U_C - R_ESR I_SC, also
I_SC = (U_C - sqrt(U_C² - 4 R_ESR P)) / (2 R_ESR) mit der Leistungsgrenze P ≤ U_C² / (4 R_ESR).
Aus dem emulierten Profil: Energiehub, C_req,emu, Spitzen- und RMS-Strom, Zelltemperatur aus
p_v = i_Zelle² R_ESR und dem thermischen Tiefpass erster Ordnung. P wird erhöht, bis weder U_C0min/U_C0max noch
die Strom- oder Leistungsgrenze erreicht werden und ΔT eingehalten ist. Standard ist die Prüfung am Lebensende
(C x L_EOL, ESR x 2).

### Leiterplatten-Layout

Die Zellen stehen auf einer Leiterplatte, P Zellen bilden eine Gruppe, S Gruppen in Serie. Der Serienpfad läuft
als U: vorne in der Mitte los, in der linken Hälfte nach hinten, in der rechten Hälfte zurück nach vorne. Plus
und Minus liegen vorne nebeneinander. Freie Gruppenplätze bleiben vorne in der Mitte, dort sitzen die Stecker
(bei Bedarf um 90° gedreht), damit die Platine nicht größer wird. Packung quadratisch oder hexagonal versetzt,
Ausrichtung auto, eher lang oder eher breit, Gruppenform automatisch. Masse = Zellen + Leiterplatte,
Volumen = Platine x (Zelllänge + Leiterplatte + Abstand), daraus die Moduldichte.

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
| Thermische Simulation | Endlosschleife, emulationsbasierte Auslegung | `emulation.py`, `emudesign.py`, `current.py` |
| Mechanik | Leiterplatten-Layout | `layout.py` |

Ablaufsteuerung mit Status je Schritt: `pipeline.run_design`.

## Optionen, die übersprungen oder erweitert werden können

- **Profil fehlt**: manuelle Werte (P_peak, P_avg, E_DCH,max, E_CH,max) im Preset oder in Tab 1.
- **Kapazität**: `datasheet` (IECON), `ceq_M1` (C0, k; Fenster 90-20 % oder 80-40 %; 90-70 % ist nur für ESR geeignet und wird nicht angeboten), `cn_M2`. Ohne Charakterisierung fällt jede Zelle automatisch auf das Datenblatt zurück (steht in der Spalte `C_source`).
- **ESR**: Datenblatt DC (IECON), Datenblatt AC oder gemessen (Methodenreihenfolge M4, M3, M2, M1, nur plausible Werte).
- **Thermik**: `skip`, `available_only` (Flussdiagramm: nur Zellen mit eigenen R_th/C_th), `proxy` (gleiche Bauform, sonst nächst größere charakterisierte Zelle, immer mit Warnung; eine größere Zelle hat meist kleineres R_th, die Temperatur wird dann eher unterschätzt). Im Zelldetail zusätzlich eigene Werte.
- **Modulstrom für die Thermik**: `energy_balance` (P_SC(t)/U(t), U aus Energiebilanz ab U_C0inM) oder `conservative_umin` (P_SC(t)/U_C0min).
- **Thermisches Profil**: `periodic` (Profil endlos, periodischer Beharrungszustand) oder `single_pass` (ein Durchlauf ab Gleichgewicht, wie Anhang A01).
- **Energieauswertung**: `zero_crossing_legacy` (MATLAB-Algorithmus IECON), `zero_crossing_exact`, `cumulative_swing` (Spannweite der kumulierten SC-Energie, erfasst aufeinanderfolgende Pulse).
- **Volumen**: `pcb` (Leiterplatten-Layout, Standard), `corrected` oder `legacy_iecon`.
- **Vorgaben**: maximale Breite, Tiefe, Höhe, Volumen, Masse, Kosten (`req.max_*`), jeweils optional.

## Erweitern

- **Neue Zelle**: Zeile in `data/database/cells.csv` (oder Tab Datenbank), `candidate = True`. Pflicht: `c_nom_F, u_rated_V, esr_dc_mOhm, i_max_A, lifetime_series_id`. Für den Trade-off: Masse, D, L, Preis. Fehlende Pflichtfelder: Zelle wird übersprungen und gemeldet.
- **Charakterisierung**: Zeilen in `capacitance_fits.csv`, `esr_measurements.csv`, `thermal_params.csv` mit derselben `cell_id`. Datenblatt-R_th/C_th gehen auch direkt in `cells.csv`.
- **Eigener Filter**: Datei in `plugins/` ablegen (Beispiel `plugins/example_split_plugin.py`), erscheint automatisch in GUI und CLI. Für schnelle Versuche: Strategie *Custom (Python expression)* in Tab 1.
- **Balancing**: Modell mit `@register_balancing` in `sc_module_design/balancing.py` registrieren.

## Daten und Git

- Alles wird committet außer `data/profiles/private/` (eigene Messprofile, Uploads), `outputs/`, `Old/` und der automatisch gebauten `sc_database.sqlite`.
- Das Roboterarm-Profil liegt als `data/profiles/private/robot_arm_profile.npz` (149 s, 0.1 ms, in W) nur lokal. Fehlt es, rechnen die Presets mit den veröffentlichten Werten weiter.
- Öffentliches Vergleichsprofil: `data/profiles/public/legged_robot_LeggedRobotMPC2025.csv`, mit den Presets
  `legged_robot_24V` und `legged_robot_48V` (Moving Average 3 s, I_C0max = 2 P_max / U_Bus).
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
app/                 Streamlit-GUI (app.py, ui.py, widgets.py)
sc_module_design/    Rechenkern (config, database, profile, splits, energy, equations, selection,
                     thermal, current, converter, layout, tradeoff, emulation, emudesign,
                     pipeline, plotting, style, report, cli)
data/database/       CSV-Datenbank + SOURCES.md (SQLite wird automatisch gebaut)
data/presets/        YAML-Presets (iecon2023_legacy, thesis_datasheet, thesis_characterized,
                     legged_robot_24V, legged_robot_48V)
data/profiles/       public/ (in git), private/ (nicht in git)
plugins/             eigene Filter (automatisch geladen)
tests/               pytest
outputs/             Exporte (nicht in git)
```
