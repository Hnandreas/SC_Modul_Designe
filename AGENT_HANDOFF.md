# Gemeinsame Übergabe: Codex ↔ Claude

## Verbindliche Vorgaben des Nutzers

- Bestehendes Moduldesign nicht selbstständig ändern: keine neue Zellwahl, S/P, Abmessungen oder Spannungsgrenzen.
- Kontinuierlicher Betrieb über Profilperioden, kein Reset auf U_intermediate.
- Ladestrom in den SC positiv; Entladestrom negativ. U_Klemme = U_C + R I_SC. RMS ist stets nichtnegativ.
- Jede identische Zelle hat zur selben Zeit dieselben I²·ESR-Verluste; beide Stromrichtungen erwärmen.
- Nachladen über Spannungsreferenz und PI; inneren Stromregler idealisieren. Keine zusätzliche Ladeleistungsgrenze erfinden.
- **Only_PWM_Interlaved6 ist die richtige Reglerquelle**: Kp = 0,2, Ki = 0,5, PI-Ausgang ±5 A Gesamtstrom.
  Die ursprüngliche Erinnerung an 15 A wurde durch den Nutzer auf „kann auch ±5 sein“ präzisiert.
- **Aktuelle explizite Zeitbasis vom 04.11.2026:** f_sw = 100 kHz, PI alle 5 PWM-Perioden = 20 kHz, Ts = 50 µs.
  Die gefundenen 30 µs im alten Firmware-Initializer sind kein Grund, diese neue Vorgabe zu überschreiben.
- Leistungsprüfung entlang des realen Profils; oben direkt anklickbare Navigation und ausblendbare Prüfdetails.
- Energie darf oberhalb der Referenz verbleiben, bis Profil und Regler sie verändern; kein sofortiges künstliches Entleeren.

## Quellen

- Dissertation: `/Users/andreashanschek/Documents/GitHub/Dissertation-Andreas-Hanschek/chapters/06_interface_converter.tex`,
  Abschnitt PI Controller Implementation (ab Zeile 1622 im geprüften Stand).
- Firmware: `/Users/andreashanschek/Documents/Programming/STM32/Only_PWM_Interlaved6/Core/User/Src/user_control.c:54`.
  Der PI-Ausgang wird erst danach durch drei auf die Wandlerphasen aufgeteilt.
- Anti-Windup: `Only_PWM_Interlaved6/Core/User/Inc/user_config.h`, Funktion PI_Compute.
- Diese Dateien wurden nur gelesen. Die Firmware und die Dissertation werden durch diesen Auftrag nicht verändert.

## Aktueller Stand

Korrekturen durch Codex implementiert und in separater Arbeitskopie geprüft.
Die unten aufgeführten Dateien sind als geprüfter Stand in den Projektordner übernommen.
Ausgangscommit des geprüften Projektordners: `f5b60a9`, Arbeitsbaum war sauber.

## Architekturentscheidungen

- Die analytische Auslegung und ihre historischen Referenzzahlen bleiben unverändert. `current.py` bleibt
  explizit entladestrom-positiv für bestehende Berechnungen; die Emulation adaptiert diesen Helfer und liefert
  ladepositive signierte Ströme. Positive SC-Leistung bedeutet weiterhin Abgabe an den DC-Bus.
- `VoltagePI`: Kp ist A/V; Ki ist A/(V·s). Der Faktor Ts wird je tatsächlichem PI-Aufruf genau einmal angewendet.
  Kp wird nicht blind mit f_sw oder Aufrufzahl multipliziert. Bei Ts = 50 µs ist Ki·Ts = 0,000025 A/V.
- Für lange Profile werden PI-Aufrufe bei innerhalb eines Pflanzenbins gehaltenem Spannungsfehler analytisch
  zusammengefasst, einschließlich Firmware-Anti-Windup. Die Clock-Restzeit wird weitergeführt. Das ist
  eine gemittelte Pflanzen-/Feedback-Simulation, keine schaltgenaue 20-kHz-Pflanzensimulation.
- BOL/EOL-, Kapazitäts-, Layout- und thermische Auslegungsannahmen des vorhandenen Projekts werden nicht
  neu festgelegt. In der Emulation verwenden Strom und Joule-Verluste denselben tatsächlich gewählten ESR.
- An Energiegrenzen werden tatsächlicher SC-Strom, Klemmenleistung und Erwärmung gemeinsam begrenzt.
  Die nicht erfüllte SC-Anforderung wird der Batterie bilanziert und als Übernahme angezeigt.
- Periodische kausale Filterfortsetzung gilt nur für die Emulation; Legacy-Auslegung bleibt verfügbar.
- Die Live-Ansicht übernimmt denselben Zustand und dieselbe Konfiguration aus dem berechneten Lauf.

## Änderungsprotokoll

### 2026-11-04 — Codex — kontinuierliche Emulation und gemeinsames Protokoll

Auftrag: alle zuvor im Review gefundenen Emulationsfehler korrigieren und eine gemeinsame Datei für Codex/Claude anlegen.

Geänderte Bereiche:
- `sc_module_design/emulation.py`: Spannungs-PI, 100-kHz/5-Clock, ladepositiver Strom, vollständige Speicherbilanz,
  tatsächliche Grenzbegrenzung, RMS aus wirklichen Profilströmen, fortgesetzter PI-/Energiezustand, Konvergenzprüfung.
- `sc_module_design/emudesign.py`: periodischer Standardlauf statt Einzeldurchlauf, unverändertes S/P,
  periodische Filterbasis, konsistenter ESR der Wärmequelle und fortsetzbarer Emulator.
- `sc_module_design/splits.py`: gesonderte kausale periodische Filterfortsetzung für die Emulation.
- `app/app.py`, `sc_module_design/plotting.py`, `sc_module_design/formulas.py`: passende Anzeige/CSV,
  Referenz getrennt von Startspannung, Leistungsprüfung und Live-Fortsetzung.
- Tests: physikalische Regressionen und Vergleich des zusammengefassten PI mit der echten diskreten Rekurrenz;
  alte Tests, die Phantomstrom und fehlende ESR-Energie ausdrücklich erwarteten, entsprechend korrigiert.
- `README.md`, `AGENTS.md`, `CLAUDE.md`, diese Datei: Erklärung und Zusammenarbeit.

Tests/Ergebnis (vor Übertragung):
- `.venv/bin/python -m pytest tests/test_continuous_emulation.py tests/test_design.py -q -p no:cacheprovider`:
  **59 passed**, 44,57 s. Enthält die bestehenden analytischen Referenzprüfungen.
- Streamlit `AppTest`: Startseite, Emulationsseite, Berechnen und Live starten ohne Exceptions.
- Reales voreingestelltes Profil/Szenario: **S = 23, P = 3 unverändert**, 4 Perioden bis Konvergenz,
  Referenz 41,93167 V, Start und Ende des gezeigten periodischen Laufs 43,82804 V.
- PI-Aufrufe 20 kHz; Live setzte den Aufrufzähler von 11.920.000 auf 12.020.000 fort und behielt den EOL-Zustand.
- Alte CSV-Spalte `P_sc_cell_W` heißt nun `P_sc_terminal_out_W`: sie bezeichnet die gesamte Klemmenleistung,
  nicht eine einzelne Zelle. `I_sc_A` hat jetzt das vom Nutzer geforderte ladepositive Vorzeichen.
- Kein Commit oder Push beauftragt/ausgeführt. Keine Änderung an Dissertation oder STM32-Firmware.
- **Prüfung nach Übertragung im tatsächlichen Projektordner:** erneut **59 passed in 44.51s**.
- `git diff --check`: ohne Befund.
- Hashvergleich zum Ausgangsstand: `config.py`, `selection.py`, `equations.py`, `layout.py`, `pipeline.py`,
  `current.py` und `thermal_network.py` unverändert. Presets und Zell-/Messdaten wurden nicht bearbeitet.
- Die neuen Protokoll-/Anweisungsdateien und der neue Regressionstest sind absichtlich noch unversioniert;
  alle Änderungen sind lokal gespeichert, aber weder gestaged, committed noch gepusht.

### 2026-11-04 — Codex — fehlende Bus-/Tastgradgrenze korrigiert

Auslöser: Nutzer meldet SC-Klemmenspannung oberhalb der Busspannung; Screenshot zeigt ungefähr 40 V Bus.
Der vorherige kontinuierliche Modellstand war gespeichert, enthielt aber noch keine Spannungsreserve des Buck-Wandlers.

Änderungen:
- `emulation.py`: D_max = 0,98 aus den bestehenden inneren Interleaved-6-PI-Ausgangsgrenzen.
  Beim Laden I <= max(0, (D_max*U_Bus-U_C)/R); zusätzlich erreichbare Energie innerhalb eines Pflanzenbins
  berücksichtigen, auch bei R=0. Begrenzung wirkt auf tatsächlichen Strom, Klemmenleistung, Batterieanteil
  und ESR-Wärme – nicht durch nachträgliches Abschneiden der gezeichneten Spannung.
- Bereits vorhandene Energie oberhalb des erreichbaren Wandlerbereichs wird nicht künstlich entladen.
  Ein nicht regelbarer Portzustand wird ausgewiesen; eine unerreichbare Spannungsreferenz wird als solche gemeldet.
- `emudesign.py`: Diagnose einer unerreichbaren Referenz bzw. eines überschrittenen regelbaren Portbereichs.
- `app.py`, `plotting.py`: Wandlergrenze sichtbar, tatsächlicher Bus und konservative Stromreferenz getrennt,
  Diagnose im CSV, Klemmenspannung auch live. Veraltete Ergebnisse ohne neue Diagnosefelder werden verworfen
  und müssen neu berechnet werden, statt alte Kurven weiterzuzeigen.
- Physikalischer Wandlerbus ist `inp.req.U_bus_V`. `U_bus_min_V` bleibt eine optionale konservative
  Stromabschätzung im entsprechend markierten Diagramm; er ersetzt nicht heimlich die tatsächliche Busspannung.
- Moduldesign, Zell-/Messdaten, PI-Gains, 100-kHz/5-Clock und Firmware unverändert.

Prüfungen:
- **64 Tests bestanden (44,31 s)**, einschließlich neuer Fälle für Rekuperation, Rohprofil-Spitzen statt
  nur Mittelwert, Null-ESR, unerreichbare Referenz und schon oberhalb des Busses gespeicherte Energie.
- GUI-/Live-Test mit tatsächlichen 48 V: maximale SC-Klemmenspannung **47,04 V**, Begrenzung für 0,73 s,
  keine Portbereichsverletzung; S=23/P=3 unverändert, Live-Puffer mit zusätzlicher Klemmenspannung gültig.
- Zeitschrittvergleich mit tatsächlichen 40 V (periodischer Profilfall, ohne zusätzliches Bremsereignis):
  dt=10/5/1 ms: jeweils U_Klemme,max = **39,2 V**, 4 Perioden bis Konvergenz und 11.920.000 PI-Aufrufe.
  I_RMS = 9,24338 / 9,24754 / 9,25000 A; Differenz 10 zu 1 ms ca. 0,07 %.
  U_C,max = 39,198946 / 39,198932 / 39,198922 V. Das spricht gegen einen großen Zeitschrittfehler
  als Ursache der gemeldeten Überschreitung; der vorherige Modellstand hatte die Wandlergrenze nicht erzwungen.
- Bei 40 V echtem Bus ist U_ref≈41,93 V nicht erreichbar (D_max*U_Bus=39,2 V). Nicht ohne Nutzeranweisung
  Referenz, Busspannung oder Moduldesign verändern. Tatsächliche Bedeutung der 40 V wurde beim Nutzer nachgefragt.

Prüfung nach Übertragung im tatsächlichen Projektordner: 27 passed in 4.46s. `git diff --check` ohne Befund. GUI: konservative 40-V-Stromreferenz lässt tatsächlichen 48-V-Bus unverändert; alte Ergebnisobjekte werden erkannt und verworfen.

## Offene Punkte / Grenzen

- Die gemittelte elektrische Pflanze hält die interne Spannung und PI-Rückführung je Pflanzenbin näherungsweise konstant.
  Frequenzskalierung ist korrekt, die Diskretisierungsabweichung ist dennoch mit kleineren Bins zu prüfen.
- Ein beliebiges benutzerdefiniertes Filter-Plugin hat keinen bekannten internen Zustand; seine periodische
  Eignung muss der Plugin-Autor sicherstellen. Standardfilter erhalten eine periodische Fortsetzung.
- Ein höherer Startwert als U_intermediate ist erlaubt, wird aber nicht künstlich als garantiertes Ergebnis erzwungen.
- Bestehendes thermisches Modell bleibt ein Precheck mit den besprochenen Grenzen (keine neue Unterseiten-
  Wärmeableitung, keine neue temperatur-/spaltabhängige freie Konvektion). Keine Messvalidierung behaupten.
- Ergebnisse eines früheren Emulationslaufs müssen nach dem Update neu berechnet werden.

## Vorlage für den nächsten Eintrag

### Datum/Zeit (Europe/Vienna) — Codex oder Claude — Aufgabe
- Ausgangsstand / parallele Änderungen:
- Dateien und konkrete Verhaltensänderung:
- Prüfungen mit Ergebnis:
- Unveränderte Designparameter:
- Offene Punkte / nächster Schritt:

### 2026-11-04 — Codex — Nutzer bestätigt tatsächlichen Bus mit 48 V

- Die 40 V im alten Screenshot sind für diesen Aufbau falsch. Tatsächlicher Bus ist laut ausdrücklicher Nutzerbestätigung 48 V; keine Änderung am Moduldesign.
- Bestehende Trennung bleibt: physikalischer Bus aus U_bus_V (48 V im zugehörigen Preset), konservative Stromreferenz separat.
- app/app.py: Bei abweichenden Einstellungen alte Ergebnisse und Live-Zustände verwerfen und Darstellung beenden. Dadurch werden alte Kurven nicht mit neuen Buswerten kombiniert.
- Browserinventar enthält keine laufende Emulationsoberfläche; deren persönliche Sitzung konnte nicht geändert werden.
- Prüfung im Projekt: 27 Emulationsregressionstests bestanden (1,10 s); git diff --check ohne Befund. Keine neue physikalische Messvalidierung.

### 2026-11-04 — Codex — kein künstliches Bremsereignis in der Emulation
- Nutzer stellt klar: Ein zusätzlich eingefügtes Bremsereignis war nicht beauftragt.
- app/app.py: Einfüge-Schalter entfernt, Emulation nutzt ausschließlich das vorhandene Profil. Versionskennung aktualisiert, damit alte Kurven verworfen werden.
- emudesign.py: Auch ohne explizites include_braking-Argument wird kein Ereignis aus dem Szenarionamen abgeleitet. Explizite API-Option bleibt für gezielte separate Untersuchungen vorhanden.
- Rekuperation im Originalprofil bleibt erhalten. Moduldesign und analytische Auslegung unverändert.
- Prüfung: vollständige Testsuite 64 passed (39,81 s), git diff --check ohne Befund. Streamlit-AppTest: kein Einfüge-Schalter, kein Zusatzereignis (braking_at_s=None), tatsächlicher Bus 48 V, Lastminimum im Standardlauf +11,9052 W statt künstlicher negativer Bremsspitze.

### 2026-11-05 — Codex — optionale rote Vishay-Alterungskurven
- Nutzer: nur Ruggedized; Voralterung schaltbar, vorgealtert entspricht zweitem Messpunkt.
- aging.py: aus Screenshot näherungsweise abgelesene ESR/C-Punkte, stückweise linear; nur 85°C/85%rF/2,3V, keine Lebensdauerextrapolation. Zweiter Punkt näherungsweise 100 h. Vorgealtert auf diesen normiert, bestehende Anfangsparameter nicht doppelt abgewertet.
- app.py: optionaler Kurvenschalter, Voralterungsschalter und zusätzliche Prüfzeit; ersetzt feste EOL-Faktoren ausschließlich für Emulation; emulation/emudesign nutzen konsistente ESR/C-Faktoren auch thermisch. Moduldesign und analytische Dimensionierung unverändert.
- Kein automatischer Lebensdauer-Suchlauf oder Arrhenius-Parameter ohne Datengrundlage ergänzt.
- Nach Nutzervereinfachung: Schalter heißt Lebenszeit anpassen (Vishay), nur sichtbar bei ausschließlich Vishay-Auswahl. Einfache Gerade vom Start zum Endpunkt statt stückweiser Interpolation. Ohne Schalter unveränderte EOL-Prüfung. Voralterung schaltet normierten Start zwischen erstem und zweitem Punkt um.
- Prüfung vereinfachter Stand: 31 gezielte Tests bestanden (1,02 s), git diff --check sauber. Streamlit-AppTest: Schalter einschaltbar, 1000 zusätzliche Referenzprüfstunden vorgealtert ergeben C-Faktor 0,897025 und ESR-Faktor 1,490626; Modul bleibt 23S3P, keine App-Ausnahme.

### 2026-11-05 — Codex — verständliche Bezeichnung der Alterungszeit
- UI-Feld von Zusätzliche Prüfzeit in Alterungsdauer [h] umbenannt. Gemeint ist Alterung ab gewähltem Anfangszustand unter den angegebenen Referenzprüfbedingungen; keine zusätzliche elektrische Simulationsdauer. Rechenmodell unverändert. Textänderung geprüft, keine neuen Tests erforderlich.

### 2026-11-05 — Codex — korrigierter einfacher Arrhenius-Lebensdauercheck
- Nutzer korrigiert: Screenshot ist Accelerated-Test, keine direkte Betriebszeit. Gewünscht R/R0=1+t/Lref und Vergleich Wunsch zu Emulation; zusätzliche Batterieübernahme bei angeforderter SC-Entladung ist Lebensdauerfehler.
- aging.py ersetzt Screenshot-Digitalisierung komplett: Lref aus bestehender equations.lifetime_h und DB-Serienparametern bei eingestellter T_cell_C und konservativ oberer Zellspannung Umax/S. C linear bis bestehendem req.L_EOL, ESR linear bis 2. Keine erfundenen Voralterungszeiten oder Feuchteumrechnung.
- Ein einziger optionaler Schalter Lebensdauer prüfen; Stunden aus bestehendem req.t_life_h; alte Stunden-/Voralterungsregler entfernt, fester EOL-Schalter durch Hinweis ersetzt. Balkenvergleich Wunsch/Arrhenius/geprüft erreichbar und Abstand.
- Suche bis Arrhenius-EOL, BOL zuerst, dann sieben Bisektionsschritte unter Monotonieannahme; untere geprüfte Grenze plus Intervall ausgegeben. Bei erfülltem Wunsch wird Wunschzustand gezeigt, sonst letzter geprüfter funktionierender Zustand. Keine S/P- oder sonstige Moduldesignänderung.
- emulation.py: unmet_discharge_W erfasst zusätzliche Batterieübernahme pro Rohsample vor Mittelung (nur positive angeforderte Entladeleistung); vorgesehener Batterieanteil ist kein Fehler.
- Einfacher Precheck: konstante eingestellte Zelltemperatur, keine rückgekoppelte Alterungsintegration über thermische Historie. Quelle bestehender Arrhenius-Näherung bestätigt in https://www.vishay.com/docs/28578/_3vedlcsdeliverlongeruseflife.pdf ; Gültigkeitsgrenzen beachten, lineare R/C-Trajektorie bleibt Annahme.
- Nutzerpräzisierung: BOL-Ausfälle direkt verwerfen, alternativ wäre P zu erhöhen. Ergebnis setzt feasible=False und expliziten Verwerfungsgrund auch in Ergebnistabelle; keine automatische P-Änderung. Bei nicht erfülltem Lebensdauerwunsch ebenfalls negatives Gesamtergebnis, obwohl gezeigter früherer Zustand noch funktioniert.
- Nutzer bestätigt: fester EOL-Bezug exakt ESR ×2. Funktionale Lebensdauer darf früher enden; R/R0=1+t/Lref, daher Faktor 1,8 entspricht 0,8 Lref. EOL-Bezug wird nicht auf 1,8 umdefiniert.
- Validierung: komplette Suite 69 passed (39,99 s); nach Ergänzung eindeutiger Verwerfung erneut 5 Alterungstests bestanden (0,84 s). Abschließender Streamlit-AppTest ohne Ausnahme: Standardfall wird bereits bei BOL als nicht geeignet verworfen (zusätzliche Batterie-Spitzenleistung 329,34 W); S23/P3 unverändert. Eine Lebensdauerverkürzung behebt diesen BOL-Ausfall nicht. git diff --check sauber.

### 2026-11-05 — Codex — Fehlerkriterium der Lebensdauer korrigiert
- Nutzer präzisiert: normale Batterieaufteilung ist erlaubt. Nicht jede residuale Leistungsabweichung ist Lebensdauerfehler; entscheidend ist, ob der angeforderte SC-Strom die Modulstromgrenze aktiviert.
- aging.py nutzt tatsächliches t_current_limit_s aus letztem periodischem Zyklus statt unmet_discharge_W. Bestehende sonstige Modellfehler bleiben separat relevant (feasible). PI-Spannungsregler-Ausgangssättigung ist nicht die Modulstromgrenze.
- app.py Texte/Diagnose entsprechend korrigiert, Modellversion v6 verwirft Ergebnisse mit altem Kriterium. Keine Strom-/Spannungs-/S/P-Grenzen verändert. Vorige Interpretation des BOL-Verwerfens wegen 329,34 W ist hiermit überholt und war kein gültiger Stromgrenzennachweis.
- Plotkorrektur: current_limited und angeforderte signierte Rohstromextrema gespeichert; Diagramm markiert echte Begrenzungen und zeigt angeforderte Stromspitzen vor Begrenzung. Pflanzenbin-Mittelwerte können diese Rohspitzen verschleiern. Modellversion v7.
- Prüfung: 34 gezielte Tests bestanden (0,92 s); GUI-/Diagrammtest ohne Ausnahme und mit beiden Markierungen. Tatsächlicher Standardfall hat weiterhin reale Modulstrombegrenzung bei BOL: ca. 5,13–5,21 s, insgesamt 0,09 s betroffene Bins; angefordert bis 104,436 A, tatsächlich auf 77,7 A begrenzt. Daher BOL-Verwerfung in diesem Standardfall jetzt auf korrektem Kriterium begründet. Andere Nutzersitzungseinstellungen können abweichen. S23/P3 unverändert; git diff --check sauber.

### 2026-11-05 — Codex — unabhängiger PI und Filter, konfigurierbarer Takt und Nachladesuche
- Letzte Nutzerkorrektur maßgeblich: PI MUSS laden UND entladen dürfen. Beibehalten, kein unidirektionaler Regler eingebaut. Startwerte Kp=.2, Ki=.5, Grenzen -5/+5 A, PWM100kHz, Aufruf alle5=20kHz.
- Reglerwerte/Takt, beide PI-Stromgrenzen und MA-Fenster unabhängig in einem Expander einstellbar. Keine automatische Kp/Ki-Anpassung an Filterfenster. Tatsächliche Ts geht weiterhin genau einmal in Integral ein.
- Optionale Suche: Nachladegrenze vom eingestellten Wert in 1-A-Schritten bis10A; optional MA-Fenster 1/.75/.5/.25 des Startwerts. Filtervariation nur in kopierten Emulationseingaben, Auslegung und S/P bleiben unverändert. Negative PI-Grenze/Gains/Takt bleiben bei Suche gleich.
- Zuerst bestehendes Filterfenster, dann kleinere; erste funktionierende Variante gewählt, sonst beste geprüfte Variante nach gültigem Modellzustand/Stromgrenzendauer/Abweichung zum Filter-Batteriestrom. Kein globales Optimum behaupten. Lebensdauersuche wird danach einmal für gewählte Abstimmung ausgeführt.
- Diagramm zeigt zusätzlich Batterie-Filterreferenz ohne PI; durch PI angepasster Batteriestrom bleibt separat. Konstantbus macht Filterung von P und I äquivalent. Neue Modellversionv8, Live übernimmt die tatsächlich gewählte Konfiguration.
- Validierung: komplette Suite 73 passed (42,59 s). AppTest mit Kp=.3/Ki=.4, Grenzen -4/+6A, 80kHz/8 zeigt Ts100µs; angewählte Werte erreichen Emulator. Nachladesuche prüft6/7/8/9/10A, negative Grenze bleibt4A.
- Zweiter AppTest Filter+Nachladen: Fenster20/15/10/5s bei positiver Grenze10A getestet, negative Grenze bleibt5A, S23/P3 und ursprüngliches20-s-Auslegungsfenster unverändert. Kein Appfehler. In diesen Tests keine Variante erfüllt alle bestehenden sonstigen Kriterien; Suche weist das aus, behauptet keinen Erfolg. Zeitbasis-/Vorzeichentests beide Richtungen bestanden. git diff --check sauber.


### 2026-11-05, Claude: Arbeitsauftrag v6 vereinbart (noch NICHT begonnen)
- Status: nur geplant. Umsetzung erst nach ausdrücklichem "leg los" des Nutzers. Bis dahin keine Änderungen an Code, Presets oder Abbildungen durch Claude.
- Bitte parallel nichts an `plotting.py`, `style.py`, `report.py`, `tradeoff.py` und den Preset-/Export-Teilen von `app/app.py` ändern, ohne diesen Eintrag zu prüfen.
- Abbildungen (Thesis-Stil): alle exakt 8,0 cm breit; energy_excursions gemeinsame Legende unten (SC power nur einmal), Beschriftung (b) weg von der Spitze bei 125 s; lifetime_SC Legende unten wie Diss-Abbildung livetime_SC.pdf (Serie, T0, U_C0N1, ΔU_tC01 aus DB), Beschriftung 2.14 V ohne Kurvenüberdeckung; power_peaks Werte 1411 W / 2511 W waagrecht im Histogramm neben den gestrichelten Linien (Variante A); Modules_Cells zusätzlich wählbar: Zellzahl S x P, Modulhöhe.
- Bewertung: Kriterium "data" (gemessen statt Datenblatt) vorerst in allen Gewichtungen 0 und nicht angeboten (Nutzer: sonst unfair, später ggf. wieder).
- Export: speichert immer auch das Preset, Name Pflichtfeld, kein stilles Überschreiben; Preset enthält alle Einstellungen inkl. Emulation; YAML laden.
- Gesamtbericht als PDF (Englisch) beim Export: je Modul höchstens 3 Kandidaten, Module nebeneinander, Layout, alle Abbildungen beschriftet, Annahmen mit Herkunft; LaTeX mit matplotlib-Fallback.
- Designparameter (Zellwahl, S/P, Abmessungen, Spannungsgrenzen) und Codex-Emulationsstand bleiben unverändert.


### 2026-11-05, Claude: Arbeitsauftrag v6 umgesetzt
- Ausgangsstand: Codex-Stand vom 05.11. (unversioniert, 73 Tests grün). Keine Codex-Änderung zurückgesetzt; Emulation, PI, Alterung und Moduldesign unverändert.
- Abbildungen (`style.py`, `plotting.py`, `app/ui.py`): Export und GUI-Downloads exakt so breit wie die Figur (8,0 / 16,0 cm), nur die Höhe wird zugeschnitten (`style.exact_width_bbox`). Legenden unten lesen zeilenweise. energy_excursions: eine gemeinsame Legende unten, SC power nur einmal. lifetime_SC: Legende wie Diss-Abbildung (Series, T0, U_C0N1, ΔU_tC01 aus der DB), Höhe 7 cm, y-Raster 0,2 V. power_peaks: Werte waagrecht im Histogramm neben den Linien. Beschriftungen werden über `_place_free` an eine freie Stelle gesetzt (prüft Linien, Balken, Flächen, Marker, Texte und den Pfeil gegen andere Texte). Modules_Cells 8,0 cm, neue wählbare Größen `n_cells` und `module_height_mm` (neue Kandidatenspalte aus dem Layout).
- Bewertung (`tradeoff.py`): Kriterium "data" in `DISABLED_CRITERIA`, aus allen Vorlagen entfernt, auch bei eigener Gewichtung ignoriert; Code bleibt. Folge im Preset thesis_characterized: 50 F fällt in der Gesamtbewertung zurück (Modul 1 Rang 13/16), gewählt bleibt 50 F.
- Presets (`config.py`, `app/app.py`, `.streamlit/config.toml`): neues Feld `ui_state` (GUI-Einstellungen ohne Einfluss auf die Auslegung: emu_*, mc_*, mcr_*, ni_*, meas_*, net_tend, px, py). Werden in `ui_mem` gespiegelt, überleben Seitenwechsel und gehen ins Preset. `design_dict()` ohne ui_state, Name und Beschreibung als Cache-Schlüssel (auch im Emulations-cfg_key). Export: Name Pflicht, speichert immer das Preset, Überschreiben nur nach Bestätigung, gleicher Name für Preset und Ordner. Seitenleiste: Speichern sofort in der Liste und ausgewählt, YAML laden.
- Bericht (`report_pdf.py`, neu; `report.py` mit `standard_figures`/`save_standard_figures`; CLI `--report`): Englisch, LaTeX (latexmk/pdflatex, auch in /Library/TeX/texbin gesucht), sonst matplotlib. Inhalt: Eingaben, Schritte, je Modul höchstens 3 Kandidaten (gewählte zuerst), Modulvergleich, Layouts, Thermik, Emulation falls gerechnet, alle Abbildungen, Tool-Meldungen. Keine internen Herkunftsangaben (Nutzer: intern).
- Prüfungen: komplette Suite 86 passed (13 neue in `tests/test_v6_figures_report.py`). Streamlit-AppTest: Export mit/ohne Namen, Überschreibschutz, Kp 0,3 gespeichert und nach Laden eines anderen Presets und Zurückladen wiederhergestellt, Emulation rechnen und Export mit Emulationsbild im Bericht, Emulationsergebnis bleibt nach dem Speichern erhalten. git diff --check sauber.
- Hinweis: `outputs/thesis_characterized` wurde mit dem gespeicherten Preset neu exportiert (überschreibt den letzten GUI-Export des Nutzers, der mit Profilskalierung 0,95 gerechnet war). Testreste in `outputs/_apptest/` und `outputs/claude_check*/` (gitignored).
- Offen: YAML-Upload nur im Code geprüft (AppTest kann keine Datei hochladen). Kein Commit, kein Push.
- Nachtrag (Nutzer): lifetime_SC zeigt im Bild "Assumed cell temperature T = ... °C" (T aus Gl. voltagelife, req.T_cell_C), frei platziert; T_0 der Serie steht in der Legende.
