"""Formula registry: what is calculated, how, and where each relation comes from.

Every entry: title, LaTeX formula, plain-text formula (for tooltips), origin (who/what it comes from) and source.
The GUI shows these in pop-ups next to metrics, tables and figures.
"""
from __future__ import annotations

from dataclasses import dataclass

ANDY_DISS = "Andy, Dissertation"
ANDY_MEAS = "Andy, eigene Messung"
ANDY_SPEC = "Andy, Spezifikation Netzwerkmodell"
ANDY_CHAT = "Andy, Vorgabe im Gespräch"
DATASHEET = "Datenblatt"
CLAUDE = "Annahme Claude"
LIT = "Literatur"
PHYS = "Physik (Grundgleichung)"
IECON = "IECON 2023 (Andy)"


@dataclass(frozen=True)
class Formula:
    title: str
    latex: str
    text: str
    origin: str
    source: str = ""


F: dict[str, Formula] = {}


def _f(key, title, latex, text, origin, source=""):
    F[key] = Formula(title, latex, text, origin, source)


# ---------------------------------------------------------------------- profile and split
_f("P_avg", "Mittlere Leistung", r"P_\mathrm{avg}=\frac{1}{T}\int_0^T P(t)\,\mathrm{d}t", "P_avg = 1/T ∫ P dt",
   ANDY_DISS, "Diss Tab. load_metrics")
_f("P_rms", "Effektivwert der Leistung", r"P_\mathrm{RMS}=\sqrt{\frac{1}{T}\int_0^T P^2\,\mathrm{d}t}",
   "P_RMS = sqrt(1/T ∫ P² dt)", ANDY_DISS, "Diss Tab. load_metrics")
_f("fence", "Grenze Leistungsspitzen", r"P_\mathrm{fence}=Q_3+1.5\,\mathrm{IQR}",
   "P_fence = Q3 + 1,5 IQR (Quartile wie MATLAB)", IECON, "IECON 2023, Folie 13")
_f("split", "Aufteilung Batterie / Supercap",
   r"P_\mathrm{Bat}(t)=\frac{1}{T_\mathrm{w}}\int_{t-T_\mathrm{w}}^{t}P(\tau)\,\mathrm{d}\tau,\quad "
   r"P_\mathrm{SC,Bus}=P-P_\mathrm{Bat}",
   "P_Bat = gleitender Mittelwert über T_w (bzw. gewählter Filter), P_SC,Bus = P - P_Bat", ANDY_DISS,
   "Diss Gl. movave")
_f("eta", "Wandler-Wirkungsgrad je Zeitschritt",
   r"P_\mathrm{SC}=\begin{cases}P_\mathrm{SC,Bus}/\eta_\mathrm{dch} & P_\mathrm{SC,Bus}>0\\ "
   r"P_\mathrm{SC,Bus}\,\eta_\mathrm{ch} & P_\mathrm{SC,Bus}<0\end{cases}",
   "Entladen: P_SC = P_SC,Bus / η_dch, Laden: P_SC = P_SC,Bus η_ch", ANDY_CHAT)
_f("E_dch", "Maximale Entladeenergie",
   r"E_\mathrm{DCH,max}=\max_k\int_{t_k}^{t_{k+1}}P_\mathrm{SC}^{+}\,\mathrm{d}t",
   "E_DCH,max = größte zusammenhängende Entladeenergie des SC (zwischen Nulldurchgängen)", ANDY_DISS,
   "Diss Gl. DChmax, IECON 2023")
_f("E_ch", "Maximale Ladeenergie",
   r"E_\mathrm{CH,max}=\max_k\int_{t_k}^{t_{k+1}}\left|P_\mathrm{SC}^{-}\right|\mathrm{d}t",
   "E_CH,max = größte zusammenhängende Ladeenergie des SC", ANDY_DISS, "Diss Gl. Chmax")
_f("P_sc_max", "Maximale SC-Leistung", r"P_\mathrm{SC,max}=\frac{\max P}{\eta_\mathrm{dch}}",
   "P_SC,max = P_peak / η_dch (bzw. P_peak - P_avg oder manuell)", ANDY_DISS, "Diss Abschnitt Module Specification")
_f("E_brake", "Bremsenergie", r"E_\mathrm{B}=\tfrac12 m v^2,\quad P_\mathrm{B}=\frac{2E_\mathrm{B}}{t_\mathrm{B}},"
   r"\quad E_\mathrm{B,SC}=E_\mathrm{B}\,\eta_\mathrm{ch}",
   "E = ½ m v², P = 2E/t_Brems, am SC E η_ch", ANDY_DISS, "Diss, IECON 2023 (Modul 2)")
_f("E_pre", "Precharge-Energie (erste Ladung)", r"E_\mathrm{pre}=-\int_0^T P_\mathrm{SC}(t)\,\mathrm{d}t",
   "E_pre = -∫ P_SC dt über ein Profil; Anlauf und Auslauf des Filters (Fenster am Rand)", ANDY_CHAT)

# ---------------------------------------------------------------------- voltages
_f("U_C01max", "Zellspannung für die Lebensdauer",
   r"U_\mathrm{C01max}=U_\mathrm{C0N1}-\left(\log_2\frac{t_\mathrm{life}}{t_0}-\frac{T_0-T}{\Delta T}\right)\Delta U",
   "U_C01max = U_C0N1 - (log2(t_life/t0) - (T0 - T)/ΔT) ΔU", ANDY_DISS, "Diss Gl. voltagelife (Arrhenius)")
_f("S", "Zellen in Serie", r"S=\left\lceil\frac{U_\mathrm{C0max}}{U_\mathrm{C01max}}\right\rceil",
   "S = ceil(U_C0max / U_C01max)", ANDY_DISS, "Diss Gl. seriescells")
_f("U_C0min", "Minimale Modulspannung",
   r"U_\mathrm{C0min}=r_Q\,U_\mathrm{C0max}\quad\text{oder}\quad U_\mathrm{C0min}="
   r"\frac{P_\mathrm{SC,max}}{I_\mathrm{C0max}}",
   "U_C0min = r_Q U_C0max (Vorgabe, Start 0,5) oder P_SC,max / I_C0max (Diss Schritt 2) oder fest",
   ANDY_DISS, "Diss Gl. currentlimit; r_Q-Vorgabe: Andy im Gespräch")
_f("r_Q", "Entladeverhältnis und nutzbare Energie",
   r"r_Q=\frac{U_\mathrm{C0min}}{U_\mathrm{C0max}},\quad\frac{W_\mathrm{used}}{W_\mathrm{total}}=1-r_Q^2",
   "r_Q = U_C0min / U_C0max, nutzbare Energie = 1 - r_Q²", ANDY_DISS, "Diss Gl. cap3, cap4")
_f("I_rec", "Wandlerstrom aus dem Profil",
   r"I_\mathrm{C0max,emp}=\frac{P_\mathrm{SC,max}}{U_\mathrm{C0min}},\quad I_\mathrm{RMS}\approx"
   r"\frac{P_\mathrm{SC,RMS}}{U}",
   "Spitze: P_SC,max / U_C0min; Dauer (RMS): P_SC,RMS / U_C0min (konservativ) bzw. / U_C0inM (typisch)",
   ANDY_DISS, "Diss Schritt 2 umgestellt; RMS-Abschätzung: Annahme Claude")
_f("U_C0inM", "Energiegewichtete Mittenspannung",
   r"U_\mathrm{C0inM}=\sqrt{\frac{E_\mathrm{DCH}U_\mathrm{C0max}^2+E_\mathrm{CH}U_\mathrm{C0min}^2}"
   r"{E_\mathrm{DCH}+E_\mathrm{CH}}}",
   "U_C0inM = sqrt((E_DCH U_C0max² + E_CH U_C0min²) / (E_DCH + E_CH))", ANDY_DISS, "Diss Gl. Intermediate")
_f("C_req", "Erforderliche Modulkapazität",
   r"C_\mathrm{req}=\frac{2W_\mathrm{used}}{L_\mathrm{EOL}\left(U_\mathrm{C0max}^2-U_\mathrm{C0min}^2\right)},"
   r"\quad W_\mathrm{used}=E_\mathrm{DCH,max}+E_\mathrm{CH,max}",
   "C_req = 2 W_used / (L_EOL (U_C0max² - U_C0min²))", ANDY_DISS, "Diss Gl. cap6")

# ---------------------------------------------------------------------- cells and module
_f("C_eq", "Energieäquivalente Kapazität",
   r"C_\mathrm{eq,E}=C_0+\frac{2k}{3}\frac{u_2^3-u_1^3}{u_2^2-u_1^2}",
   "C_eq,E = C0 + (2k/3)(u2³ - u1³)/(u2² - u1²), u = Zellspannungen bei U_C0min/S und U_C0max/S", ANDY_DISS,
   "Diss Anhang A02, C0/k aus eigener Charakterisierung")
_f("P_C", "Parallele Stränge aus der Kapazität", r"P_C=\left\lceil\frac{S\,C_\mathrm{req}}{C_\mathrm{Zelle}}\right\rceil",
   "P_C = ceil(S C_req / C_Zelle)", ANDY_DISS, "Diss Gl. parallel_capacitance")
_f("P_I", "Parallele Stränge aus dem Strom", r"P_I=\left\lceil\frac{I_\mathrm{C0max}}{I_\mathrm{Zelle,max}}\right\rceil",
   "P_I = ceil(I_C0max / I_Zelle,max)", ANDY_DISS, "Diss Gl. parallel_current")
_f("P_T", "Parallele Stränge aus der Thermik",
   r"P_T=\left\lceil\frac{I_\mathrm{Modul,RMS}}{I_\mathrm{Zelle,RMS,max}}\right\rceil,\quad "
   r"I_\mathrm{Zelle,RMS,max}=\sqrt{\frac{\Delta T_\mathrm{max}}{ESR\,R_\mathrm{th,eff}}}",
   "P_T = ceil(I_Modul,RMS / I_Zelle,RMS,max), I_Zelle,RMS,max = sqrt(ΔT_max / (ESR R_th,eff))", ANDY_DISS,
   "Diss Anhang A01, Gl. app_thermal_rms_limit")
_f("P_driver", "Was P bestimmt", r"P=\max\left(P_C,P_I,P_T,\dots\right)",
   "Treiber = der Term, der das Maximum liefert (Kapazität, Strom, Thermik RMS, Thermik transient, Netzwerk)",
   ANDY_DISS, "Diss Schritte 6, 8, 10")
_f("C_mod", "Modulkapazität und Modul-ESR", r"C_\mathrm{Modul}=\frac{P}{S}C_\mathrm{Zelle},\quad "
   r"R_\mathrm{Modul}=\frac{S}{P}ESR_\mathrm{Zelle}", "C_Modul = P/S C_Zelle, R_Modul = S/P ESR_Zelle", PHYS)
_f("reserve", "Kapazitätsreserve",
   r"\mathrm{Res}_\mathrm{EOL}=\frac{C_\mathrm{Modul}L_\mathrm{EOL}}{C_\mathrm{ben}}-1=\frac{C_\mathrm{Modul}}"
   r"{C_\mathrm{req}}-1,\quad\mathrm{Res}_\mathrm{neu}=\frac{C_\mathrm{Modul}}{C_\mathrm{ben}}-1,\quad "
   r"C_\mathrm{ben}=\frac{2W_\mathrm{used}}{U_\mathrm{C0max}^2-U_\mathrm{C0min}^2}",
   "Reserve EOL = C_Modul L_EOL / C_benötigt - 1 = C_Modul / C_req - 1; neu = C_Modul / C_benötigt - 1. "
   "Sehr große Reserve, wenn P durch Strom oder Thermik bestimmt ist.", ANDY_CHAT)
_f("E_usable", "Nutzbare Energie am Lebensende",
   r"E_\mathrm{nutz}=\tfrac12 C_\mathrm{Modul}L_\mathrm{EOL}\left(U_\mathrm{C0max}^2-U_\mathrm{C0min}^2\right)",
   "E = ½ C_Modul L_EOL (U_C0max² - U_C0min²)", PHYS)
_f("mass", "Modulmasse und Moduldichte",
   r"m=S P\,m_\mathrm{Zelle}+A_\mathrm{PCB}\,m''_\mathrm{PCB},\quad \rho=\frac{m}{V},\quad "
   r"V=B\,T\,(L_\mathrm{Zelle}+1.6\,\mathrm{mm}+h_\mathrm{Abstand})",
   "m = S P m_Zelle + A_PCB 0,35 g/cm², V = Breite Tiefe Höhe der Platine, ρ = m/V", CLAUDE,
   "Leiterplatten-Flächengewicht FR4 1,6 mm mit Kupfer: Annahme")
_f("cost", "Modulkosten", r"K=S P\,K_\mathrm{Zelle}", "K = S P Zellpreis", DATASHEET, "Preise aus Workspace_Supercaps.mat")
_f("P_loss", "Mittlere ESR-Verluste", r"P_\mathrm{v}=I_\mathrm{Modul,RMS}^2R_\mathrm{Modul}",
   "P_v = I_Modul,RMS² R_Modul", PHYS)
_f("layout", "Leiterplatten-Layout",
   r"p=d+s,\quad p_\mathrm{Reihe}=\begin{cases}p&\text{quadratisch}\\ p\sqrt3/2&\text{hexagonal}\end{cases},"
   r"\quad A=B\,T",
   "Teilung p = d + s, Reihenabstand p (quadratisch) oder p √3/2 (hexagonal), U-Pfad, Stecker im freien Platz",
   ANDY_CHAT, "Referenz EDLC-Layoutvergleich 60x1, d = 24 mm, s = 0,5 mm")
_f("score", "Bewertung", r"\mathrm{Score}=\frac{\sum_k w_k s_k}{\sum_k w_k},\quad s_k\in[0,1]",
   "Score = gewichteter Mittelwert der Kriterienwerte (100 % = bester Kandidat je Kriterium)", CLAUDE,
   "Gewichtung wählbar, Kriterien Kosten, Masse, Volumen, Verluste, Thermik, Zellen, Reserve, Messdaten")

# ---------------------------------------------------------------------- thermal
_f("th_first", "Thermisches Modell erster Ordnung (Einzelzelle)",
   r"C_\mathrm{th}\frac{\mathrm{d}\Delta T}{\mathrm{d}t}=P_\mathrm{v}-\frac{\Delta T}{R_\mathrm{th}},\quad "
   r"P_\mathrm{v}=i_\mathrm{Zelle}^2ESR",
   "C_th dΔT/dt = P_v - ΔT/R_th, P_v = i² ESR; R_th = 22,1 K/W, C_th = 19,7 J/K (50 F)", ANDY_DISS,
   "Diss Anhang A01; Werte: Diss Kap. 5, Tab. thermal_sim_conditions (eigene Messung)")
_f("R_th_eff", "Effektiver Wärmewiderstand im Modulblock (Lumped)",
   r"R_\mathrm{th,eff}=k\,R_\mathrm{th},\quad k=\frac{n\,A_\mathrm{Zelle}}{A_\mathrm{Box}},\quad "
   r"h=\frac{1}{R_\mathrm{th}A_\mathrm{Zelle}}",
   "R_th,eff = k R_th, k = n A_Zelle / A_Box: gleiche Wärmeübergangszahl wie die Einzelzelle, Wärme nur über "
   "die Boxoberfläche. Gibt nur einen Mittelwert, keinen Gradienten.", CLAUDE,
   "Lumped Capacitance / Newton-Kühlung: Incropera et al., Fundamentals of Heat and Mass Transfer, Kap. 5 "
   "(gültig für Bi < 0,1); Anwendung auf SC-Module nicht validiert")
_f("th_scale", "Skalierung auf andere Zellgrößen",
   r"R_\mathrm{th}=R_\mathrm{th,ref}\frac{A_\mathrm{ref}}{A},\quad C_\mathrm{th}=C_\mathrm{th,ref}\frac{m}{m_\mathrm{ref}}",
   "R_th ~ 1/Oberfläche, C_th ~ Masse (von der gemessenen 50-F-Zelle)", CLAUDE)
_f("esr_eol", "ESR am Lebensende", r"ESR_\mathrm{EOL}=2\,ESR_\mathrm{neu}", "ESR x 2 am Lebensende", CLAUDE,
   "übliches Lebensende-Kriterium (ESR +100 %)")
_f("net", "Thermisches Netzwerkmodell (Precheck)",
   r"C_\mathrm{th}\frac{\mathrm{d}T_i}{\mathrm{d}t}=P_\mathrm{Zelle}-\frac{T_i-T_{\mathrm{air},i}}{R_\mathrm{air}}"
   r"-\sum_{j\in\mathcal N_i}\frac{T_i-T_j}{R_\mathrm{gap}}-\dot Q_{\mathrm{rad},i},\quad "
   r"T_{\mathrm{air},i}=\frac{\sum_j T_j+(z-m_i+1)T_\infty}{z+1}",
   "Jede Zelle Wärmequelle, lokale Luft aus Nachbarn (Gleichgewichtung, gewählte Vereinfachung), Spaltleitung, "
   "Strahlung nach oben optional", ANDY_SPEC, "claude/thermal_network_model_spec.md; Literatur zu "
   "Netzwerkmodellen: Hijazi et al., IEEE TIE 59(2), 2012")
_f("R_gap", "Luftspaltwiderstand", r"R_\mathrm{gap}\approx\frac{s+r(2-\pi/2)}{2rL\lambda}",
   "R_gap = (s + r(2 - π/2)) / (2 r L λ), Längen in m", ANDY_SPEC, "MathWorks, Model Heat Exchange Between Cells")
_f("Q_rad", "Strahlung der oberen Kreisfläche",
   r"\dot Q_\mathrm{rad}=\varepsilon\sigma A_\mathrm{top}\left(T_K^4-T_{\mathrm{rad},K}^4\right)",
   "Q_rad = ε σ A_top (T⁴ - T_rad⁴), ε = 0,05 (Arbeitsannahme)", ANDY_SPEC, "COMSOL Surface-to-Ambient Radiation")

# ---------------------------------------------------------------------- emulation
_f("I_sc", "SC-Strom aus der Leistungsbilanz mit ESR-Abfall",
   r"P_\mathrm{Bus}=U_\mathrm{Bus}I_\mathrm{Bus}=U_\mathrm{SC}I_\mathrm{SC},\quad U_\mathrm{SC}=U_C-R I_\mathrm{SC}"
   r"\;\Rightarrow\;I_\mathrm{SC}=\frac{U_C-\sqrt{U_C^2-4RP}}{2R}",
   "P = (U_C - R I) I, also I = (U_C - sqrt(U_C² - 4RP)) / (2R)", ANDY_CHAT, "Leistungsbilanz am Wandler")
_f("P_limit", "Leistungsgrenze", r"P\le\frac{U_C^2}{4R}", "P_max = U_C² / (4R) (Anpassung, halbe Spannung am ESR)",
   PHYS)
_f("U_term", "Klemmenspannung", r"U_\mathrm{Klemme}=U_C-R_\mathrm{Modul}I_\mathrm{SC},\quad "
   r"R_\mathrm{Modul}=\frac{S}{P}ESR_\mathrm{Zelle}\;(\times2\ \text{am Lebensende})",
   "U_Klemme = U_C - R_Modul I_SC, R_Modul = S/P ESR_Zelle (x 2 am Lebensende)", PHYS)
_f("W_swing", "Energiehub und Kapazität aus der Emulation",
   r"W_\mathrm{swing}=\max E_\mathrm{SC}-\min E_\mathrm{SC},\quad C_\mathrm{req,emu}=\frac{2W_\mathrm{swing}}"
   r"{L_\mathrm{EOL}\left(U_\mathrm{C0max}^2-U_\mathrm{C0min}^2\right)}",
   "W_swing = max E_SC - min E_SC im periodischen Zyklus, C_req,emu wie C_req", CLAUDE,
   "emulationsbasierte Auslegung (Erweiterung)")
_f("P_rech", "Nachladeregler", r"P_\mathrm{rech}=K\left(E_\mathrm{Ziel}-E\right),\ |P_\mathrm{rech}|\le P_\mathrm{rech,max}",
   "P_rech = K (E_Ziel - E), begrenzt", CLAUDE)


def text(key: str, calc: str = "") -> str:
    """Tooltip text: formula, inserted values, origin."""
    f = F.get(key)
    if f is None:
        return calc
    out = f"**{f.title}**  \n{f.text}"
    if calc:
        out += f"  \n**Eingesetzt:** {calc}"
    out += f"  \n*Herkunft: {f.origin}{(', ' + f.source) if f.source else ''}*"
    return out
