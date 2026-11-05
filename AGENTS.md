# Zusammenarbeit in diesem Projekt

Vor jeder Änderung `AGENT_HANDOFF.md` lesen und den aktuellen Git-Arbeitsbaum prüfen.
Die ausdrücklichen Nutzeranweisungen haben Vorrang vor dieser Datei.

- Moduldesign (Zellwahl, S/P, Abmessungen und Spannungsgrenzen) nicht ohne ausdrücklichen Nutzerauftrag ändern.
- Keine Änderungen des anderen Assistenten überschreiben oder zurücksetzen. Bei Abweichungen zuerst den Diff lesen.
- Nach jeder zusammengehörigen Änderung einen Eintrag in `AGENT_HANDOFF.md` ergänzen: Datum, Autor (Codex/Claude),
  Auftrag, Dateien, Verhalten, ausgeführte Prüfungen, Ergebnisse, offene Punkte und nächster sinnvoller Schritt.
- Fremde Einträge nicht als eigene Arbeit ausgeben. Historie nicht löschen. Aktuellen Status knapp aktualisieren.
- Ungeprüfte Annahmen ausdrücklich markieren. Ein bestandener Test ist kein Nachweis, dass das Modell physikalisch validiert ist.
- Die Übergabedatei ist das gemeinsame asynchrone Protokoll; sie sendet keine Nachrichten und startet keinen anderen Agenten.
