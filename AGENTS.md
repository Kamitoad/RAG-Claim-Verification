# AGENTS.md

## Geltungsbereich

Diese Datei enthält dauerhafte Arbeitsregeln für das gesamte Repository. Kurzlebige
Statusinformationen, bekannte Bugs, TODOs, offene Entscheidungen, zuletzt verifizierte
Befehle und nächste Meilensteine gehören in `CODEX_HANDOFF.md` oder eine andere
Statusdatei, nicht hierher.

## Projektzweck und fachlicher Umfang

Dieses Repository implementiert eine strukturierte und reproduzierbare
Python-Forschungsanwendung zur evidenzgestützten Verifikation vorformulierter atomarer
Faktenbehauptungen in einer geschlossenen Wissensdomäne.

Die Forschungsdomäne ist die vollständige Formel-1-Saison 2023, sofern der Nutzer den
Projektumfang nicht ausdrücklich ändert. Jede Behauptung erhält genau eines der Labels:

- `SUPPORTED`
- `REFUTED`
- `NOT_ENOUGH_EVIDENCE`

Der kontrollierte Hauptvergleich besteht aus:

1. einer LLM-Baseline ohne Retrieval;
2. RAG über einem sauberen Korpus;
3. RAG über demselben vollständigen sauberen Korpus plus reproduzierbar definiertem
   Rauschen.

Eingaben sind bereits atomisierte Claims. Das Projekt ist kein allgemeiner
Fake-News-Detektor.

## Ausdrückliche Nichtziele

Ohne vorherige Zustimmung des Nutzers nicht implementieren:

- automatische Claim-Extraktion aus Artikeln;
- allgemeine Websuche, Scraping oder automatische Dokumentbeschaffung;
- Verifikation ganzer Artikel oder Aggregation mehrerer Claims zu einem Artikelurteil;
- Web-UI, interaktives Wiki, REST-API oder Endnutzer-Ingestion;
- erfundene Konfidenzwerte oder Fallback-Labels;
- zusätzliche RAG-Frameworks, Retriever oder Provider nur auf Vorrat;
- breite Modell-Sweeps oder statistische Subsysteme ohne festgelegte Forschungsfrage.

## Architekturgrenzen

- Retrieval und Claim-Verifikation bleiben getrennte Schnittstellen.
- LightRAG-spezifischer Code bleibt auf `retrieval/lightrag_adapter.py` begrenzt.
- Andere Retriever implementieren das bestehende `Retriever`-Protokoll.
- Modellclients implementieren das bestehende `LLMClient`-Protokoll.
- Prompts liegen in versionierten Dateien unter `prompts/`; keine konkurrierenden
  Promptfragmente im Anwendungscode einbetten.
- Konfiguration, Manifestdaten, Modellantworten und persistierte Ergebnisse werden
  strikt validiert.
- Vor externen Aufrufen oder Schreiboperationen zuerst alle lokal prüfbaren Eingaben
  validieren.
- Fehlendes Evidenzmaterial, Retrievalfehler, Providerfehler und Parsefehler sind
  unterschiedliche Zustände. Betriebliche Fehler dürfen nicht in leere Evidenz oder
  `NOT_ENOUGH_EVIDENCE` umgedeutet werden.
- Ungültige Modellantworten erhalten höchstens den vorgesehenen begrenzten
  Reparaturversuch. Niemals ein Ersatzlabel erfinden.
- Auswertung gespeicherter Predictions muss ohne erneute Retriever- oder Modellaufrufe
  möglich bleiben.
- Persistierte Dateien atomar schreiben und bestehende Run-Verzeichnisse nicht
  überschreiben.

## Verzeichnisverantwortlichkeiten

- `configs/`: strikt validierte Benchmark- und Corpus-YAMLs.
- `data/manifests/`: technische Dokumentmetadaten als JSONL.
- `data/ground_truth/`: atomare Claims und Goldannotation als JSONL.
- `data/corpora/`: ausschließlich zulässige Forschungsdaten oder klar markierte
  synthetische Fixtures.
- `prompts/`: versionierte System- und User-Prompts samt Ausgabevertrag.
- `src/rag_claim_verification/models/`: Domänen- und Persistenzmodelle.
- `src/rag_claim_verification/ingestion/`: Manifestvalidierung, Dokumentladen und
  geschützte Index-Ingestion.
- `src/rag_claim_verification/retrieval/`: Retriever-Protokoll und Adapter.
- `src/rag_claim_verification/llm/`: Providerprotokoll, HTTP-Clients und strukturierte
  Ausgabevalidierung.
- `src/rag_claim_verification/verification/`: Promptaufbau und Claim-Verifikation.
- `src/rag_claim_verification/evaluation/`: Benchmarkorchestrierung, Metriken,
  Fehleranalyse und Reporting.
- `src/rag_claim_verification/utils/`: allgemeine Datei- und Hashing-Hilfen.
- `tests/unit/`: isolierte, schnelle Tests ohne Netzwerkzugriff.
- `tests/integration/`: komponentenübergreifende Workflows; externe Tests bleiben
  ausdrücklich markiert und opt-in.
- `runs/`: generierte, unveränderliche Experimentausgaben.
- `indices/`: generierte Retrieverindizes, keine Quelldaten.
- `CODEX_HANDOFF.md`: aktueller Stand, bekannte Grenzen, offene Entscheidungen und
  nächste Schritte; keine dauerhaften Arbeitsregeln duplizieren.

## Entwicklungsumgebung und kanonische Befehle

Python 3.11 oder neuer verwenden. Eine lokale virtuelle Umgebung im Repository anlegen:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

Die deklarierte optionale LightRAG-Integration nur installieren, wenn sie für die
Aufgabe benötigt wird:

```bash
python -m pip install -e ".[dev,lightrag]"
```

CLI starten:

```bash
python -m rag_claim_verification --help
```

Standardprüfungen:

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src
```

Formatierung anwenden:

```bash
python -m ruff format .
```

Externe Tests benötigen die dafür deklarierte optionale Umgebung und müssen mit
`RAGCV_RUN_EXTERNAL_TESTS=1` ausdrücklich aktiviert werden. Standardtests dürfen keine
externen oder kostenpflichtigen Dienste aufrufen.

## Coding-Konventionen

- `src`-Layout und bestehende Modulgrenzen beibehalten.
- Ruff-Konfiguration aus `pyproject.toml` einhalten; maximale Zeilenlänge 100.
- Öffentliche Klassen und Funktionen knapp und präzise dokumentieren.
- `pathlib.Path` für Dateipfade verwenden.
- Asynchrone APIs für Netzwerk-, Provider- und Retriever-I/O verwenden.
- Externe SDK-Details hinter kleinen, typisierten Adaptern kapseln.
- Abhängigkeiten durch Protokolle oder Factories injizierbar halten, damit Offline-Tests
  möglich bleiben.
- Keine neue Abhängigkeit einführen, wenn Standardbibliothek oder bestehende
  Abhängigkeiten die Aufgabe angemessen lösen.
- Keine domain- oder fixture-spezifischen Sonderfälle in generische Kernkomponenten
  einbauen.
- Keine Secrets, lokalen absoluten Pfade oder generierten Artefakte in Quellcode
  hardcodieren.

## Typisierung und Fehlerbehandlung

- Der mypy-Strict-Modus muss für `src` bestehen bleiben.
- Öffentliche Signaturen vollständig typisieren.
- `Any` nur an nachweislich dynamischen externen SDK- oder Serialisierungsgrenzen
  verwenden und möglichst früh in validierte Typen überführen.
- Für persistierte oder extern gelieferte Daten von `StrictModel` abgeleitete
  Pydantic-Modelle verwenden.
- Unbekannte Felder, ungültige Labels, leere Pflichtwerte und doppelte IDs ablehnen.
- Erwartete Fehler mit den bestehenden projektspezifischen Exceptiontypen oder einer
  passend abgegrenzten neuen Unterklasse ausdrücken.
- Fehlertexte müssen betroffene Datei, Datensatz-ID oder Zeilennummer nennen, soweit
  bekannt.
- Breite `except Exception`-Blöcke nur an CLI-, Provider-, SDK- oder
  Per-Prediction-Grenzen verwenden. Fehlerursache und beobachtbarer Fehlertyp müssen
  erhalten bleiben.
- Externe Ressourcen in `finally` oder über einen sicheren Lifecycle schließen.
- Fehlgeschlagene Predictions dürfen kein `predicted_label` enthalten.
- CLI-Konventionen erhalten: `0` für Erfolg, `1` für abgeschlossene
  Verify-/Benchmarkläufe mit fehlgeschlagenen Predictions und `2` für Eingabe- oder
  Betriebsfehler.

## YAML-, JSON- und JSONL-Konventionen

- UTF-8 verwenden.
- YAML-Konfigurationen bleiben strikt, deklarativ und frei von Secrets.
- Relative YAML-Pfade werden relativ zur deklarierenden YAML-Datei interpretiert.
- Secrets ausschließlich über Namen von Umgebungsvariablen referenzieren.
- JSONL enthält genau ein JSON-Objekt pro nichtleerer Zeile.
- IDs sind innerhalb einer Datei stabil und eindeutig; ungültige Datensätze niemals
  still überspringen.
- Relative Manifestpfade werden relativ zur Manifestdatei interpretiert.
- Datumswerte verwenden ISO `YYYY-MM-DD`; Veröffentlichungs- und Ereignisdatum nicht
  vermischen.
- Standardkonformes JSON verwenden: keine doppelten Schlüssel, `NaN`, `Infinity`,
  Kommentare oder Markdown-Hüllen.
- Persistierte JSON-/JSONL-Ausgaben deterministisch serialisieren und atomar schreiben.
- Schemaänderungen an Konfigurationen, Manifesten, Predictions oder Run-Metadaten als
  Kompatibilitätsänderungen behandeln.

## Testanforderungen

- Jede Verhaltensänderung benötigt mindestens einen gezielten Test.
- Fehlerbehebungen benötigen nach Möglichkeit einen Test, der ohne die Änderung
  fehlschlägt.
- Unit-Tests bleiben deterministisch, isoliert und netzwerkfrei.
- Dateisystemtests verwenden temporäre Verzeichnisse und verändern keine eingecheckten
  Daten oder Runs.
- Provider und externe Retriever in Standardtests über Protokolle, Fakes oder
  Mocktransporte ersetzen.
- Integrationstests prüfen nicht nur Dateiexistenz, sondern relevante Inhalte,
  Fehlerzustände und Re-Evaluierbarkeit.
- Ein Fake-Verifizierer darf nicht als Beleg semantischer oder evidenzbasierter
  Korrektheit dargestellt werden.
- Externe Integrationstests mit `external` markieren und standardmäßig deaktivieren.
- Für externe Adapter Erfolgs-, Fehler-, malformed-response-, Lifecycle- und
  ID-Mapping-Fälle abdecken.
- Neue Metriken mit kleinen, von Hand nachvollziehbaren Beispielen testen.
- Keine Tests schreiben, deren Erfolg von aktueller Uhrzeit, zufälliger Reihenfolge,
  Internetzugriff oder einem kostenpflichtigen Dienst abhängt.

## Reproduzierbare Experimente

- Alle Bedingungen verwenden denselben geordneten Claim-Satz.
- Bei kontrollierten Vergleichen Modellparameter, Prompts, Retrievalmodus und `top_k`
  konstant halten, sofern die untersuchte Variable nicht ausdrücklich eine davon ist.
- Ein Noisy-Korpus muss den vollständigen Clean-Korpus mit unverändertem Inhalt und
  unveränderter semantischer Metadatenbasis enthalten.
- Claims, Prompts, Konfigurationen, Manifeste und Korpusinhalt hashen.
- Aufgelöste, nicht geheime Konfiguration und Laufzeit-/Paketversionen im Run erfassen.
- Providerseitige Modellrevisionen festhalten, wenn der Provider dies ermöglicht.
- Synthetische Fixtures klar als solche kennzeichnen und niemals als
  Forschungsergebnisse darstellen.
- Fehlende Predictions als Fehler ausweisen; keine Fälle aus Kennzahlen entfernen.
- Retrievalmetriken nur aus echten, konkret zugeordneten Dokument-IDs berechnen.
- Keine Signifikanz-, Kausal- oder Generalisierbarkeitsaussage ohne dafür definierte
  Methodik und ausreichende Daten machen.
- Ein Experiment erst als reproduzierbar bezeichnen, wenn seine exakten Eingaben,
  Konfiguration, Promptversion, Abhängigkeitsauflösung und Rohpredictions erhalten sind.

## Zufallswerte und Seeds

- Deterministische Verfahren bevorzugen.
- Jede fachlich relevante Zufallsquelle benötigt einen expliziten, konfigurierbaren
  Seed.
- Verwendete Seeds in der aufgelösten Konfiguration oder den Run-Metadaten speichern.
- In Tests feste Seeds verwenden; keine unseeded globale Zufälligkeit.
- Bei mehreren Bibliotheken alle relevanten Zufallsquellen initialisieren.
- Zeitstempel und UUIDs dürfen Run-IDs eindeutig machen, gelten aber nicht als
  Experimentseeds.
- Temperatur `0` und ein Seed garantieren bei externen Modellen keine vollständige
  Deterministik; Provider, Modellrevision und relevante Laufzeitbedingungen zusätzlich
  dokumentieren.

## Secrets und externe Dienste

- API-Schlüssel und Tokens niemals committen, loggen, in Exceptions aufnehmen oder in
  Run-Artefakte schreiben.
- `.env` bleibt lokal und ignoriert; `.env.example` enthält ausschließlich Namen und
  leere beziehungsweise offensichtlich nicht geheime Beispielwerte.
- Konfigurationen speichern nur den Namen der API-Key-Umgebungsvariable.
- Endpunkte vor Persistierung von Credentials, Queryparametern und Fragmenten bereinigen.
- Tests verwenden keine echten Schlüssel.
- Vor kostenpflichtigen Provideraufrufen, realer externer Ingestion oder Übertragung
  lokaler Korpusdaten die ausdrückliche Zustimmung des Nutzers einholen.
- Secrets nicht über Shellausgaben oder Diagnosebefehle offenlegen; nur ihre Anwesenheit
  prüfen, wenn nötig.

## Regeln für `runs/`

- Jeden Benchmark in ein neues eindeutiges Unterverzeichnis schreiben.
- Bestehende Runs niemals überschreiben, löschen oder inhaltlich vermischen.
- `predictions.jsonl` und primäre Metadaten als Rohbeobachtungen behandeln.
- Abgeleitete Metriken und Reports dürfen ausschließlich reproduzierbar aus den
  persistierten Predictions regeneriert werden.
- Generierte Runs grundsätzlich nicht committen, sofern der Nutzer nicht ausdrücklich
  die Archivierung eines bestimmten, geprüften Runs verlangt.
- Keine Run-Artefakte manuell schönen oder fehlende Ergebnisse nachtragen.
- Vor Löschen, Verschieben oder Neuaufbau von Runs oder Indizes Zustimmung einholen.
- Run-Ergebnisse nur zusammen mit Scope, Eingaben, Fehlerzahl und Einschränkungen
  interpretieren.

## Änderungen mit Zustimmungsvorbehalt

Vor der Umsetzung Zustimmung des Nutzers einholen für:

- Änderung von Forschungsdomäne, Labelmenge oder kontrolliertem Vergleich;
- Änderungen an Goldlabels, Annotationen oder fachlicher Ground Truth;
- Beschaffung, Scraping, Transformation oder Committen nichtsynthetischer Korpusdaten;
- neue oder aktualisierte Laufzeitabhängigkeiten, Python-Mindestversionen oder
  Lockfile-Strategien;
- Austausch oder Erweiterung von RAG-Framework, Retriever, Provider oder
  Embeddingstrategie;
- Änderungen an Prompts, Metriken oder experimentellen Parametern, die bestehende
  Ergebnisse unvergleichbar machen;
- rückwärtsinkompatible CLI-, YAML-, JSON-, JSONL- oder Run-Schemaänderungen;
- neue UI-, API-, Claim-Extraktions- oder Artikelaggregationsschichten;
- externe beziehungsweise kostenpflichtige Ausführung mit realen Credentials;
- Löschen, Überschreiben oder Migration bestehender Daten, Indizes oder Runs;
- großflächiges Refactoring außerhalb des konkret beauftragten Problems.

## Definition of Done

Eine Implementierungsaufgabe ist erst abgeschlossen, wenn:

- der vereinbarte Umfang vollständig umgesetzt ist;
- relevante Erfolgs- und Fehlerpfade getestet sind;
- `pytest`, Ruff-Lint, Ruff-Formatcheck und mypy für den betroffenen Stand bestehen;
- keine neuen Secrets, lokalen Pfade oder unbeabsichtigten generierten Dateien vorliegen;
- Konfiguration, Prompts, Schemas und Dokumentation konsistent aktualisiert sind, soweit
  die Änderung sie betrifft;
- experimentelle Änderungen Vergleichbarkeit, Hashing, Provenienz und Seeds korrekt
  erfassen;
- externe Grenzen entweder real qualifiziert oder ausdrücklich als nicht überprüft
  benannt sind;
- keine unrelated Änderungen in den Diff aufgenommen wurden;
- `CODEX_HANDOFF.md` aktualisiert wurde, falls sich aktueller Stand, bekannte Grenzen
  oder offene Entscheidungen materiell geändert haben;
- die Abschlussmeldung ausgeführte Prüfungen, Ergebnis und verbleibende offene Punkte
  präzise nennt.
