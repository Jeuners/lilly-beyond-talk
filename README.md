# Lilly Beyond Talk

Fans treffen die KI-Figur **Lilly Beyond** im Browser und reden live mit ihr, Auge in Auge in der jeweiligen Szene, in vier Szenen, die nacheinander frei werden (Frühstück im Hotel, Fitness, Abendessen, Hotelzimmer als 18+-Premium). Alles läuft lokal auf Apple Silicon, nichts geht an Dritte.

Lilly Beyond ist eine **KI-generierte Figur**. Die App sagt das vor jedem Gespräch deutlich, und Lilly behauptet nie, ein Mensch zu sein.

## Wie es funktioniert

| Teil | Was |
|---|---|
| Hören | Nemotron ASR (MLX, lokal) |
| Denken | Gemma 4 12B über Ollama (lokal, Standard), jedes andere Ollama-Modell möglich |
| Sprechen | Pocket TTS (lokal) mit Stimmprobe, englische Wörter über `persona/aussprache.json` |
| Persona | `persona/lilly_talk_prompt.txt` (Fan-Prompt) plus Szene aus `lilly_talk/scenes.json` |
| Server | `lilly_talk/server.py`: schlanker FastAPI-Server, keine Werkzeuge, Uploads, Suche oder Telefonie. Lauscht nur auf `127.0.0.1` |
| Oberfläche | `lilly_talk/web/`: Altersabfrage, Szenenwahl, Gesprächsansicht im FaceTime-Stil, Video-Loop pro Szene, Untertitel |
| Sprachkette | `astra/`: Pipecat-Pipeline, Abkömmling von [astra-local-voice](https://github.com/Jeuners/astra-local-voice) |

## Starten

Voraussetzungen: Apple Silicon, Python 3.12, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com) mit `ollama pull gemma4:12b-mlx`.

```
uv sync
export LILLY_VOICE=/pfad/zur/stimmprobe.wav   # siehe „Stimme“
./start_talk.sh                                # → http://localhost:9903
```

| Variable | Wirkung |
|---|---|
| `LILLY_TALK_ALLES_FREI=1` | Testmodus: alle Szenen frei, Premium ohne Zahlung |
| `ASTRA_MODEL`, `ASTRA_OLLAMA_URL` | anderes Modell bzw. anderer Ollama-Endpunkt |
| `LILLY_TALK_FESTE_BEGRUESSUNG=1` | feste Begrüßung aus `scenes.json` statt vom Modell erzeugter |
| `LILLY_TALK_LLM_ANBIETER` | Text für den Datenschutzhinweis, wenn `ASTRA_OLLAMA_URL` nicht lokal ist, z. B. „einen KI-Dienst in Deutschland (Anbieter X)“. Bei lokalem Modell zeigt der Hinweis automatisch „nichts geht an Dritte“ |
| `LILLY_TALK_MAX_SECONDS` | Gesprächsdauer (Standard 300) |
| `LILLY_TALK_CALLS_PER_HOUR` | Anrufe pro Stunde und Adresse (Standard 10) |
| `LILLY_TALK_MIN_SECONDS` | Mindestdauer, damit eine Szene als gespielt zählt (Standard 60) |

## Stimme

Eine Stimmprobe (WAV, ca. 8 s) ist **nicht** im Repo. Nimm nur eine Stimme, die du selbst eingesprochen hast, für die eine schriftliche Zustimmung vorliegt oder die lizenziert ist. Pocket TTS verbietet das Klonen fremder Stimmen ohne Zustimmung. Das Modell `kyutai/pocket-tts` ist gesperrt: Bedingungen auf Hugging Face annehmen, Token nur für den Download.

## Schutz (eingebaut)

- 18+-Abfrage und KI-Hinweis vor jedem Gespräch.
- Fan-Prompt: keine Treffen, keine Kontaktdaten, keine Fragen nach persönlichen Daten, kein Flirten mit Minderjährigen, Hinweis auf die Telefonseelsorge bei Not, lässt sich nicht aus der Rolle bringen.
- Ein Anruf gleichzeitig, höchstens 5 Minuten, 10 Anrufe pro Stunde und Adresse, nur gleicher Ursprung, strikte CSP.
- Logs (nur stderr, Stufe INFO) enthalten keine Gesprächsinhalte, keine Texte und keine IP-Adressen, nur Start und Ende eines Gesprächs mit Szenen-ID und Dauer. Der HTTP-Zugriffslog ist aus, die Diagnose-Aufnahme der Basis (`ASTRA_STT_DUMP_DIR`) wird in Lilly Talk ignoriert.
- Gespeichert wird nur eine zufällige Cookie-ID mit den gespielten Szenen (`lilly_talk/data/`, nicht in Git). Keine Namen, keine Aufnahmen, keine Gesprächsinhalte.

## Szenenmedien

Standbilder (z-image turbo, Apache 2.0) und 5-s-Loops (Wan 2.2 I2V, Apache 2.0) liegen in `lilly_talk/web/scenes/`. Neue Loops: `COMFYUI_URL=http://… lilly_talk/tools/wan_loop.py bild.png "Bewegung" out.mp4`. Sprachmodelle vergleichen: `lilly_talk/tools/modelltest.py`.

## Vor einem öffentlichen Betrieb zu klären

Dieses Repo ist der Code, kein fertiger Dienst.

1. **Stimme:** lizenzierte oder eingewilligte Stimme (siehe oben).
2. **Server:** eigener Server mit GPU, TURN für WebRTC, Kostenbremse. Die Oberfläche nie ungeschützt ins Netz stellen.
3. **Szene 4 (18+):** echte Altersverifikation (JMStV), Zahlungsanbieter statt Simulation, Bedingungen des Modellanbieters. Der Prompt ist bewusst nicht explizit.
4. **Rechtliches:** Datenschutzerklärung, Impressum, Kennzeichnung der KI.

## Lizenz

Code: MIT (siehe `LICENSE`). Die Figur Lilly Beyond, ihr Name, die Bilder, Videos und Texte in `lilly_talk/web/` und `persona/` sind urheberrechtlich geschützt und **nicht** unter der MIT-Lizenz freigegeben, außer für Tests dieses Projekts. Die Schriften (Space Grotesk, Unbounded, JetBrains Mono) stehen unter der SIL OFL.
