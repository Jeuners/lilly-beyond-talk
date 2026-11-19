#!/bin/zsh
# Startet Lilly Beyond Talk (nur 127.0.0.1). Voraussetzung: Ollama läuft, Modell geladen (Standard gemma4:12b-mlx),
# Stimmprobe LILLY_VOICE gesetzt (eigene oder lizenzierte Stimme, siehe README "Stimme").
set -eu
HIER="${0:A:h}"
: "${LILLY_VOICE:?LILLY_VOICE=/pfad/zu/stimmprobe.wav setzen (nur Stimmen mit Zustimmung/Lizenz)}"
export ASTRA_PORT="${LILLY_TALK_PORT:-9903}"
export ASTRA_OLLAMA_URL="${ASTRA_OLLAMA_URL:-http://127.0.0.1:11434}" ASTRA_MODEL="${ASTRA_MODEL:-gemma4:12b-mlx}"
export ASTRA_VOICE="${ASTRA_VOICE:-eve}" ASTRA_VOICE_FILE="$LILLY_VOICE" ASTRA_AUSSPRACHE_FILE="$HIER/persona/aussprache.json"
export ASTRA_TEMPERATURE=0.8 LILLY_TALK_PROMPT_FILE="$HIER/persona/lilly_talk_prompt.txt"
export LILLY_TALK_ALLES_FREI="${LILLY_TALK_ALLES_FREI:-0}" LILLY_TALK_MAX_SECONDS="${LILLY_TALK_MAX_SECONDS:-300}" LILLY_TALK_CALLS_PER_HOUR="${LILLY_TALK_CALLS_PER_HOUR:-10}"
cd "$HIER"; exec python -m lilly_talk.server
