"""Lilly Beyond Talk: Fans reden mit Lilly Beyond.

Nur die Sprachkette aus Astra (astra/) (Nemotron hört, Sprachmodell denkt, Pocket TTS spricht).
Bewusst NICHT dabei: Werkzeuge, lokale Suche, Uploads, Wissen, Aufgaben, ComfyUI, AgentClaw,
Telefonie, Modell- oder Stimmenwahl. Schutz: 18+-Bestätigung, KI-Hinweis, Zeitlimit pro Anruf,
ein Anruf gleichzeitig, Anrufe pro Adresse und Stunde begrenzt.

Start: ./start_talk.sh. Läuft nur auf 127.0.0.1.
"""
import asyncio
import contextlib
import json
import os
import secrets
import random
import sys
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from astra.core import Settings, build_request
from astra.inference import Models, on_executor

HERE = Path(__file__).resolve().parent
WEB = HERE / "web"
MAX_SECONDS = int(os.getenv("LILLY_TALK_MAX_SECONDS", "300"))
FESTE_BEGRUESSUNG = os.getenv("LILLY_TALK_FESTE_BEGRUESSUNG") == "1"  # 1 = Sätze aus scenes.json statt Modell
OPENING_CUE = ("(Die Person ist gerade bei dir angekommen. Begrüße sie jetzt mit ein bis zwei kurzen, natürlichen Sätzen, "
               "passend zur Szene und in deinen eigenen Worten. Fang nicht mit Lachen oder „haha“ an, stell höchstens "
               "eine einfache Frage am Ende oder lieber gar keine.)")
CALLS_PER_HOUR = int(os.getenv("LILLY_TALK_CALLS_PER_HOUR", "10"))
DATA = Path(os.path.expanduser(os.getenv("LILLY_TALK_DATA", str(HERE / "data"))))
COOKIE = "lilly_fan"
SCENE_CONFIG = json.loads((HERE / "scenes.json").read_text(encoding="utf-8"))
SCENES = SCENE_CONFIG["scenes"]
MIN_SECONDS = int(os.getenv("LILLY_TALK_MIN_SECONDS", SCENE_CONFIG.get("min_seconds", 60)))
ALLES_FREI = os.getenv("LILLY_TALK_ALLES_FREI") == "1"  # nur zum Testen: alle Szenen frei, Premium ohne Zahlung


class Progress:
    """Welche Szenen ein Besucher (Zufalls-ID im Cookie) schon gespielt hat, ob Premium (simuliert) bezahlt ist.
    Kleine JSON-Datei in DATA; keine Namen, keine Aufnahmen, keine Inhalte."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict = json.loads(path.read_text()) if path.is_file() else {}

    def get(self, fan: str) -> dict:
        return self.data.setdefault(fan, {"played": [], "paid": False})

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data))
        tmp.replace(self.path)

    def state(self, fan: str) -> list[dict]:
        me = self.get(fan)
        out = []
        for i, scene in enumerate(SCENES):
            unlocked = ALLES_FREI or i == 0 or SCENES[i - 1]["id"] in me["played"]
            needs_payment = scene["premium"] and not me["paid"] and not ALLES_FREI
            out.append({
                "id": scene["id"], "title": scene["title"], "teaser": scene["teaser"],
                "premium": scene["premium"], "played": scene["id"] in me["played"],
                "unlocked": unlocked, "needs_payment": needs_payment,
                "playable": unlocked and not needs_payment,
            })
        return out


class Offer(BaseModel):
    sdp: str
    type: str
    age_confirmed: bool = False
    scene: str = ""


class Disconnect(BaseModel):
    pc_id: str


def load_prompt() -> str:
    path = os.path.expanduser(os.getenv("LILLY_TALK_PROMPT_FILE", ""))
    if not path or not os.path.isfile(path):
        raise SystemExit("LILLY_TALK_PROMPT_FILE fehlt (Fan-Prompt von Lilly)")
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def create_app(config: Settings) -> FastAPI:
    prompt = load_prompt()
    models = Models(config)
    status = {"ready": False, "stage": "Lilly macht sich fertig", "error": None}
    sessions: dict[str, tuple] = {}
    calls: dict[str, deque] = defaultdict(deque)
    lock = asyncio.Lock()
    progress = Progress(DATA / "progress.json")

    def fan_of(request: Request) -> str | None:
        fan = request.cookies.get(COOKIE, "")
        return fan if len(fan) == 32 and fan.isalnum() else None

    def with_cookie(response: JSONResponse, request: Request) -> JSONResponse:
        if not fan_of(request):
            response.set_cookie(COOKIE, secrets.token_hex(16), max_age=180 * 86400,
                                httponly=True, samesite="strict", secure=request.url.scheme == "https")
        return response

    async def warmup():
        try:
            def prepare():
                import nltk
                from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
                from pipecat.audio.vad.silero import SileroVADAnalyzer

                from astra import services  # noqa: F401
                nltk.sent_tokenize("Hallo. Alles bereit.")
                LocalSmartTurnAnalyzerV3()
                SileroVADAnalyzer()

            await asyncio.to_thread(prepare)
            await on_executor(models.stt_executor, models.load_stt)
            await on_executor(models.tts_executor, models.load_tts)
            await on_executor(models.tts_executor, models.get_voice, config.voice)
            async with httpx.AsyncClient(timeout=120) as client:
                payload = build_request(config, [{"role": "user", "content": "Sage Hallo."}])
                payload["stream"] = False
                payload["options"]["num_predict"] = 8
                (await client.post(f"{config.ollama_url}/api/chat", json=payload)).raise_for_status()
            status.update(ready=True, stage="Lilly ist da")
            logger.info("Lilly Beyond Talk bereit auf http://localhost:{}", config.port)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Start fehlgeschlagen")
            status.update(ready=False, stage="Lilly ist gerade nicht erreichbar", error=type(exc).__name__)

    @asynccontextmanager
    async def lifespan(app):
        warming = asyncio.create_task(warmup())
        try:
            yield
        finally:
            warming.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await warming
            for connection, task in list(sessions.values()):
                await connection.disconnect()
                task.cancel()
            await models.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        allowed = {"localhost", "127.0.0.1"} | {h for h in os.getenv("LILLY_TALK_HOSTS", "").split(",") if h}
        if request.url.hostname not in allowed:
            return JSONResponse({"detail": "Nicht erreichbar"}, status_code=403)
        if request.method == "POST":
            origin = request.headers.get("origin", "")
            if not any(origin == f"{scheme}://{host}{port}" for host in allowed
                       for scheme in ("http", "https") for port in (f":{config.port}", "")):
                return JSONResponse({"detail": "Ungültiger Ursprung"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "microphone=(self), camera=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; "
            "media-src 'self' blob:; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
        )
        return response

    @app.get("/")
    async def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/status")
    async def get_status():
        from urllib.parse import urlparse
        local = (urlparse(config.ollama_url).hostname or "") in ("127.0.0.1", "localhost", "::1")
        return {**status, "busy": bool(sessions), "max_seconds": MAX_SECONDS, "llm_local": local,
                "llm_provider": os.getenv("LILLY_TALK_LLM_ANBIETER", "")}

    @app.get("/api/scenes")
    async def scenes(request: Request):
        fan = fan_of(request)
        state = progress.state(fan) if fan else Progress(Path("/nonexistent")).state("neu")
        return with_cookie(JSONResponse({"scenes": state, "min_seconds": MIN_SECONDS}), request)

    @app.post("/api/pay-simulated")
    async def pay_simulated(request: Request):
        """SIMULATION: schaltet Premium ohne echte Zahlung frei (nur nachdem Szene 3 gespielt wurde)."""
        fan = fan_of(request)
        if not fan:
            raise HTTPException(400, "Bitte Seite neu laden.")
        state = {s["id"]: s for s in progress.state(fan)}
        premium = next(s for s in SCENES if s["premium"])
        if not state[premium["id"]]["unlocked"]:
            raise HTTPException(403, "Spiel erst die Szenen davor.")
        progress.get(fan)["paid"] = True
        progress.save()
        logger.info("Premium (simuliert) freigeschaltet")
        return {"scenes": progress.state(fan)}

    @app.post("/api/offer")
    async def offer(body: Offer, request: Request):
        if not body.age_confirmed:
            raise HTTPException(403, "Lilly Beyond Talk ist erst ab 18.")
        fan = fan_of(request)
        if not fan:
            raise HTTPException(400, "Bitte Seite neu laden.")
        scene = next((s for s in SCENES if s["id"] == body.scene), None)
        if scene is None:
            raise HTTPException(422, "Bitte zuerst eine Szene wählen.")
        if not {s["id"]: s for s in progress.state(fan)}[scene["id"]]["playable"]:
            raise HTTPException(403, "Diese Szene ist noch nicht freigeschaltet.")
        if not status["ready"]:
            raise HTTPException(503, status["stage"])
        client = request.client.host if request.client else "?"
        now = time.time()
        recent = calls[client]
        while recent and now - recent[0] > 3600:
            recent.popleft()
        if len(recent) >= CALLS_PER_HOUR:
            raise HTTPException(429, "Lilly braucht eine Pause. Versuch es in einer Stunde wieder.")
        async with lock:
            if sessions:
                raise HTTPException(409, "Lilly ist gerade in einem Gespräch. Versuch es gleich noch mal.")
            recent.append(now)
            from pipecat.processors.aggregators.llm_context import LLMContext
            from pipecat.transports.base_transport import TransportParams
            from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
            from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

            from astra.server import _run_pipeline

            voice_state = await on_executor(models.tts_executor, models.get_voice, config.voice)
            connection = SmallWebRTCConnection(ice_servers=[], connection_timeout_secs=20)
            try:
                await connection.initialize(body.sdp, body.type)
            except Exception:
                await connection.disconnect()
                raise
            notify = connection.send_app_message
            scene_prompt = f"{prompt}\n\nDie Szene, in der ihr gerade seid: {scene['situation']}"
            context = LLMContext([{"role": "system", "content": scene_prompt}])  # keine Werkzeuge
            transport = SmallWebRTCTransport(connection, TransportParams(
                audio_in_enabled=True, audio_out_enabled=True,
                audio_in_sample_rate=16000, audio_out_sample_rate=24000,
            ))

            async def session():
                started = time.time()
                try:
                    await asyncio.wait_for(_run_pipeline(
                        transport, models, config, voice_state, config.voice, context, notify,
                        initial_greeting=random.choice(scene["greetings"]) if FESTE_BEGRUESSUNG else None,
                        opening_cue=None if FESTE_BEGRUESSUNG else OPENING_CUE,
                    ), timeout=MAX_SECONDS)
                except asyncio.TimeoutError:
                    notify({"type": "limit_reached"})
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Anruf fehlgeschlagen")
                    notify({"type": "error", "message": "Die Verbindung zu Lilly ist abgebrochen."})
                finally:
                    seconds = time.time() - started
                    if seconds >= MIN_SECONDS and scene["id"] not in progress.get(fan)["played"]:
                        progress.get(fan)["played"].append(scene["id"])
                        progress.save()
                    await connection.disconnect()
                    sessions.pop(connection.pc_id, None)
                    logger.info("Anruf ({}) beendet nach {:.0f} s", scene["id"], seconds)

            sessions[connection.pc_id] = (connection, asyncio.create_task(session()))
            logger.info("Anruf gestartet: {}", scene["id"])
            return {**connection.get_answer(), "max_seconds": MAX_SECONDS}

    @app.post("/api/disconnect")
    async def disconnect(body: Disconnect):
        pair = sessions.get(body.pc_id)
        if pair:
            connection, task = pair
            await connection.disconnect()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        return {"ok": True}

    app.mount("/static", StaticFiles(directory=WEB), name="static")
    return app


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.pop("ASTRA_STT_DUMP_DIR", None)  # Diagnose-Aufnahme der Basis ist hier nie erlaubt: keine Aufnahmen
    logger.remove()
    logger.add(sys.stderr, level=os.getenv("ASTRA_LOG_LEVEL", "INFO"))
    config = Settings.from_env()
    # Fans: keine Werkzeuge und keine Stichwort-Auslöser für lokale Suche
    config = replace(config, model_tools=False, local_search_roots=(), comfyui_url="", agentclaw_url="")
    uvicorn.run(create_app(config), host="127.0.0.1", port=config.port, access_log=False)


if __name__ == "__main__":
    main()
