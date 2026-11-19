"""Loopback-only WebRTC voice app. One live session shares the warm models."""

import asyncio
import base64
import contextlib
import os
import secrets
import sys
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from starlette.websockets import WebSocketDisconnect

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from astra.checkup import run_checkup
from astra.core import (
    DEFAULT_OLLAMA_BACKEND,
    OLLAMA_BACKEND_IDS,
    OLLAMA_BACKENDS,
    VOICE_NAMES,
    VOICES,
    Settings,
    build_phone_prompt,
    phone_prompt_file,
    build_request,
    build_system_prompt,
    local_origin_allowed,
    persona_name,
)
from astra.documents import extract_pdf_text
from astra.extensions import ExtensionRegistry
from astra.inference import Models, on_executor
from astra.knowledge import (
    MAX_FILE_BYTES,
    SUPPORTED_TYPES,
    KnowledgeError,
    KnowledgeNotFound,
    KnowledgeStore,
)
from astra.tasks import TaskExtension
from astra.triggers import dispatch_deterministic_trigger  # noqa: F401 (compatibility export)
from astra.version import source_fingerprint

ROOT = Path(__file__).resolve().parent.parent
SOURCE_FINGERPRINT = source_fingerprint()
MAX_UPLOAD_BYTES = 15 * 1024 * 1024


class Offer(BaseModel):
    sdp: str = Field(min_length=10, max_length=65536)
    type: Literal["offer"]
    voice: str | None = None
    model: str | None = Field(default=None, min_length=1, max_length=256)
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    backend: str | None = None
    knowledge_enabled: bool = False

    @field_validator("voice")
    @classmethod
    def voice_must_be_known(cls, value: str | None) -> str | None:
        if value is not None and value not in VOICE_NAMES:
            raise ValueError("Unbekannte Stimme")
        return value

    @field_validator("backend")
    @classmethod
    def backend_must_be_known(cls, value: str | None) -> str | None:
        if value is not None and value not in OLLAMA_BACKEND_IDS:
            raise ValueError("Unbekannte Ollama-Quelle")
        return value


class Disconnect(BaseModel):
    pc_id: str = Field(max_length=100)


async def _run_pipeline(
    transport, models, config, voice_state, voice_name, context, notify,
    knowledge_store=None, initial_greeting=None, vad_params=None, no_speech_phrases=None,
    extensions=None,
):
    """Shared Pipecat pipeline body for both the browser (WebRTC) and
    telephony (WebSocket) entrypoints - only the transport and how `notify`
    delivers events differ; STT/LLM/TTS wiring is identical either way.

    `initial_greeting`, when set, is spoken as soon as the transport connects
    - unlike the browser, a phone caller has no visual cue that the line is
    live, and (unlike the browser flow) nothing here otherwise makes the bot
    speak first. Also useful on its own to test the output path (TTS ->
    serializer -> mod_audio_stream) independent of STT/VAD ever working.

    `vad_params`, when set, replaces the browser-tuned VAD thresholds below -
    telephone audio arrives at completely different levels than a browser
    microphone, see Settings.telefon_vad_* for the numbers and why.

    `no_speech_phrases`, when set, makes the bot say one of them whenever VAD
    fires but Nemotron transcribes nothing for the whole turn - confirmed
    live on the phone line, where a caller otherwise gets dead silence and
    hangs up (see astra/services.py::NO_SPEECH_PHRASES). Not used for the
    browser, which already shows a "listening" state visually.
    """
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        InterruptionFrame,
        LLMFullResponseStartFrame,
        TTSSpeakFrame,
        VADUserStartedSpeakingFrame,
    )
    from pipecat.observers.base_observer import BaseObserver
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.workers.runner import WorkerRunner

    from astra.services import LocalPocketTTSService, NativeOllamaService, NemotronSTTService

    class UIObserver(BaseObserver):
        def __init__(self):
            super().__init__()
            self.seen = deque(maxlen=128)

        async def on_push_frame(self, data):
            frame = data.frame
            relevant = (
                BotStartedSpeakingFrame,
                BotStoppedSpeakingFrame,
                InterruptionFrame,
                LLMFullResponseStartFrame,
                VADUserStartedSpeakingFrame,
            )
            if not isinstance(frame, relevant) or frame.id in self.seen:
                return
            self.seen.append(frame.id)
            if isinstance(frame, BotStartedSpeakingFrame):
                notify({"type": "state", "state": "speaking"})
            elif isinstance(frame, LLMFullResponseStartFrame):
                notify({"type": "state", "state": "responding"})
            else:
                notify({"type": "state", "state": "listening"})

    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=vad_params
                or VADParams(
                    confidence=0.7,
                    start_secs=0.1,
                    stop_secs=0.7,
                    min_volume=0.6,
                )
            ),
        ),
    )
    stt = NemotronSTTService(models, notify, no_speech_phrases)
    llm = NativeOllamaService(config, notify, knowledge_store, extensions)
    tts = LocalPocketTTSService(models, voice_state, voice_name)
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=24000),
        enable_rtvi=False,
        idle_timeout_secs=None,
        observers=[UIObserver()],
    )

    @transport.event_handler("on_client_connected")
    async def connected(transport, client):
        notify({"type": "state", "state": "listening"})
        if initial_greeting:
            await llm.push_frame(TTSSpeakFrame(text=initial_greeting, append_to_context=True))

    @transport.event_handler("on_client_disconnected")
    async def disconnected(transport, client):
        await worker.cancel()

    @aggregators.user().event_handler("on_user_turn_stopped")
    async def user_turn(aggregator, strategy, message):
        if message.content:
            notify({"type": "transcript", "role": "user", "text": message.content})

    @aggregators.assistant().event_handler("on_assistant_turn_stopped")
    async def assistant_turn(aggregator, message):
        if message.content or message.interrupted:
            notify(
                {
                    "type": "transcript",
                    "role": "assistant",
                    "text": message.content or "…",
                    "interrupted": message.interrupted,
                }
            )

    @worker.event_handler("on_pipeline_error")
    async def error(worker, frame):
        notify({"type": "error", "message": frame.error})

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()


async def run_voice(
    connection, models, config, voice_state, voice_name, context,
    knowledge_store=None, extensions=None,
):
    from pipecat.transports.base_transport import TransportParams
    from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

    transport = SmallWebRTCTransport(
        connection,
        TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        ),
    )
    await _run_pipeline(
        transport, models, config, voice_state, voice_name, context,
        connection.send_app_message, knowledge_store, extensions=extensions,
    )


async def _warm_ollama(config, call_id):
    """Best-effort nudge to get config.model loaded in Ollama before the
    caller's first real answer needs it - see the call site's comment.
    """
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            payload = build_request(config, [{"role": "user", "content": "Hallo"}])
            payload["stream"] = False
            payload["options"]["num_predict"] = 1
            await client.post(f"{config.ollama_url}/api/chat", json=payload)
    except Exception:
        logger.debug("Telefonie[{}]: Ollama-Warmup fehlgeschlagen, ignoriert", call_id)


async def _model_supports_tools(config) -> bool:
    """Whether config.model natively supports tool calling - /api/offer checks
    this per model (see `ollama_models` above) before turning on
    `model_tools`; the telephony route has no equivalent capability list to
    look this up in, so it asks Ollama directly for just this one model.
    False (text-manifest tool fallback, see astra/routing.py) on any error -
    a model that turns out not to support tools well is a worse call than a
    tool-less one, not a broken one.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{config.ollama_url}/api/show", json={"model": config.model}
            )
            response.raise_for_status()
            return "tools" in response.json().get("capabilities", [])
    except Exception:
        return False


async def run_telephony_call(
    websocket, models, config, voice_state, voice_name, context, call_id,
    media_store, knowledge_store, extensions=None,
):
    """Live phone-call counterpart of `run_voice` - same STT/LLM/TTS pipeline,
    fed by FreeSWITCH's mod_audio_stream over a plain WebSocket instead of the
    browser's WebRTC connection. See astra/telephony.py for the wire protocol
    and the "Live-Telefonagent" plan for the overall architecture.
    """
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.transports.websocket.fastapi import (
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )

    from astra.services import NO_SPEECH_PHRASES
    from astra.telephony import FreeswitchAudioStreamSerializer

    def notify(payload: dict):
        # No UI is listening on a phone call - the call-relevant events
        # (transcript, error) are worth a log line for operating the line;
        # everything else (state/metric/operation/partial) is debug noise.
        level = "INFO" if payload.get("type") in ("transcript", "error", "call_metadata") else "DEBUG"
        logger.log(level, "Telefonie[{}]: {}", call_id, payload)
        if payload.get("type") == "state" and payload.get("state") == "listening":
            # "listening" also covers InterruptionFrame/VADUserStartedSpeakingFrame,
            # not just the bot genuinely finishing - see astra/telephony.py's
            # module docstring for why this, not a fixed idle timeout, is
            # serializer.flush_now()'s intended trigger. Redundant calls are a
            # no-op (nothing buffered, or telephony.py's own InterruptionFrame
            # handling already cleared it).
            asyncio.create_task(serializer.flush_now())

    serializer = FreeswitchAudioStreamSerializer(
        input_sample_rate=16000, notify=notify, send=websocket.send_text
    )
    transport = FastAPIWebsocketTransport(
        websocket,
        FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            serializer=serializer,
        ),
    )
    vad_params = VADParams(
        confidence=config.telefon_vad_confidence,
        start_secs=config.telefon_vad_start_secs,
        stop_secs=config.telefon_vad_stop_secs,
        min_volume=config.telefon_vad_min_volume,
    )
    logger.info("Telefonie[{}]: VAD-Parameter {}", call_id, vad_params)
    # The prompt alone doesn't reliably keep answers to 1-2 short sentences -
    # confirmed live, see Settings.telefon_max_tokens - so cap generation
    # length too, not just ask for it.
    config = replace(
        config, max_tokens=config.telefon_max_tokens,
        model_tools=await _model_supports_tools(config),
    )
    from astra.tools import build_tools

    # Full Astra tool set (ask_martin, search_local, search_knowledge, ...) -
    # the caller should be able to actually get something done over the
    # phone, not just prove the audio path works. generate_image is still
    # offered even though there's no screen to show the result on; the
    # model can describe it instead of pretending it can't make one.
    # Demo-Leitung: nur Tagesschau-Nachrichten (read_news), alles andere bleibt aus.
    # ASTRA_TELEFON_TOOLS=1 gibt den vollen Werkzeugsatz frei.
    phone_tools = build_tools(config, notify, media_store, knowledge_store, extensions)
    if os.getenv("ASTRA_TELEFON_TOOLS", "0") != "1":
        phone_tools = ToolsSchema(standard_tools=[
            tool for tool in phone_tools.standard_tools if tool.name == "read_news"
        ])
    context.set_tools(phone_tools)
    await _run_pipeline(
        transport, models, config, voice_state, voice_name, context, notify,
        initial_greeting=phone_prompt_file()[0] or (
            "Hallo Süßer, hier ist Dolly von der Hotline. Unsere Preise sind heute unschlagbar. Was darf es sein?"
            if os.getenv("ASTRA_TELEFON_PERSONA", "") == "gimmick" else
            "Hallo, hier ist die Sprachdemo von dillenberg Punkt net. Wie kann ich dir helfen?"
        ),
        vad_params=vad_params, no_speech_phrases=NO_SPEECH_PHRASES, extensions=extensions,
    )


def create_app(config=None, *, load_models=True):
    config = config or Settings.from_env()
    models = Models(config)
    status = {"ready": False, "stage": "starting", "error": None}
    sessions = {}
    media_store: dict[str, bytes] = {}
    knowledge_store = KnowledgeStore(config.knowledge_dir)
    extensions = ExtensionRegistry((TaskExtension(config.tasks_dir),))
    lock = asyncio.Lock()

    async def warmup():
        try:
            status["stage"] = "Gesprächssteuerung wird vorbereitet"

            def prepare_pipeline():
                import nltk
                from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import (
                    LocalSmartTurnAnalyzerV3,
                )
                from pipecat.audio.vad.silero import SileroVADAnalyzer

                from astra import services  # noqa: F401

                nltk.data.path.insert(0, str(ROOT / ".cache" / "nltk"))
                nltk.data.find("tokenizers/punkt_tab")
                nltk.sent_tokenize("Hallo. Alles bereit.")
                LocalSmartTurnAnalyzerV3()
                SileroVADAnalyzer()

            await asyncio.to_thread(prepare_pipeline)
            status["stage"] = "Spracherkennung wird geladen"
            await on_executor(models.stt_executor, models.load_stt)
            status["stage"] = "Deutsche Stimme wird geladen"
            await on_executor(models.tts_executor, models.load_tts)
            await on_executor(models.tts_executor, models.get_voice, config.voice)
            status["stage"] = f"{config.model} wird vorbereitet"
            async with httpx.AsyncClient(timeout=180) as client:
                payload = build_request(config, [{"role": "user", "content": "Sage Hallo."}])
                payload["stream"] = False
                payload["options"]["num_predict"] = 12
                response = await client.post(f"{config.ollama_url}/api/chat", json=payload)
                response.raise_for_status()
                result = response.json()
                if result.get("error") or not result.get("message", {}).get("content"):
                    raise RuntimeError(f"Ollama: {result.get('error', 'Keine Antwort')}")
                if result["message"].get("thinking"):
                    raise RuntimeError("Ollama hat Thinking nicht ausgeschaltet")
            status.update(ready=True, stage="Bereit")
            try:
                for warning in await run_checkup(config):
                    logger.warning(warning)
            except Exception:
                logger.exception("Checkup beim Start fehlgeschlagen")
            logger.info("MARTI bereit auf http://localhost:{}", config.port)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Modelle konnten nicht geladen werden")
            message = str(exc) or type(exc).__name__
            if isinstance(exc, httpx.TimeoutException):
                message = "Ollama antwortet nicht rechtzeitig. Bitte den lokalen Modelldienst prüfen."
            status.update(ready=False, stage="Start fehlgeschlagen", error=message)

    @asynccontextmanager
    async def lifespan(app):
        warming = asyncio.create_task(warmup()) if load_models else None
        await extensions.start()
        try:
            yield
        finally:
            await extensions.stop()
            if warming:
                warming.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await warming
            for connection, task, _ in list(sessions.values()):
                if connection:
                    await connection.disconnect()
                if task:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            await models.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    extensions.mount(app)
    app.state.status = status
    app.state.extensions = extensions

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        allowed_hosts = {"localhost", "127.0.0.1"}
        if config.tailnet_host:
            allowed_hosts.add(config.tailnet_host)
        if request.url.hostname not in allowed_hosts:
            return JSONResponse({"detail": "Nur lokal erreichbar"}, status_code=403)
        if request.method not in ("GET", "HEAD", "OPTIONS") and not local_origin_allowed(
            request.headers.get("origin", ""), config.port, config.tailnet_host
        ):
            return JSONResponse({"detail": "Ungültiger Ursprung"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; media-src 'self' blob:; img-src 'self' data:; frame-ancestors 'none'"
        )
        return response

    @app.get("/")
    async def index():
        return FileResponse(ROOT / "web" / "index.html")

    @app.get("/api/status")
    async def health():
        return {
            **status, "busy": bool(sessions), "model": config.model,
            "temperature": config.temperature, "thinking": False,
            "source_fingerprint": SOURCE_FINGERPRINT,
        }

    def backend_url(backend: str | None) -> str:
        try:
            return config.backend_url(backend)
        except KeyError as exc:
            raise HTTPException(422, "Unbekannte Ollama-Quelle.") from exc

    def backend_label(backend: str | None) -> str:
        selected = backend or DEFAULT_OLLAMA_BACKEND
        return next(entry["label"] for entry in OLLAMA_BACKENDS if entry["id"] == selected)

    def backend_default_model(backend: str | None) -> str:
        try:
            return config.backend_default_model(backend)
        except KeyError as exc:
            raise HTTPException(422, "Unbekannte Ollama-Quelle.") from exc

    async def ollama_models(backend: str | None = None):
        url = backend_url(backend)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{url}/api/tags")
                response.raise_for_status()
                entries = response.json()["models"]
                result = []
                for entry in entries:
                    if entry.get("remote_host") or entry.get("remote_model"):
                        continue
                    capabilities = entry.get("capabilities")
                    if capabilities is None:
                        details = await client.post(
                            f"{url}/api/show", json={"model": entry["name"]}
                        )
                        details.raise_for_status()
                        capabilities = details.json()["capabilities"]
                    if "completion" in capabilities:
                        result.append({"name": entry["name"], "capabilities": capabilities})
                return sorted(result, key=lambda item: item["name"])
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise HTTPException(
                503,
                f"Ollama-Modelle von „{backend_label(backend)}“ konnten nicht geladen werden.",
            ) from exc

    @app.get("/api/models")
    async def available_models(backend: str | None = None):
        return {"models": await ollama_models(backend), "default": backend_default_model(backend)}

    @app.get("/api/backends")
    async def backends():
        return {
            "backends": [{"id": entry["id"], "label": entry["label"]} for entry in OLLAMA_BACKENDS],
            "default": DEFAULT_OLLAMA_BACKEND,
        }

    @app.get("/api/voices")
    async def voices():
        return {"voices": list(VOICES), "default": config.voice}

    @app.get("/api/knowledge")
    async def knowledge_documents():
        return {"documents": await asyncio.to_thread(knowledge_store.list_documents)}

    @app.post("/api/knowledge/upload")
    async def knowledge_upload(files: list[UploadFile] = File(...)):  # noqa: B008
        if not files or len(files) > 20:
            raise HTTPException(400, "Bitte 1 bis 20 Dateien auswählen.")
        results = []
        for file in files:
            filename = file.filename or "Datei"
            if Path(filename).suffix.lower() not in SUPPORTED_TYPES:
                results.append({"filename": filename, "status": "error", "error": "Nicht unterstützt"})
                continue
            data = await file.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                results.append({"filename": filename, "status": "error", "error": "Datei zu groß (maximal 25 MB)."})
                continue
            try:
                document = await asyncio.to_thread(knowledge_store.add, filename, data)
            except KnowledgeError as exc:
                results.append({"filename": filename, "status": "error", "error": str(exc)})
                continue
            except Exception:
                # A single malformed upload (e.g. a corrupt PDF, or a .docx
                # that's a valid zip but not a real Word document) must not
                # 500 the whole batch and lose the results of every file
                # already committed earlier in this loop.
                logger.exception("Unerwarteter Fehler beim Verarbeiten von {}", filename)
                results.append({"filename": filename, "status": "error", "error": "Datei konnte nicht verarbeitet werden."})
                continue
            results.append({"status": "ready", **document})
        return {"documents": results}

    @app.delete("/api/knowledge/{document_id}")
    async def delete_knowledge(document_id: str):
        try:
            await asyncio.to_thread(knowledge_store.remove, document_id)
        except KnowledgeNotFound as exc:
            raise HTTPException(404, str(exc)) from exc
        except KnowledgeError as exc:
            raise HTTPException(503, str(exc)) from exc
        return {"ok": True}

    @app.post("/api/offer")
    async def offer(body: Offer):
        if not status["ready"]:
            raise HTTPException(503, status["error"] or status["stage"])
        async with lock:
            if sessions:
                raise HTTPException(
                    409, "Ein Gespräch läuft bereits. Beende es im anderen Fenster."
                )
            model_name = body.model or backend_default_model(body.backend)
            available = await ollama_models(body.backend)
            selected = next((item for item in available if item["name"] == model_name), None)
            if selected is None:
                raise HTTPException(
                    422,
                    f"Das Modell ist kein installiertes Sprachmodell auf „{backend_label(body.backend)}“.",
                )
            session_config = replace(
                config, ollama_url=backend_url(body.backend),
                model=model_name, model_tools="tools" in selected["capabilities"],
                temperature=body.temperature if body.temperature is not None else config.temperature,
            )
            voice_name = body.voice or config.voice
            try:
                voice_state = await on_executor(models.tts_executor, models.get_voice, voice_name)
            except Exception as exc:
                raise HTTPException(
                    503, f"Stimme „{voice_name}“ konnte nicht geladen werden."
                ) from exc
            from pipecat.processors.aggregators.llm_context import LLMContext
            from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection

            from astra.tools import build_tools

            connection = SmallWebRTCConnection(ice_servers=[], connection_timeout_secs=20)
            try:
                await connection.initialize(body.sdp, body.type)
            except Exception:
                await connection.disconnect()
                raise

            notify = connection.send_app_message
            active_knowledge_store = knowledge_store if body.knowledge_enabled else None
            context = LLMContext(
                [{"role": "system", "content": build_system_prompt(persona_name(model_name))}],
                tools=build_tools(
                    session_config, notify, media_store, active_knowledge_store, extensions
                ),
            )

            async def session():
                try:
                    await run_voice(
                        connection, models, session_config, voice_state, voice_name,
                        context, active_knowledge_store, extensions,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception("Gespräch fehlgeschlagen")
                    connection.send_app_message({"type": "error", "message": str(exc)})
                finally:
                    await connection.disconnect()
                    sessions.pop(connection.pc_id, None)

            task = asyncio.create_task(session())
            sessions[connection.pc_id] = (connection, task, context)
            return connection.get_answer()

    @app.post("/api/disconnect")
    async def disconnect(body: Disconnect):
        pair = sessions.get(body.pc_id)
        if pair:
            connection, task, _ = pair
            await connection.disconnect()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        return {"ok": True}

    class _TelephonyConnection:
        """Lets a phone call share `sessions`'s (connection, task, context)
        shape with browser sessions, so the single lifespan-shutdown loop
        and the single-session gate below work for both without special-casing.
        """

        def __init__(self, websocket: WebSocket):
            self._websocket = websocket

        async def disconnect(self):
            with contextlib.suppress(Exception):
                await self._websocket.close()

    @app.websocket("/telefon/inbound")
    async def telefon_inbound(websocket: WebSocket):
        # No local_only equivalent applies here - `@app.middleware("http")`
        # does not see websocket scopes at all, so this route is reachable
        # from anywhere the port is reachable from (the tailnet, once
        # ASTRA_TELEFON_SECRET is set - see main()). The shared secret is
        # this route's entire access control; without it the route refuses
        # every connection rather than falling back to "trust the network".
        if not config.telefon_secret or not secrets.compare_digest(
            websocket.query_params.get("token", ""), config.telefon_secret
        ):
            await websocket.close(code=4403)
            return
        await websocket.accept()
        if not status["ready"]:
            await websocket.close(code=1013)
            return
        call_id = uuid4().hex[:8]
        # Ollama can end up with config.model evicted from memory between calls
        # (e.g. another model got loaded in between, despite keep_alive=-1) -
        # confirmed 2026-09-13: a cold generate() took ~12s just to load the
        # model versus <1s once warm, which blew past the caller's patience on
        # the very first turn. Kick a throwaway generation off in parallel with
        # the greeting/voice-loading below so the load, if needed, happens
        # while the caller is still hearing "Hallo, hier ist..." instead of
        # during their first real answer. Best-effort - a failure here must
        # not abort the call, the real request will just pay the cold cost.
        asyncio.create_task(_warm_ollama(config, call_id))
        async with lock:
            if sessions:
                logger.info("Telefonie[{}]: abgelehnt, es läuft bereits eine Sitzung", call_id)
                await websocket.close(code=1013)
                return
            voice_name = config.telefon_voice
            try:
                voice_state = await on_executor(models.tts_executor, models.get_voice, voice_name)
            except Exception:
                logger.exception("Telefonie[{}]: Stimme „{}“ konnte nicht geladen werden", call_id, voice_name)
                await websocket.close(code=1011)
                return
            from pipecat.processors.aggregators.llm_context import LLMContext

            # Tools (ask_martin, search_local, search_knowledge, ...) are
            # attached in run_telephony_call, not here - building them needs
            # `notify`, which doesn't exist until that function's serializer
            # does (see its own comment).
            context = LLMContext(
                [{"role": "system", "content": build_phone_prompt(persona_name(config.model))}],
            )
            key = f"telefon-{call_id}"
            sessions[key] = (_TelephonyConnection(websocket), asyncio.current_task(), context)
        logger.info("Telefonie[{}]: Anruf verbunden", call_id)
        try:
            await run_telephony_call(
                websocket, models, config, voice_state, voice_name, context, call_id,
                media_store, knowledge_store, extensions,
            )
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("Telefonie[{}]: Gespräch fehlgeschlagen", call_id)
        finally:
            sessions.pop(key, None)
            logger.info("Telefonie[{}]: Anruf beendet", call_id)

    @app.get("/api/media/{image_id}")
    async def media(image_id: str):
        image = media_store.get(image_id)
        if image is None:
            raise HTTPException(404, "Bild nicht gefunden")
        return Response(content=image, media_type="image/png")

    @app.post("/api/upload")
    async def upload(pc_id: str = Form(...), file: UploadFile = File(...)):  # noqa: B008
        pair = sessions.get(pc_id)
        if not pair:
            raise HTTPException(404, "Keine aktive Sitzung")
        connection, _, context = pair
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Datei zu groß (max. 15 MB)")
        if file.content_type == "application/pdf":
            try:
                text = await asyncio.to_thread(extract_pdf_text, data)
            except Exception as exc:
                logger.warning("PDF-Upload konnte nicht gelesen werden: {}", type(exc).__name__)
                raise HTTPException(422, "PDF konnte nicht gelesen werden.") from exc
            if not text:
                raise HTTPException(422, "Im PDF wurde kein Text gefunden")
            context.add_message(
                {
                    "role": "user",
                    "content": f'[Hochgeladenes Dokument "{file.filename}"]\n\n{text}',
                }
            )
        elif file.content_type in {"image/png", "image/jpeg", "image/webp"}:
            context.add_message(
                {
                    "role": "user",
                    "content": f'[Hochgeladenes Bild "{file.filename}"]',
                    "images": [base64.b64encode(data).decode()],
                }
            )
        else:
            raise HTTPException(415, "Nur PDF, PNG, JPEG oder WebP werden unterstützt")
        connection.send_app_message({"type": "upload", "filename": file.filename})
        return {"ok": True}

    app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
    return app


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    logger.remove()
    logger.add(sys.stderr, level=os.getenv("ASTRA_LOG_LEVEL", "INFO"))
    config = Settings.from_env()
    # Always loopback-only, telephony included: reachability beyond localhost
    # goes through `tailscale serve --bg 7860` (see README "Im Tailnet
    # freigeben"), which binds only the tailnet interface and terminates TLS -
    # binding uvicorn itself any wider than 127.0.0.1 would bypass that and
    # expose every interface on the machine, not just the tailnet.
    uvicorn.run(create_app(config), host="127.0.0.1", port=config.port, access_log=False)


if __name__ == "__main__":
    main()
