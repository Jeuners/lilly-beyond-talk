"""Pipecat adapters for native Ollama, live Nemotron ASR, and Pocket TTS."""

import asyncio
import json
import logging
import os
import random
import time
from contextlib import aclosing
from uuid import uuid4

import httpx
from loguru import logger
from openai.types.chat import ChatCompletionChunk
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.settings import STTSettings, TTSSettings
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.types import is_given

from astra.core import Settings, budget_request, build_request
from astra.inference import Models, Recognizer, drain_stream, on_executor
from astra.knowledge import KnowledgeStore, find_relevant_search
from astra.routing import route_tool, text_messages, tool_manifest
from astra.triggers import (
    detect_local_search,
    dispatch_deterministic_trigger,
    latest_tool_result,
    references_previous_article,
    trigger_tool,
    wants_article_summary,
)

# Varied filler lines instead of one fixed sentence, so back-to-back waits
# don't sound like a recording. Kept short — these are spoken, not read.
GENERIC_WAIT_PHRASES = (
    "Einen Moment bitte.",
    "Mh, Moment.",
    "Ich schau.",
    "Moment, ich schau nach.",
    "Sekunde.",
)
LOCAL_SEARCH_WAIT_PHRASES = (
    "Einen Moment, ich suche in deinen Projekten.",
    "Mh, Moment, ich schau in deinen Projekten nach.",
    "Ich schau kurz in deinen Projekten.",
    "Moment, ich durchsuche deine Projekte.",
)
KNOWLEDGE_WAIT_PHRASES = (
    "Einen Moment, ich schaue im Wissensspeicher nach.",
    "Mh, Moment, ich schau im Wissensspeicher nach.",
    "Ich schau kurz in deinen Dateien nach.",
    "Moment, ich durchsuche den Wissensspeicher.",
)
# Confirmed live on the phone line: VAD (tuned loose for quiet telephone
# audio, see Settings.telefon_vad_min_volume) sometimes fires on something
# that isn't intelligible speech - line noise, a breath - and Nemotron
# correctly transcribes nothing for the whole turn. Silently doing nothing
# then reads as a dead line to the caller; they wait a while and hang up.
# Only used when NemotronSTTService is given a phrase list (telephony only -
# the browser UI already shows its own "listening" state).
NO_SPEECH_PHRASES = (
    "Wie bitte?",
    "Kannst du das wiederholen?",
    "Ich habe das nicht verstanden.",
    "Entschuldige, das kam nicht an.",
)


class NativeOllamaService(OpenAILLMService):
    """Use /api/chat so think=False cannot be lost in compatibility translation."""

    supports_developer_role = False

    def __init__(
        self, config: Settings, notify, knowledge_store: KnowledgeStore | None = None,
        extensions=None,
    ):
        super().__init__(
            api_key="local",
            base_url=f"{config.ollama_url}/v1",
            settings=self.Settings(model=config.model),
        )
        self.config = config
        self.notify = notify
        self.knowledge_store = knowledge_store
        self.extensions = extensions
        self.last_headlines: list[dict] = []

    def _article_policy(self, context):
        messages = context.get_messages()
        article = latest_tool_result(messages, "read_article", current_turn=True)
        wiki = latest_tool_result(messages, "read_wikipedia", current_turn=True)
        search = latest_tool_result(messages, "search_wikipedia", current_turn=True)
        request_text = next((m.get("content", "") for m in reversed(messages)
                             if m.get("role") == "user"), "")
        summary = isinstance(request_text, str) and wants_article_summary(request_text)
        if wiki is not None:
            if wiki.get("status") != "ok":
                return "Wikipedia-Artikel konnte nicht gelesen werden: " + wiki.get("error", "Kein Text verfügbar."), False
            if wiki.get("disambiguation"):
                return ("Wikipedia: " + wiki.get("title", "") +
                        ". Das ist eine Begriffsklärung. Welchen Begriff meinst du genau?"), False
            if not summary:
                suffix = " Der verfügbare Auszug endet hier; der Artikel ist länger." if wiki.get("truncated") else ""
                return "Wikipedia: " + wiki.get("title", "") + ".\n" + wiki.get("text", "") + suffix, False
            return None, True
        if search is not None and article is None:
            if search.get("status") != "ok":
                return "Wikipedia-Suche fehlgeschlagen: " + search.get("error", "Unbekannter Fehler."), False
            results = search.get("results", [])
            return ("Wikipedia-Treffer: " + " ".join(
                f"{i}. {result['title']}." for i, result in enumerate(results, 1)
            ) if results else "Keine passenden Wikipedia-Artikel gefunden."), False
        if article is not None and (not summary or article.get("status") != "ok"):
            reply = (article["text"] if article.get("status") == "ok" and article.get("text")
                     else "Artikel konnte nicht gelesen werden: " + article.get("error", "Kein Text verfügbar."))
            return reply, False
        reuse = summary and references_previous_article(request_text)
        return None, bool(summary and (article is not None or (
            reuse and (latest_tool_result(messages, "read_article") is not None
                       or latest_tool_result(messages, "read_wikipedia") is not None)
        )))

    async def get_chat_completions(self, context: LLMContext):
        messages = context.get_messages()
        user_turn = bool(messages and messages[-1].get("role") == "user")
        text = messages[-1].get("content", "") if user_turn else ""
        is_text = isinstance(text, str)
        extension_reply = (
            await self.extensions.on_user_text(context, text)
            if user_turn and is_text and self.extensions else None
        )
        if isinstance(extension_reply, str):
            return self._direct_reply(extension_reply)
        extension_handled = bool(extension_reply)
        headlines = latest_tool_result(context.get_messages(), "read_news")
        if headlines and headlines.get("status") == "ok":
            self.last_headlines = headlines.get("headlines", [])
        dispatched = None
        if user_turn and is_text and not extension_handled:
            dispatched = await dispatch_deterministic_trigger(
                self.config, context, text, self.last_headlines
            )
        if dispatched and dispatched[0] == "wikipedia_selection":
            return self._direct_reply(dispatched[1]["reply"])
        article_reply, article_summary = self._article_policy(context)
        if article_reply is not None:
            return self._direct_reply(article_reply, title="Artikel vorlesen")
        prepared = bool(dispatched) or article_summary
        local_query = (
            detect_local_search(text, self.config.local_search_roots)
            if is_text and not extension_handled and not prepared else None
        )
        knowledge_result = (
            await asyncio.to_thread(find_relevant_search, self.knowledge_store, text)
            if is_text and not local_query and not extension_handled and not prepared and self.knowledge_store
            else None
        )
        adapted_tools = self.get_llm_adapter().from_standard_tools(context.tools)
        tools = list(adapted_tools) if is_given(adapted_tools) and adapted_tools else []
        if article_summary:
            tools = []
        elif dispatched:
            tools = [tool for tool in tools if tool["function"]["name"] != dispatched[0]]
        if local_query:
            await self._speak_wait(random.choice(LOCAL_SEARCH_WAIT_PHRASES))
            # This is deliberately before payload construction: the turn-stopped
            # event is asynchronous and cannot guarantee tool-before-answer order.
            await trigger_tool(context, "search_local", local_query)
        elif knowledge_result:
            await self._speak_wait(random.choice(KNOWLEDGE_WAIT_PHRASES))
            await trigger_tool(
                context, "search_knowledge", {"query": text},
                params_extra={"precomputed_result": knowledge_result},
            )
        elif user_turn and not extension_handled and not prepared and not self.config.model_tools and tools:
            await self._speak_wait(random.choice(GENERIC_WAIT_PHRASES))
            await route_tool(self.config, context, tools, notify=self.notify)
            article_reply, article_summary = self._article_policy(context)
            if article_reply is not None:
                return self._direct_reply(article_reply, title="Artikel vorlesen")
            if article_summary:
                tools = []
        payload = build_request(self.config, context.get_messages())
        context.set_messages(payload["messages"])
        # Request-only additions must not accumulate in the persistent system prompt.
        payload["messages"] = [dict(message) for message in payload["messages"]]
        if self.config.model_tools and tools:
            payload["tools"] = tools
        elif tools:
            manifest = tool_manifest(tools)
            if payload["messages"] and payload["messages"][0]["role"] == "system":
                payload["messages"][0]["content"] += "\n" + manifest
            else:
                payload["messages"].insert(0, {"role": "system", "content": manifest})
        payload = budget_request(payload)
        if not self.config.model_tools:
            payload["messages"] = text_messages(payload["messages"])

        async def chunks():
            start = time.monotonic()
            first = True
            complete = False
            async with httpx.AsyncClient(timeout=httpx.Timeout(90, connect=5)) as client:
                async with client.stream(
                    "POST", f"{self.config.ollama_url}/api/chat", json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        event = json.loads(line)
                        if event.get("error"):
                            raise RuntimeError(event["error"])
                        message = event.get("message", {})
                        if message.get("thinking"):
                            raise RuntimeError("Ollama liefert Thinking trotz think=false.")
                        for tool_index, call in enumerate(message.get("tool_calls") or []):
                            function = call.get("function", {})
                            yield ChatCompletionChunk(
                                id="local",
                                object="chat.completion.chunk",
                                created=int(time.time()),
                                model=self.config.model,
                                choices=[
                                    {
                                        "index": 0,
                                        "delta": {
                                            "tool_calls": [
                                                {
                                                    "index": tool_index,
                                                    "id": f"call_{uuid4().hex}",
                                                    "type": "function",
                                                    "function": {
                                                        "name": function.get("name", ""),
                                                        "arguments": json.dumps(
                                                            function.get("arguments") or {}
                                                        ),
                                                    },
                                                }
                                            ]
                                        },
                                        "finish_reason": None,
                                    }
                                ],
                            )
                        content = message.get("content", "")
                        if content:
                            if first:
                                self.notify(
                                    {
                                        "type": "metric",
                                        "name": "llm_ms",
                                        "value": round((time.monotonic() - start) * 1000),
                                    }
                                )
                                first = False
                            yield ChatCompletionChunk(
                                id="local",
                                object="chat.completion.chunk",
                                created=int(time.time()),
                                model=self.config.model,
                                choices=[
                                    {
                                        "index": 0,
                                        "delta": {"content": content},
                                        "finish_reason": None,
                                    }
                                ],
                            )
                        if event.get("done"):
                            complete = True
                    if not complete:
                        raise RuntimeError("Ollama hat den Antwortstream vorzeitig geschlossen.")

        async def with_wait_notice():
            notice = (
                asyncio.create_task(self._delayed_wait())
                if user_turn and not local_query and self.config.model_tools else None
            )
            operation_id = uuid4().hex
            started = time.monotonic()
            last_update = 0.0
            response_text = ""
            phase = "cancelled"
            self.notify({"type": "operation", "id": operation_id, "phase": "running",
                         "title": "Antwort formulieren", "detail": self.config.model})
            try:
                async with aclosing(chunks()) as stream:
                    async for chunk in stream:
                        if notice is not None:
                            notice.cancel()
                            await asyncio.gather(notice, return_exceptions=True)
                            notice = None
                        content = chunk.choices[0].delta.content
                        if content:
                            first_text = not response_text
                            response_text += content
                            now = time.monotonic()
                            if first_text or now - last_update >= 0.1:
                                self.notify({"type": "response", "id": operation_id,
                                             "phase": "start" if first_text else "delta",
                                             "text": response_text})
                                last_update = now
                        yield chunk
                phase = "done"
            except Exception:
                phase = "error"
                raise
            finally:
                if response_text:
                    self.notify({"type": "response", "id": operation_id,
                                 "phase": phase, "text": response_text})
                self.notify({"type": "operation", "id": operation_id, "phase": phase,
                             "title": "Antwort formulieren",
                             "elapsed_ms": round((time.monotonic() - started) * 1000)})
                if notice is not None:
                    notice.cancel()
                    await asyncio.gather(notice, return_exceptions=True)

        return with_wait_notice()

    def _direct_reply(self, reply: str, *, title: str = "Aufgabe bearbeiten"):
        async def stream():
            operation_id = uuid4().hex
            self.notify({"type": "operation", "id": operation_id,
                         "phase": "running", "title": title})
            self.notify({"type": "response", "id": operation_id,
                         "phase": "start", "text": reply})
            try:
                yield ChatCompletionChunk(
                    id="local-task", object="chat.completion.chunk",
                    created=int(time.time()), model=self.config.model,
                    choices=[{"index": 0, "delta": {"content": reply},
                              "finish_reason": None}],
                )
            finally:
                self.notify({"type": "response", "id": operation_id,
                             "phase": "done", "text": reply})
                self.notify({"type": "operation", "id": operation_id,
                             "phase": "done", "title": title})
        return stream()

    async def _speak_wait(self, text: str):
        if os.getenv("ASTRA_WAIT_SOUNDS", "1") == "0":
            return
        self.notify({"type": "activity", "text": text})
        await self.push_frame(TTSSpeakFrame(text=text, append_to_context=False))

    async def _delayed_wait(self):
        await asyncio.sleep(1)
        await self._speak_wait(random.choice(GENERIC_WAIT_PHRASES))


class NemotronSTTService(STTService):
    def __init__(self, models: Models, notify, no_speech_phrases: tuple[str, ...] | None = None):
        super().__init__(
            sample_rate=16000,
            audio_passthrough=True,
            ttfs_p99_latency=1.0,
            settings=STTSettings(model=models.settings.stt_model, language="de-DE"),
        )
        self.models = models
        self.notify = notify
        self.no_speech_phrases = no_speech_phrases
        self.queue = asyncio.Queue(maxsize=100)
        self.worker = None
        self.active = False
        self.preroll = bytearray()
        self.pending = bytearray()
        self.failed = False

    async def run_stt(self, audio):
        # Audio is fed by process_audio_frame to avoid blocking VAD on inference.
        if False:
            yield None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self.worker = self.create_task(self._worker())

    async def stop(self, frame: EndFrame):
        self._dump_close()
        if self.worker:
            await self.cancel_task(self.worker)
            self.worker = None
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        self._dump_close()
        if self.worker:
            await self.cancel_task(self.worker)
            self.worker = None
        await super().cancel(frame)

    def _enqueue(self, item):
        if self.failed:
            return
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull:
            self.failed = True
            self.notify(
                {"type": "error", "message": "Spracherkennung überlastet. Bitte neu verbinden."}
            )

    def _dump(self, pcm: bytes):
        """Diagnose: ASTRA_STT_DUMP_DIR schreibt das Audio vor der Erkennung als WAV mit."""
        directory = os.getenv("ASTRA_STT_DUMP_DIR", "")
        if not directory:
            return
        if getattr(self, "_dump_file", None) is None:
            import wave
            os.makedirs(directory, exist_ok=True)
            self._dump_file = wave.open(
                os.path.join(directory, time.strftime("stt-%Y%m%d-%H%M%S.wav")), "wb")
            self._dump_file.setnchannels(1)
            self._dump_file.setsampwidth(2)
            self._dump_file.setframerate(16000)
        self._dump_file.writeframes(pcm)

    def _dump_close(self):
        f = getattr(self, "_dump_file", None)
        if f is not None:
            f.close()
            self._dump_file = None

    async def process_audio_frame(self, frame: InputAudioRawFrame, direction: FrameDirection):
        if frame.sample_rate != 16000 or frame.num_channels != 1:
            raise ValueError("Nemotron benötigt 16 kHz Mono-PCM")
        self._dump(frame.audio)
        if self.active:
            self.pending.extend(frame.audio)
            while len(self.pending) >= 10240:
                self._enqueue(("audio", bytes(self.pending[:10240])))
                del self.pending[:10240]
        else:
            self.preroll.extend(frame.audio)
            # 600 ms of 16 kHz mono PCM preserve quiet word onsets before VAD fires.
            del self.preroll[:-19200]

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, VADUserStartedSpeakingFrame) and not self.active:
            self.active = True
            self._enqueue(("start", bytes(self.preroll)))
            self.preroll.clear()
        elif isinstance(frame, VADUserStoppedSpeakingFrame) and self.active:
            self.active = False
            self._enqueue(("final", bytes(self.pending)))
            self.pending.clear()
        await super().process_frame(frame, direction)

    async def _worker(self):
        recognizer = None
        previous = ""
        try:
            while True:
                kind, pcm = await self.queue.get()
                logger.debug("Nemotron chunk: {}, {} bytes", kind, len(pcm))
                if kind == "start":
                    recognizer = await on_executor(
                        self.models.stt_executor, Recognizer, self.models.stt
                    )
                    previous = ""
                if recognizer is None:
                    continue
                text = await on_executor(
                    self.models.stt_executor, recognizer.push, pcm, kind == "final"
                )
                logger.debug("Nemotron result: {}, {} characters", kind, len(text))
                if kind == "final":
                    if text:
                        await self.push_frame(
                            TranscriptionFrame(
                                text=text,
                                user_id="local",
                                timestamp=time_now_iso8601(),
                                finalized=True,
                            )
                        )
                    elif self.no_speech_phrases:
                        await self.push_frame(
                            TTSSpeakFrame(
                                text=random.choice(self.no_speech_phrases),
                                append_to_context=False,
                            )
                        )
                    self.notify({"type": "partial", "text": ""})
                    recognizer = None
                elif text and text != previous:
                    await self.push_frame(
                        InterimTranscriptionFrame(
                            text=text, user_id="local", timestamp=time_now_iso8601()
                        )
                    )
                    self.notify({"type": "partial", "text": text})
                    previous = text
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.failed = True
            self.notify({"type": "error", "message": f"Spracherkennung: {exc}"})
            await self.push_error_frame(ErrorFrame(error=f"Nemotron: {exc}", fatal=True))


def next_chunk(stream):
    return next(stream, None)


_POCKET_TTS_TRUNCATION = "Maximum generation length reached without EOS"


class _PocketTTSTruncationWatch(logging.Filter):
    """Count pocket_tts generations that ran out of budget before their own
    end-of-speech token - which silently cuts the audio short mid-sentence.

    Attaching to the package's stdlib logger is the only non-destructive way to
    see it happen: generate_audio_stream() has no return flag or exception for
    it (KPOCKET_TTS_ERROR_WITHOUT_EOS=1 turns it into a RuntimeError, but that
    discards the audio generated so far instead of playing it), and the budget
    itself is not a parameter either - TTSModel._estimate_max_gen_len derives it
    from the chunk's token count, so it can only be raised by patching the model.
    """

    def __init__(self):
        super().__init__()
        self.count = 0

    def filter(self, record: logging.LogRecord) -> bool:
        if _POCKET_TTS_TRUNCATION in record.getMessage():
            self.count += 1
        return True


_truncation_watch = _PocketTTSTruncationWatch()


_AUSSPRACHE: list | None = None


def aussprache(text: str) -> str:
    """Labor-Patch: Wörter vor der Sprachausgabe umschreiben (ASTRA_AUSSPRACHE_FILE, JSON {wort: lautschrift})."""
    global _AUSSPRACHE
    import re
    if _AUSSPRACHE is None:
        path = os.path.expanduser(os.getenv("ASTRA_AUSSPRACHE_FILE", ""))
        try:
            with open(path, encoding="utf-8") as f:
                table = json.load(f)
        except (OSError, ValueError):
            table = {}
        _AUSSPRACHE = [(re.compile(r"(?<!\w)" + re.escape(k) + r"(?!\w)", re.I), v)
                       for k, v in sorted(table.items(), key=lambda kv: -len(kv[0]))]
    for pattern, spoken in _AUSSPRACHE:
        text = pattern.sub(spoken, text)
    return text


class LocalPocketTTSService(TTSService):
    def __init__(self, models: Models, voice_state: dict, voice_name: str):
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            settings=TTSSettings(model="pocket-tts", voice=voice_name, language="de"),
        )
        self.models = models
        self.voice_state = voice_state
        logging.getLogger("pocket_tts.models.tts_model").addFilter(_truncation_watch)

    async def run_tts(self, text: str, context_id: str):
        import torch

        text = aussprache(text)
        truncations = _truncation_watch.count
        stream = self.models.tts.generate_audio_stream(self.voice_state, text, copy_state=True)

        async def audio():
            try:
                while True:
                    chunk = await on_executor(self.models.tts_executor, next_chunk, stream)
                    if chunk is None:
                        break
                    yield (chunk.clamp(-1, 1) * 32767).to(torch.int16).numpy().tobytes()
            finally:
                await on_executor(self.models.tts_executor, drain_stream, stream)

        try:
            async with aclosing(audio()) as audio_stream:
                async with aclosing(
                    self._stream_audio_frames_from_iterator(
                        audio_stream,
                        in_sample_rate=self.models.tts.sample_rate,
                        context_id=context_id,
                    )
                ) as frames:
                    async for frame in frames:
                        yield frame
            if _truncation_watch.count > truncations:
                logger.warning(
                    "Sprachausgabe vermutlich abgeschnitten: Pocket TTS erreichte die maximale "
                    "Generierungslänge ohne Satzende für „{}“", text,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield ErrorFrame(error=f"Sprachausgabe: {exc}")
