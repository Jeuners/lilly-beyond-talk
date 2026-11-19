"""Bridge FreeSWITCH's mod_audio_stream WebSocket protocol into a Pipecat pipeline.

Wire protocol, verified against the free/open-source edition of
https://github.com/amigniter/mod_audio_stream (audio_streamer_glue.cpp:318-370
is the authority here, the README is close but this is what the module does):

FreeSWITCH -> here (this is what `deserialize` sees):
    - one text frame with whatever `metadata` string the dialplan's
      `uuid_audio_stream <uuid> start <url> <mix-type> <rate> <metadata>`
      passed, sent once right after the WebSocket connects
    - binary frames after that: raw L16 PCM, mono, at the rate requested in
      `start` (we request 16k so it lines up with NemotronSTTService's hard
      16 kHz requirement - no input resampling needed on this side)
    - occasional later text frames from `uuid_audio_stream <uuid> send_text`
      (e.g. DTMF) - not used yet, logged and dropped

here -> FreeSWITCH (`serialize`): JSON text frames only - the module has no
raw-binary playback path (and no dialplan application either, only the ESL API
command above - see the "Live-Telefonagent" plan for how the dialplan hands
the channel to an outbound-ESL bootstrap that issues it). Each `streamAudio`
message is written to a temp file and played as one unit
(audio_streamer_glue.cpp:377-390 handles "wav" directly, no separate sample-
rate whitelist the way "raw" needs), so bot speech is buffered per utterance
rather than streamed frame-by-frame the way the browser transport does - a
real, inherent trade-off of the free module versus WebRTC, not a bug to fix
here.
    {"type": "streamAudio", "data": {"audioDataType": "wav", "audioData": "<base64 WAV>"}}

The buffer is flushed on whichever of these happens first: it grows past
_MAX_BUFFER_SECONDS (a long answer), run_telephony_call calls flush_now()
because the pipeline reported the bot has genuinely stopped speaking - the
normal case - or (a fallback, should that notification somehow not arrive)
_FLUSH_IDLE_SECONDS pass with no new audio frame. NOT on TTSStoppedFrame
itself: confirmed 2026-09-13 by tracing every frame type
FastAPIWebsocketOutputTransport actually calls serialize() with - only
OutputAudioRawFrame and InterruptionFrame ever arrive here. The transport's
own process_frame() only forwards InterruptionFrame on to serialize() as a
control frame; TTSStoppedFrame is consumed upstream (drives "bot stopped
speaking" bookkeeping instead) and never reaches this class - hence
flush_now() instead of handling TTSStoppedFrame directly. A short idle
timeout alone doesn't work either: a multi-sentence answer has a real gap
between each sentence's TTS generation finishing and the next one's audio
arriving, long enough that a timeout short enough to feel responsive after a
one-sentence answer also splits every multi-sentence one into separate,
audibly disjointed streamAudio messages (confirmed live - "hackig"). Because
both flush_now() and the idle fallback fire outside of serialize()'s return
value, they need their own way to send: see `send` in __init__.

No "stop current playback" message exists in the module's processMessage
handler (unrecognized `type` values are silently ignored, audio_streamer_glue.cpp:318-319)
- genuine barge-in (cutting off audio FreeSWITCH is already playing) would need
`uuid_break` sent over the ESL control connection, which only the FreeSWITCH-
side bootstrap has, not this WebSocket. On InterruptionFrame we still clear our
own buffer so we don't belatedly send stale audio, but we can't stop what's
already mid-playback there - flagged, not solved, here.
"""

import asyncio
import base64
import json
import os

import numpy as np
from loguru import logger
from pipecat.audio.utils import create_stream_resampler, pcm_to_wav
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.serializers.base_serializer import FrameSerializer

# Wave-wrapped playback rate sent back to FreeSWITCH. Fixed rather than reusing
# the pipeline's audio_out_sample_rate (24 kHz, Pocket TTS's native rate) so the
# WAV header FreeSWITCH ends up playing is always the same, well-supported rate.
_PLAYBACK_SAMPLE_RATE = 16000

# Pegelabsenkung fuer die Telefonleitung in dB (negativ = leiser). Der Pocket-TTS-
# Pegel (Spitze ~-3 dBFS) klang ueber das Mobilfunknetz scheppernd, vermutlich
# durch Uebersteuerung beim Umcodieren im Netz des Anrufers.
def _gain_db_from_env() -> float:
    """Ungueltiger Wert -> 0 dB mit Warnung, statt Astra beim Import abstuerzen zu lassen."""
    raw = os.environ.get("ASTRA_TELEFON_GAIN_DB", "0")
    try:
        return float(raw)
    except ValueError:
        logger.warning("ASTRA_TELEFON_GAIN_DB={!r} ist keine Zahl, verwende 0 dB", raw)
        return 0.0


_PLAYBACK_GAIN_DB = _gain_db_from_env()

# Upper bound on how long a single buffered flush may span, so one very long
# answer doesn't grow the base64 payload unboundedly. Deliberately far above a
# normal answer (the phone prompt asks for one to two short sentences, three to
# six seconds): splitting one utterance across two `streamAudio` messages means
# the second arrives while FreeSWITCH is still playing the first, and the module
# has no playback queue - the tail is then simply never heard. This is a payload
# guard, not a streaming mechanism.
_MAX_BUFFER_SECONDS = 12.0
# Fallback only - flush_now() (see module docstring) is the real flush
# trigger. This has to clear the gap between one sentence's TTS audio ending
# and the next sentence's starting (regularly 1-2s: the LLM is still
# streaming the next sentence's text at that point), or it would flush - and
# so split into audibly separate messages - every multi-sentence answer on
# its own, same failure mode as before but from this timeout instead of the
# old buffer cap. Set high enough it should in practice never fire.
_FLUSH_IDLE_SECONDS = 4.0


async def _discard(_payload: str) -> None:
    """Default `send` when the caller doesn't need idle-flush delivery (e.g.
    tests exercising serialize()'s return value directly)."""


class FreeswitchAudioStreamSerializer(FrameSerializer):
    """Adapts one mod_audio_stream WebSocket connection to Pipecat frames."""

    def __init__(self, *, input_sample_rate: int = 16000, notify=None, send=None):
        super().__init__()
        self._input_sample_rate = input_sample_rate
        self._notify = notify or (lambda _payload: None)
        # The idle-flush path (see _schedule_idle_flush) fires from a
        # background task, not from serialize()'s return value, so it needs
        # its own way out - the transport's own send, e.g. `websocket.send_text`.
        self._send = send or _discard
        self._call_metadata: dict | str | None = None
        self._output_resampler = create_stream_resampler()
        self._buffer = bytearray()
        self._buffer_rate: int | None = None
        self._flush_task: asyncio.Task | None = None

    async def setup(self, setup: FrameProcessorSetup):
        pass

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, InterruptionFrame):
            # See module docstring: nothing we send here can stop audio
            # FreeSWITCH already started playing - this only prevents
            # belatedly flushing audio that's now stale.
            self._cancel_idle_flush()
            self._buffer.clear()
            self._buffer_rate = None
            return None
        if isinstance(frame, OutputAudioRawFrame):
            if self._buffer_rate is not None and self._buffer_rate != frame.sample_rate:
                # A rate change mid-buffer would corrupt the WAV we send - flush
                # what we have at the old rate first (should not normally happen,
                # the TTS service's output rate is fixed for a session).
                flushed = await self._flush()
                if flushed:
                    logger.warning("Telefonie: TTS-Ausgaberate wechselte mitten im Puffer")
                    return flushed
            self._buffer_rate = frame.sample_rate
            self._buffer.extend(frame.audio)
            if self._buffered_seconds() >= _MAX_BUFFER_SECONDS:
                logger.warning(
                    "Telefonie: Antwort überschreitet {}s, wird in mehreren Abschnitten "
                    "gesendet - FreeSWITCH spielt davon vermutlich nur den ersten",
                    _MAX_BUFFER_SECONDS,
                )
                self._cancel_idle_flush()
                return await self._flush()
            self._schedule_idle_flush()
            return None
        return None

    async def flush_now(self):
        """Explicit flush trigger for run_telephony_call's notify() hook -
        called when the pipeline reports the bot has genuinely stopped
        speaking (BotStoppedSpeakingFrame), which fires exactly once per
        whole answer even when the TTS service split it into several
        sentence chunks (each generated - and arriving here - with a real
        gap between them, long enough that a short idle timeout would have
        cut the answer into separate streamAudio messages and made it sound
        chopped up - confirmed live). This is the primary flush path;
        _schedule_idle_flush below is only a fallback for the (untested)
        case this notification is somehow missed. Safe to call redundantly
        (e.g. on every "listening" state, not just after real speech) - a
        no-op if nothing is buffered.
        """
        self._cancel_idle_flush()
        await self._flush_and_send()

    def _schedule_idle_flush(self):
        self._cancel_idle_flush()
        self._flush_task = asyncio.create_task(self._flush_after_idle())

    def _cancel_idle_flush(self):
        if self._flush_task is not None and not self._flush_task.done():
            self._flush_task.cancel()
        self._flush_task = None

    async def _flush_after_idle(self):
        try:
            await asyncio.sleep(_FLUSH_IDLE_SECONDS)
        except asyncio.CancelledError:
            return
        await self._flush_and_send()

    async def _flush_and_send(self):
        payload = await self._flush()
        if payload is None:
            return
        try:
            await self._send(payload)
        except Exception:
            # Best-effort - the call may have ended in the meantime, the
            # websocket this would send on already gone with it.
            logger.debug("Telefonie: Flush konnte nicht gesendet werden, Anruf vorbei?")

    def _buffered_seconds(self) -> float:
        """Duration of the buffer at the rate it was actually filled at - the
        TTS output rate (24 kHz), not _PLAYBACK_SAMPLE_RATE. Measuring it at
        the playback rate made the cap fire after 2.67s instead of the intended
        4s and cut greetings in half (verified against a live call whose
        FreeSWITCH playback lasted exactly 2.64s).
        """
        rate = self._buffer_rate or _PLAYBACK_SAMPLE_RATE
        return len(self._buffer) / (rate * 2)

    async def _flush(self) -> str | None:
        if not self._buffer:
            return None
        pcm = bytes(self._buffer)
        rate = self._buffer_rate or _PLAYBACK_SAMPLE_RATE
        seconds = self._buffered_seconds()
        self._buffer.clear()
        self._buffer_rate = None
        resampled = await self._output_resampler.resample(pcm, rate, _PLAYBACK_SAMPLE_RATE)
        if _PLAYBACK_GAIN_DB:
            samples = np.frombuffer(resampled, dtype=np.int16).astype(np.float32)
            samples *= 10 ** (_PLAYBACK_GAIN_DB / 20)
            resampled = np.clip(samples, -32768, 32767).astype(np.int16).tobytes()
        wav = pcm_to_wav(resampled, _PLAYBACK_SAMPLE_RATE, num_channels=1)
        payload = base64.b64encode(wav).decode("ascii")
        # So the log says how much audio FreeSWITCH was handed, which is the
        # only way to tell a short playback there apart from a short generation
        # here without listening to the call.
        logger.info("Telefonie: {:.2f}s Audio gesendet ({} Bytes WAV)", seconds, len(wav))
        return json.dumps(
            {
                "type": "streamAudio",
                "data": {"audioDataType": "wav", "audioData": payload},
            }
        )

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, (bytes, bytearray)):
            self._binary_frames = getattr(self, "_binary_frames", 0) + 1
            if self._binary_frames <= 3 or self._binary_frames % 100 == 0:
                import struct

                samples = struct.unpack(f"<{len(data) // 2}h", bytes(data)) if data else ()
                peak = max((abs(s) for s in samples), default=0)
                logger.debug(
                    "Telefonie: Binaer-Frame #{} ({} Bytes, Peak {})",
                    self._binary_frames, len(data), peak,
                )
            return InputAudioRawFrame(
                audio=bytes(data), sample_rate=self._input_sample_rate, num_channels=1
            )
        if self._call_metadata is None:
            try:
                self._call_metadata = json.loads(data)
            except (TypeError, ValueError):
                self._call_metadata = data
            self._notify({"type": "call_metadata", "metadata": self._call_metadata})
            logger.info("Telefonie: Anruf verbunden, Metadaten: {}", self._call_metadata)
        else:
            logger.debug("Telefonie: unerwarteter Text-Frame von FreeSWITCH: {}", data)
        return None
