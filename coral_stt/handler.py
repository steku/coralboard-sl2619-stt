"""Wyoming protocol event handler for Speech-to-Text on Coralboard SL2619."""

import logging
from typing import Optional

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .engine.base import STTEngine

_LOGGER = logging.getLogger(__name__)


class WyomingSTTHandler(AsyncEventHandler):
    """Handles Wyoming protocol events from Home Assistant."""

    def __init__(
        self,
        engine: STTEngine,
        wyoming_info: Info,
        *args,
        default_language: Optional[str] = "en",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._engine = engine
        self._wyoming_info = wyoming_info
        self._default_language = default_language

        # Session state
        self._language: Optional[str] = default_language
        self._sample_rate: int = 16000
        self._sample_width: int = 2
        self._channels: int = 1
        self._audio_buffer = bytearray()

    async def handle_event(self, event: Event) -> bool:
        """Process incoming Wyoming event."""
        # 1. Describe -> Return Info
        if Describe.is_type(event.type):
            _LOGGER.debug("Received Describe request from client.")
            await self.write_event(self._wyoming_info.event())
            return True

        # 2. Transcribe -> Set language/model preferences
        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            if transcribe.language:
                self._language = transcribe.language
                _LOGGER.debug("Client requested language: %s", self._language)
            return True

        # 3. AudioStart -> Initialize streaming audio buffer
        if AudioStart.is_type(event.type):
            audio_start = AudioStart.from_event(event)
            self._sample_rate = audio_start.rate
            self._sample_width = audio_start.width
            self._channels = audio_start.channels
            self._audio_buffer.clear()
            _LOGGER.debug(
                "Audio stream started: rate=%d, width=%d, channels=%d",
                self._sample_rate,
                self._sample_width,
                self._channels,
            )
            return True

        # 4. AudioChunk -> Accumulate raw PCM chunks
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self._audio_buffer.extend(chunk.audio)
            return True

        # 5. AudioStop -> Stream finished, run inference and return Transcript
        if AudioStop.is_type(event.type):
            _LOGGER.debug(
                "Audio stream ended. Total received: %d bytes (%.2f seconds)",
                len(self._audio_buffer),
                len(self._audio_buffer) / (self._sample_rate * self._sample_width * self._channels)
                if self._sample_rate > 0
                else 0,
            )

            raw_bytes = bytes(self._audio_buffer)
            self._audio_buffer.clear()

            # Execute transcription via active engine
            try:
                transcript_text = self._engine.transcribe(
                    audio_pcm=raw_bytes,
                    sample_rate=self._sample_rate,
                    language=self._language or self._default_language,
                )
            except Exception as e:
                _LOGGER.exception("Error during speech inference: %s", e)
                transcript_text = ""

            # Send Transcript event back to Home Assistant
            _LOGGER.info("Sending Transcript: '%s'", transcript_text)
            await self.write_event(Transcript(text=transcript_text).event())
            return True

        _LOGGER.debug("Unhandled event type: %s", event.type)
        return True
