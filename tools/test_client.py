"""Test client to verify Wyoming STT service on Coralboard SL2619."""

import argparse
import asyncio
import io
import logging
import sys
import time
import wave
import numpy as np

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient
from wyoming.info import Describe, Info

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_LOGGER = logging.getLogger(__name__)


def generate_synthetic_audio(duration_sec: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Generate 16-bit mono PCM sine wave test tone."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # Generate 440 Hz tone
    samples = (np.sin(2 * np.pi * 440 * t) * 16384).astype(np.int16)
    return samples.tobytes()


def load_wav_file(wav_path: str) -> tuple[bytes, int, int, int]:
    """Load WAV audio file and return (raw_bytes, sample_rate, sample_width, channels)."""
    with wave.open(wav_path, "rb") as wf:
        rate = wf.getframerate()
        width = wf.getsampwidth()
        channels = wf.getnchannels()
        raw_pcm = wf.readframes(wf.getnframes())
    return raw_pcm, rate, width, channels


async def test_wyoming_server(
    host: str,
    port: int,
    wav_path: str = None,
    language: str = "en",
    chunk_size: int = 1024,
) -> None:
    uri = f"tcp://{host}:{port}"
    _LOGGER.info("Connecting to Wyoming STT server at %s...", uri)

    async with AsyncClient.from_uri(uri) as client:
        # 1. Test Describe
        _LOGGER.info("Sending Describe request...")
        await client.write_event(Describe().event())
        info_event = await client.read_event()

        if info_event and Info.is_type(info_event.type):
            info = Info.from_event(info_event)
            _LOGGER.info("Successfully discovered server programs:")
            for p in info.asr:
                _LOGGER.info(" - Program: %s (v%s)", p.name, p.version)
                for m in p.models:
                    _LOGGER.info("   * Model: %s (languages: %s)", m.name, len(m.languages))
        else:
            _LOGGER.warning("Did not receive Info response for Describe: %s", info_event)

        # 2. Prepare audio
        if wav_path:
            _LOGGER.info("Loading test WAV audio from %s...", wav_path)
            pcm_bytes, rate, width, channels = load_wav_file(wav_path)
        else:
            _LOGGER.info("Generating synthetic 2.0s PCM test audio...")
            pcm_bytes = generate_synthetic_audio(duration_sec=2.0)
            rate = 16000
            width = 2
            channels = 1

        total_bytes = len(pcm_bytes)
        duration_sec = total_bytes / (rate * width * channels)
        _LOGGER.info("Audio ready: %d bytes (%.2f seconds @ %d Hz)", total_bytes, duration_sec, rate)

        # 3. Send Transcribe session setup
        await client.write_event(Transcribe(language=language).event())

        # 4. Stream audio
        start_time = time.perf_counter()
        await client.write_event(
            AudioStart(rate=rate, width=width, channels=channels).event()
        )

        # Stream chunks (simulate live streaming)
        offset = 0
        while offset < total_bytes:
            chunk = pcm_bytes[offset : offset + chunk_size]
            await client.write_event(
                AudioChunk(
                    audio=chunk,
                    rate=rate,
                    width=width,
                    channels=channels,
                ).event()
            )
            offset += chunk_size
            await asyncio.sleep(0.001)

        await client.write_event(AudioStop().event())
        _LOGGER.info("Finished streaming audio. Waiting for Transcript...")

        # 5. Receive Transcript
        while True:
            event = await client.read_event()
            if event is None:
                _LOGGER.error("Connection closed before Transcript received.")
                break
            if Transcript.is_type(event.type):
                transcript = Transcript.from_event(event)
                elapsed = time.perf_counter() - start_time
                _LOGGER.info("=== TRANSCRIPTION RESULT ===")
                _LOGGER.info("Text: '%s'", transcript.text)
                _LOGGER.info("Total Roundtrip Time: %.3f s (Audio duration: %.2f s)", elapsed, duration_sec)
                break
            else:
                _LOGGER.debug("Received event: %s", event.type)


def main():
    parser = argparse.ArgumentParser(description="Test client for Wyoming STT Server on Coralboard SL2619")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=10300, help="Server port")
    parser.add_argument("--wav", default=None, help="Path to input .wav file (optional)")
    parser.add_argument("--language", default="en", help="Language code")

    args = parser.parse_args()

    try:
        asyncio.run(
            test_wyoming_server(
                host=args.host,
                port=args.port,
                wav_path=args.wav,
                language=args.language,
            )
        )
    except Exception as e:
        _LOGGER.error("Test failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
