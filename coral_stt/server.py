"""Wyoming Speech-to-Text Server runner for Coralboard SL2619 Torq NPU."""

import asyncio
import logging
import os
import signal
from typing import Optional

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from .engine.base import STTEngine
from .engine.torq_engine import TorqSTTEngine
from .handler import WyomingSTTHandler

_LOGGER = logging.getLogger(__name__)


def build_wyoming_info(engine: STTEngine) -> Info:
    """Build Wyoming Info object for Home Assistant service discovery."""
    return Info(
        asr=[
            AsrProgram(
                name="coral-sl2619-stt",
                description="Speech-to-Text on Coralboard SL2619 Torq NPU",
                attribution=Attribution(
                    name="Synaptics Astra / Google Coral NPU",
                    url="https://github.com/synaptics-torq",
                ),
                installed=True,
                version="1.0.0",
                models=[
                    AsrModel(
                        name=engine.model_name,
                        description=f"Model: {engine.model_name} (Coralboard SL2619 NPU)",
                        attribution=Attribution(
                            name="Synaptics / Useful Sensors",
                            url="https://huggingface.co/Synaptics/moonshine-tiny-bf16-torq",
                        ),
                        installed=True,
                        languages=["en"],
                    )
                ],
            )
        ]
    )


def init_npu_engine(
    model_path: Optional[str] = None,
    decoder_path: Optional[str] = None,
    vocab_path: Optional[str] = None,
) -> STTEngine:
    """Initialize and load the Torq NPU engine."""
    # Default model path lookup if not specified
    if model_path is None:
        default_candidates = [
            "models/encoder.vmfb",
            "models/encode.vmfb",
            os.path.join(os.path.dirname(__file__), "..", "models", "encoder.vmfb"),
            os.path.join(os.path.dirname(__file__), "..", "models", "encode.vmfb"),
        ]
        for candidate in default_candidates:
            if os.path.exists(candidate):
                model_path = candidate
                break

    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Torq NPU model not found: '{model_path}'. "
            "Please run 'python3 tools/download_models.py' or specify a valid .vmfb via --model-path."
        )

    # Default decoder lookup
    if decoder_path is None:
        for candidate in [
            "models/decoder.vmfb",
            "models/decode.vmfb",
            os.path.join(os.path.dirname(__file__), "..", "models", "decoder.vmfb"),
            os.path.join(os.path.dirname(__file__), "..", "models", "decode.vmfb"),
        ]:
            if os.path.exists(candidate):
                decoder_path = candidate
                break

    # Default vocab lookup
    if vocab_path is None:
        for candidate in [
            "models/tokenizer.json",
            os.path.join(os.path.dirname(__file__), "..", "models", "tokenizer.json"),
        ]:
            if os.path.exists(candidate):
                vocab_path = candidate
                break

    _LOGGER.info("Initializing Coralboard SL2619 Torq NPU engine (model: %s)", model_path)
    engine = TorqSTTEngine(
        model_path=model_path,
        decoder_path=decoder_path,
        vocab_path=vocab_path,
    )
    engine.load()
    return engine


async def run_server(
    host: str = "0.0.0.0",
    port: int = 10300,
    model_path: Optional[str] = None,
    decoder_path: Optional[str] = None,
    vocab_path: Optional[str] = None,
    default_language: str = "en",
) -> None:
    """Start Wyoming TCP server for Home Assistant."""
    engine = init_npu_engine(
        model_path=model_path,
        decoder_path=decoder_path,
        vocab_path=vocab_path,
    )

    wyoming_info = build_wyoming_info(engine)
    uri = f"tcp://{host}:{port}"
    _LOGGER.info("Starting Wyoming server on %s", uri)

    server = AsyncServer.from_uri(uri)

    def create_handler(*args, **kwargs):
        return WyomingSTTHandler(
            engine,
            wyoming_info,
            *args,
            default_language=default_language,
            **kwargs,
        )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    server_task = asyncio.create_task(server.run(create_handler))
    _LOGGER.info("Wyoming server is now active and listening on %s (Port %d)", host, port)

    try:
        await stop_event.wait()
    finally:
        _LOGGER.info("Shutting down Wyoming server...")
        server_task.cancel()
        await asyncio.gather(server_task, return_exceptions=True)
        _LOGGER.info("Server stopped.")
