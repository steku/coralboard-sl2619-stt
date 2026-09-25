"""CLI entrypoint for Wyoming Speech-to-Text on Coralboard SL2619 Torq NPU."""

import argparse
import asyncio
import logging
import sys

from .server import run_server


def main():
    parser = argparse.ArgumentParser(
        description="Wyoming Speech-to-Text Server for Home Assistant on Coralboard SL2619 Torq NPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host address to bind Wyoming TCP server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10300,
        help="TCP port to listen for Home Assistant connections",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to compiled Torq .vmfb model (e.g. models/encode.vmfb)",
    )
    parser.add_argument(
        "--decoder-path",
        default=None,
        help="Optional path to separate compiled decoder .vmfb (e.g. models/decode.vmfb)",
    )
    parser.add_argument(
        "--vocab-path",
        default=None,
        help="Path to tokenizer.json or vocab.json file",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Default language for transcription",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        asyncio.run(
            run_server(
                host=args.host,
                port=args.port,
                model_path=args.model_path,
                decoder_path=args.decoder_path,
                vocab_path=args.vocab_path,
                default_language=args.language,
            )
        )
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error("Fatal error during server execution: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
