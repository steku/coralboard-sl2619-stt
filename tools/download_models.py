"""Model downloader for Wyoming Speech-to-Text on Coralboard SL2619 Torq NPU."""

import argparse
import logging
import os
import sys
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_LOGGER = logging.getLogger(__name__)

# Hugging Face download URLs for pre-compiled Synaptics Torq NPU Moonshine model & tokenizer
HF_TORQ_BASE = "https://huggingface.co/Synaptics/moonshine-tiny-bf16-torq/resolve/main/moonshine"
HF_TOKENIZER_URL = "https://huggingface.co/UsefulSensors/moonshine-tiny/resolve/main/tokenizer.json"

MODEL_FILES = {
    "tokenizer.json": HF_TOKENIZER_URL,
    "encode.vmfb": f"{HF_TORQ_BASE}/encode.vmfb",
    "decode.vmfb": f"{HF_TORQ_BASE}/decode.vmfb",
}


def download_file(url: str, dest_path: str) -> bool:
    """Download a file with progress reporting."""
    if os.path.exists(dest_path):
        _LOGGER.info("File already exists, skipping: %s", dest_path)
        return True

    _LOGGER.info("Downloading %s -> %s...", url, dest_path)
    tmp_path = dest_path + ".tmp"
    try:
        def reporthook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100.0, block_num * block_size * 100.0 / total_size)
                sys.stdout.write(f"\r  Progress: {percent:.1f}% ({block_num * block_size / (1024*1024):.2f} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, tmp_path, reporthook=reporthook)
        sys.stdout.write("\n")
        os.rename(tmp_path, dest_path)
        _LOGGER.info("Saved %s", dest_path)
        return True
    except Exception as e:
        sys.stdout.write("\n")
        _LOGGER.warning("Failed to download %s: %s", url, e)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download speech models and tokenizers for Coralboard SL2619 Torq NPU"
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Target directory to store model weights and tokenizers",
    )

    args = parser.parse_args()
    os.makedirs(args.models_dir, exist_ok=True)

    _LOGGER.info("Downloading Torq NPU model artifacts into '%s'...", args.models_dir)
    for filename, url in MODEL_FILES.items():
        dest = os.path.join(args.models_dir, filename)
        download_file(url, dest)

    _LOGGER.info("All Torq NPU model artifacts downloaded successfully.")


if __name__ == "__main__":
    main()
