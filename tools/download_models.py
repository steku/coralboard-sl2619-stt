"""Model downloader for Wyoming Speech-to-Text on Coralboard SL2619 Torq NPU."""

import argparse
import logging
import os
import sys
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_LOGGER = logging.getLogger(__name__)

# Hugging Face download URLs for pre-compiled Synaptics Torq NPU Moonshine model & tokenizer
HF_TORQ_BASE = "https://huggingface.co/Synaptics/moonshine-tiny-bf16-torq/resolve/main"
HF_TOKENIZER_BACKUP = "https://huggingface.co/UsefulSensors/moonshine-tiny/resolve/main/tokenizer.json"

MODEL_FILES = {
    "tokenizer.json": [f"{HF_TORQ_BASE}/tokenizer.json", HF_TOKENIZER_BACKUP],
    "encoder.vmfb": [f"{HF_TORQ_BASE}/encoder.vmfb"],
    "decoder.vmfb": [f"{HF_TORQ_BASE}/decoder.vmfb"],
    "decoder_token_embeddings.npy": [f"{HF_TORQ_BASE}/decoder_token_embeddings.npy"],
}


def download_file(urls: list[str] | str, dest_path: str) -> bool:
    """Download a file from a list of fallback URLs with progress reporting."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        _LOGGER.info("File already exists, skipping: %s", dest_path)
        return True

    if isinstance(urls, str):
        urls = [urls]

    for url in urls:
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
            _LOGGER.info("Saved %s (%.2f MB)", dest_path, os.path.getsize(dest_path) / (1024 * 1024))
            return True
        except Exception as e:
            sys.stdout.write("\n")
            _LOGGER.warning("Failed to download from %s: %s", url, e)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    _LOGGER.error("All download attempts failed for %s", dest_path)
    return False


def create_alias(target: str, alias: str, base_dir: str):
    """Create symlink or copy for backwards-compatible naming."""
    target_path = os.path.join(base_dir, target)
    alias_path = os.path.join(base_dir, alias)
    if os.path.exists(target_path) and not os.path.exists(alias_path):
        try:
            os.symlink(target, alias_path)
            _LOGGER.info("Created alias symlink: %s -> %s", alias, target)
        except OSError:
            import shutil
            shutil.copyfile(target_path, alias_path)
            _LOGGER.info("Created alias copy: %s -> %s", alias, target)


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
    failed = False
    for filename, urls in MODEL_FILES.items():
        dest = os.path.join(args.models_dir, filename)
        if not download_file(urls, dest):
            failed = True

    if failed:
        _LOGGER.error("Some Torq NPU model artifacts failed to download. Please check network connectivity.")
        sys.exit(1)

    # Maintain backward compatibility aliases (encode.vmfb -> encoder.vmfb, decode.vmfb -> decoder.vmfb)
    create_alias("encoder.vmfb", "encode.vmfb", args.models_dir)
    create_alias("decoder.vmfb", "decode.vmfb", args.models_dir)

    _LOGGER.info("All Torq NPU model artifacts downloaded successfully.")


if __name__ == "__main__":
    main()
