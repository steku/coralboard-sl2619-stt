"""Tokenizer utilities for speech recognition models on Coralboard SL2619."""

import json
import logging
import os
from typing import Dict, List, Optional

_LOGGER = logging.getLogger(__name__)


class STTTokenizer:
    """Tokenizer wrapper supporting Hugging Face tokenizers and vocabulary mappings."""

    def __init__(self, vocab_file: Optional[str] = None):
        self._hf_tokenizer = None
        self._id_to_token: Dict[int, str] = {}
        self._token_to_id: Dict[str, int] = {}

        # 1. Search candidate locations if vocab_file not specified
        if not vocab_file:
            candidates = [
                "models/tokenizer.json",
                os.path.join(os.path.dirname(__file__), "..", "models", "tokenizer.json"),
                os.path.join(os.path.dirname(__file__), "tokenizer.json"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    vocab_file = c
                    break

        # 2. Try Hugging Face tokenizers library
        if vocab_file and os.path.exists(vocab_file):
            try:
                from tokenizers import Tokenizer
                self._hf_tokenizer = Tokenizer.from_file(vocab_file)
                _LOGGER.info("Loaded tokenizer from %s", vocab_file)
            except Exception as e:
                _LOGGER.warning("Could not load with tokenizers.Tokenizer: %s", e)

        # 3. Fallback: load as json dictionary
        if self._hf_tokenizer is None and vocab_file and os.path.exists(vocab_file):
            try:
                with open(vocab_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "model" in data and "vocab" in data["model"]:
                        self._token_to_id = data["model"]["vocab"]
                    elif isinstance(data, dict):
                        self._token_to_id = data
                    self._id_to_token = {v: k for k, v in self._token_to_id.items()}
                _LOGGER.info("Loaded vocabulary map from %s (%d tokens)", vocab_file, len(self._id_to_token))
            except Exception as e:
                _LOGGER.warning("Could not parse vocabulary file %s: %s", vocab_file, e)

    def token_to_id(self, token: str) -> Optional[int]:
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.token_to_id(token)
        return self._token_to_id.get(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.id_to_token(token_id)
        return self._id_to_token.get(token_id)

    @property
    def start_token(self) -> int:
        """Start-of-transcript token ID (defaults to 1 for Moonshine)."""
        tid = self.token_to_id("<|startoftranscript|>")
        if tid is not None:
            return tid
        tid = self.token_to_id("<s>")
        return tid if tid is not None else 1

    @property
    def end_token(self) -> int:
        """End-of-transcript token ID (defaults to 2 for Moonshine)."""
        tid = self.token_to_id("<|endoftranscript|>")
        if tid is not None:
            return tid
        tid = self.token_to_id("</s>")
        if tid is not None:
            return tid
        tid = self.token_to_id("<|endoftext|>")
        return tid if tid is not None else 2

    def encode(self, text: str) -> List[int]:
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.encode(text).ids
        return [self._token_to_id.get(c, ord(c)) for c in text]

    def decode(self, token_ids: List[int]) -> str:
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.decode(token_ids)
        if self._id_to_token:
            tokens = [self._id_to_token.get(t, "") for t in token_ids]
            return "".join(tokens).replace(" ", " ").strip()
        return bytes([t for t in token_ids if t < 256]).decode("utf-8", errors="replace")
