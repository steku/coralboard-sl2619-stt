"""Synaptics Torq NPU Speech-to-Text Engine for Coralboard SL2619.

Runs speech-to-text models compiled for the Torq NPU (Google Coral Kelvin ML Core)
via the Torq Runtime Python API.
"""

import logging
import os
import time
from typing import List, Optional, Tuple
import numpy as np

from ..audio import log_mel_spectrogram, pcm16_to_float32, resample_linear, SAMPLE_RATE
from ..tokenizer import STTTokenizer
from .base import STTEngine

_LOGGER = logging.getLogger(__name__)


def _import_vmfb_runner():
    """Import VMFBInferenceRunner supporting both torq.runtime and legacy paths."""
    # 1. Official Synaptics torq.runtime package
    try:
        from torq.runtime import VMFBInferenceRunner
        return VMFBInferenceRunner
    except ImportError:
        pass

    # 2. Legacy torq_runtime package
    try:
        from torq_runtime import VMFBInferenceRunner
        return VMFBInferenceRunner
    except ImportError:
        pass

    # 3. Check system site-packages if running inside a virtualenv
    import sys
    system_candidates = [
        "/usr/lib/python3.12/site-packages",
        "/usr/local/lib/python3.12/site-packages",
        "/usr/lib/python3/dist-packages",
    ]
    for candidate_dir in system_candidates:
        if os.path.exists(candidate_dir) and candidate_dir not in sys.path:
            sys.path.append(candidate_dir)

    try:
        from torq.runtime import VMFBInferenceRunner
        return VMFBInferenceRunner
    except ImportError:
        pass

    try:
        from torq_runtime import VMFBInferenceRunner
        return VMFBInferenceRunner
    except ImportError:
        pass

    # 4. Underlying iree.runtime fallback
    try:
        from iree.runtime import VMFBInferenceRunner
        return VMFBInferenceRunner
    except ImportError:
        pass

    return None


def _get_bf16_dtype():
    """Get bfloat16 dtype from ml_dtypes if available."""
    try:
        import ml_dtypes
        return ml_dtypes.bfloat16
    except Exception:
        return np.float32


def _to_bf16(arr: np.ndarray) -> np.ndarray:
    """Convert numpy array to bfloat16 using ml_dtypes if available."""
    bf16_dt = _get_bf16_dtype()
    if arr.dtype == bf16_dt:
        return arr
    if arr.dtype.kind == "V" and arr.dtype.itemsize == 2:
        return arr.view(bf16_dt)
    return arr.astype(bf16_dt)



class TorqSTTEngine(STTEngine):
    """Speech recognition inference engine running on Coralboard SL2619 Torq NPU."""

    def __init__(
        self,
        model_path: str,
        decoder_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        max_tokens: int = 128,
    ):
        """
        Args:
            model_path: Path to compiled .vmfb model (encoder or end-to-end model).
            decoder_path: Optional path to compiled decoder .vmfb (if separated).
            vocab_path: Path to tokenizer.json file.
            max_tokens: Maximum tokens to generate per utterance.
        """
        self._model_path = model_path
        self._decoder_path = decoder_path
        self._vocab_path = vocab_path
        self._max_tokens = max_tokens

        self._encoder_runner = None
        self._decoder_runner = None
        self._tokenizer = None
        self._token_embeddings = None
        self._is_end_to_end = False

    @property
    def model_name(self) -> str:
        base = os.path.basename(self._model_path)
        return os.path.splitext(base)[0]

    def load(self) -> None:
        """Load compiled .vmfb artifacts into Torq NPU runtime."""
        _LOGGER.info("Initializing Coralboard SL2619 Torq NPU runtime...")

        VMFBInferenceRunner = _import_vmfb_runner()
        if VMFBInferenceRunner is None:
            _LOGGER.error(
                "Failed to import Torq runtime (`torq.runtime`)! "
                "Ensure torq-runtime is installed: "
                "pip install https://github.com/synaptics-torq/torq-compiler/releases/download/v2.1.0/torq_runtime-2.1.0-cp312-cp312-manylinux_2_28_aarch64.whl"
            )
            raise ImportError("Could not find `torq.runtime` or `torq_runtime`.")

        if not os.path.exists(self._model_path):
            raise FileNotFoundError(f"Model VMFB file not found: {self._model_path}")

        _LOGGER.info("Loading primary model on NPU: %s", self._model_path)
        self._encoder_runner = VMFBInferenceRunner(self._model_path)

        # Inspect if separate decoder VMFB is provided
        if self._decoder_path and os.path.exists(self._decoder_path):
            _LOGGER.info("Loading decoder model on NPU: %s", self._decoder_path)
            self._decoder_runner = VMFBInferenceRunner(self._decoder_path)

            # Check for decoder token embeddings
            emb_candidates = [
                os.path.join(os.path.dirname(self._decoder_path), "decoder_token_embeddings.npy"),
                os.path.join(os.path.dirname(self._model_path), "decoder_token_embeddings.npy"),
                "models/decoder_token_embeddings.npy",
            ]
            for candidate in emb_candidates:
                if os.path.exists(candidate):
                    try:
                        raw = np.load(candidate)
                        self._token_embeddings = _to_bf16(raw)
                        _LOGGER.info(
                            "Loaded decoder token embeddings from %s: shape %s, dtype %s",
                            candidate,
                            self._token_embeddings.shape,
                            self._token_embeddings.dtype,
                        )
                        break
                    except Exception as e:
                        _LOGGER.warning("Failed to load token embeddings from %s: %s", candidate, e)
        else:
            self._is_end_to_end = True

        self._tokenizer = STTTokenizer(self._vocab_path)
        _LOGGER.info("Torq NPU Engine ready on Coralboard SL2619.")

    def transcribe(
        self,
        audio_pcm: bytes,
        sample_rate: int = SAMPLE_RATE,
        language: Optional[str] = "en",
    ) -> str:
        """Execute NPU inference on raw PCM audio."""
        if self._encoder_runner is None:
            raise RuntimeError("TorqSTTEngine is not loaded. Call load() first.")

        start_time = time.perf_counter()

        # 1. Convert PCM to float32
        audio = pcm16_to_float32(audio_pcm)
        if len(audio) == 0:
            return ""

        # 2. Resample to 16 kHz if necessary
        if sample_rate != SAMPLE_RATE:
            audio = resample_linear(audio, sample_rate, SAMPLE_RATE)

        # 3. Format and chunk audio tensor:
        # Synaptics Torq Moonshine encoder is compiled for fixed 5.0-second (80,000 samples at 16 kHz) windows.
        CHUNK_SIZE = 80000
        chunks = []
        for i in range(0, len(audio), CHUNK_SIZE):
            chunk = audio[i : i + CHUNK_SIZE]
            if len(chunk) < CHUNK_SIZE:
                chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)), mode="constant")
            chunks.append(chunk)

        if not chunks:
            return ""

        # 4. Infer each chunk using Torq NPU and combine results
        transcripts = []
        for chunk in chunks:
            audio_input = chunk[np.newaxis, :].astype(np.float32)
            if self._is_end_to_end:
                chunk_text = self._infer_end_to_end(audio_input)
            else:
                chunk_text = self._infer_encoder_decoder(audio_input)
            if chunk_text:
                transcripts.append(chunk_text)

        text = " ".join(transcripts).strip()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        _LOGGER.info(
            "Transcription completed in %.2f ms (Audio length: %.2f s, %d chunks): '%s'",
            elapsed_ms,
            len(audio) / SAMPLE_RATE,
            len(chunks),
            text,
        )
        return text

    def _infer_end_to_end(self, audio_tensor: np.ndarray) -> str:
        """Execute unified model that directly produces token IDs or strings."""
        audio_in = _to_bf16(audio_tensor)
        outputs = self._encoder_runner.infer([audio_in])
        out_arr = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

        # If model outputs token sequence
        if np.issubdtype(out_arr.dtype, np.integer):
            token_ids = out_arr.flatten().tolist()
            return self._tokenizer.decode(token_ids).strip()

        # If model outputs character logits (CTC or argmax)
        if out_arr.ndim == 3:  # (1, T, Vocab)
            token_ids = np.argmax(out_arr[0], axis=-1).tolist()
            return self._tokenizer.decode(token_ids).strip()

        return str(out_arr).strip()

    def _infer_encoder_decoder(self, audio_tensor: np.ndarray) -> str:
        """Execute NPU-accelerated Encoder followed by decoder autoregressive generation.
        
        Exact ABI for Moonshine tiny on Torq NPU:
        - Encoder: input [1, 80000] (bf16) -> 12 outputs (6 layers of cross-attention key/val pairs)
        - Decoder: 26 inputs -> 13 outputs (logits + 12 updated past self-attention KV caches)
        """
        bf16_dt = _get_bf16_dtype()

        # Step A: Encoder forward pass on Torq NPU (expects bfloat16 input with shape 1x80000)
        audio_in = _to_bf16(audio_tensor)
        encoder_outputs = self._encoder_runner.infer([audio_in])
        if not isinstance(encoder_outputs, (list, tuple)) or len(encoder_outputs) < 12:
            _LOGGER.error("Expected 12 encoder outputs for 6 decoder layers, got %s", type(encoder_outputs))
            return ""

        # Step B: Autoregressive decoding
        start_tok = self._tokenizer.start_token
        end_tok = self._tokenizer.end_token
        current_token = start_tok
        generated_tokens: List[int] = []

        # Decoder self-attention cache has fixed sequence length dimension 30
        max_steps = min(self._max_tokens, 30)

        # Initialize 12 self-attention past KV caches (2 per layer x 6 layers)
        # Each has shape (1, 8, 30, 36) and dtype bfloat16
        past_kvs = [np.zeros((1, 8, 30, 36), dtype=bf16_dt) for _ in range(12)]

        for step in range(max_steps):
            # 1. Token embedding for current_token: shape (1, 1, 288), bfloat16
            if self._token_embeddings is not None:
                token_vec = self._token_embeddings[current_token : current_token + 1]
                token_emb = np.expand_dims(token_vec, axis=0)  # shape (1, 1, 288)
                if token_emb.dtype != bf16_dt:
                    token_emb = _to_bf16(token_emb)
            else:
                token_emb = np.zeros((1, 1, 288), dtype=bf16_dt)

            # 2. Position step tensor: shape (1, 1), int32
            pos_tensor = np.array([[step]], dtype=np.int32)

            # 3. Assemble 26 inputs:
            # [token_emb, pos_tensor] + 6 layers * (past_k, past_v, cross_k, cross_v)
            decoder_args = [token_emb, pos_tensor]
            for l in range(6):
                decoder_args.append(past_kvs[2 * l])
                decoder_args.append(past_kvs[2 * l + 1])
                decoder_args.append(encoder_outputs[2 * l])
                decoder_args.append(encoder_outputs[2 * l + 1])

            # 4. Decoder forward pass
            dec_outputs = self._decoder_runner.infer(decoder_args)

            # Output 0: next-token logits of shape (1, 1, 32768)
            # Outputs 1..12: 12 updated past self-attention KV caches of shape (1, 8, 30, 36)
            logits = dec_outputs[0]
            past_kvs = list(dec_outputs[1:13])

            # Greedy next-token argmax
            logits_f32 = np.array(logits, dtype=np.float32)
            next_token = int(np.argmax(logits_f32[0, 0, :]))

            if next_token == end_tok:
                break

            generated_tokens.append(next_token)
            current_token = next_token

        return self._tokenizer.decode(generated_tokens).strip()
