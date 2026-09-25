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


def _to_bf16(arr: np.ndarray) -> np.ndarray:
    """Convert numpy array to bfloat16 using ml_dtypes if available."""
    try:
        import ml_dtypes
        return arr.astype(ml_dtypes.bfloat16)
    except Exception:
        return arr


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
                        self._token_embeddings = np.load(candidate)
                        _LOGGER.info("Loaded decoder token embeddings from %s: shape %s", candidate, self._token_embeddings.shape)
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

        # 3. Format audio tensor: provide normalized audio input
        audio_input = audio[np.newaxis, :].astype(np.float32)

        # 4. Infer using Torq NPU
        if self._is_end_to_end:
            text = self._infer_end_to_end(audio_input)
        else:
            text = self._infer_encoder_decoder(audio_input)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        _LOGGER.info(
            "Transcription completed in %.2f ms (Audio length: %.2f s): '%s'",
            elapsed_ms,
            len(audio) / SAMPLE_RATE,
            text,
        )
        return text

    def _infer_end_to_end(self, audio_tensor: np.ndarray) -> str:
        """Execute unified model that directly produces token IDs or strings."""
        audio_in = _to_bf16(audio_tensor)
        try:
            outputs = self._encoder_runner.infer([audio_in])
        except Exception:
            outputs = self._encoder_runner.infer([audio_tensor])

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
        """Execute NPU-accelerated Encoder followed by decoder autoregressive generation."""
        # Step A: Encoder forward pass on Torq NPU (expects bfloat16 input)
        audio_in = _to_bf16(audio_tensor)
        try:
            encoder_outputs = self._encoder_runner.infer([audio_in])
        except Exception as e:
            _LOGGER.warning("Encoder inference with bfloat16 failed (%s), retrying with raw input...", e)
            encoder_outputs = self._encoder_runner.infer([audio_tensor])

        audio_features = (
            encoder_outputs[0]
            if isinstance(encoder_outputs, (list, tuple))
            else encoder_outputs
        )

        # Step B: Autoregressive decoding
        start_tok = self._tokenizer.start_token
        end_tok = self._tokenizer.end_token
        tokens = [start_tok]

        for _ in range(self._max_tokens):
            tokens_tensor = np.array([tokens], dtype=np.int64)

            # Decoder inference: (tokens/embeddings, audio_features)
            # If token embeddings are loaded, project tokens to embedding vectors
            if self._token_embeddings is not None:
                token_emb = self._token_embeddings[tokens_tensor]
                token_emb_bf16 = _to_bf16(token_emb)
                try:
                    dec_outputs = self._decoder_runner.infer([token_emb_bf16, audio_features])
                except Exception:
                    try:
                        dec_outputs = self._decoder_runner.infer([token_emb.astype(np.float32), audio_features])
                    except Exception:
                        dec_outputs = self._decoder_runner.infer([tokens_tensor, audio_features])
            else:
                try:
                    dec_outputs = self._decoder_runner.infer([tokens_tensor, audio_features])
                except Exception:
                    dec_outputs = self._decoder_runner.infer([tokens_tensor.astype(np.int32), audio_features])

            logits = dec_outputs[0] if isinstance(dec_outputs, (list, tuple)) else dec_outputs

            # Convert logits to float32 for stable argmax
            logits_f32 = np.array(logits, dtype=np.float32)

            # Greedy next token: logits shape (1, seq_len, vocab_size)
            next_token = int(np.argmax(logits_f32[0, -1, :]))

            if next_token == end_tok:
                break

            tokens.append(next_token)

        # Decode tokens after the start prefix
        generated_tokens = tokens[1:]
        return self._tokenizer.decode(generated_tokens).strip()
