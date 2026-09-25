"""Audio preprocessing utilities for Speech-to-Text on Coralboard SL2619."""

import io
import wave
from typing import Optional, Tuple
import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30  # 30 seconds
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000 frames
N_MELS = 80


def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert raw 16-bit PCM bytes (mono) to normalized float32 array in [-1.0, 1.0]."""
    if len(pcm_bytes) == 0:
        return np.zeros(0, dtype=np.float32)
    # 16-bit signed integer
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    audio /= 32768.0
    return audio


def wav_bytes_to_float32(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Parse a WAV file in bytes, return (float32_array, sample_rate)."""
    with io.BytesIO(wav_bytes) as bio:
        with wave.open(bio, "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

    if sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if channels > 1:
        # Convert stereo/multichannel to mono by averaging channels
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio, framerate


def resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Fast linear resampling if sample rates differ."""
    if orig_sr == target_sr or len(audio) == 0:
        return audio
    num_target_samples = int(round(len(audio) * target_sr / orig_sr))
    return np.interp(
        np.linspace(0.0, 1.0, num_target_samples, endpoint=False),
        np.linspace(0.0, 1.0, len(audio), endpoint=False),
        audio,
    ).astype(np.float32)


def pad_or_trim(array: np.ndarray, length: int = N_SAMPLES) -> np.ndarray:
    """Pad with zeros or trim audio array to exactly `length` samples."""
    if len(array) > length:
        return array[:length]
    if len(array) < length:
        return np.pad(array, (0, length - len(array)), mode="constant")
    return array


def get_mel_filters(sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS) -> np.ndarray:
    """Generate triangular mel filterbank matrix (n_mels, 1 + n_fft // 2)."""
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    f_min = 0.0
    f_max = float(sr // 2)
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    n_freqs = 1 + n_fft // 2
    weights = np.zeros((n_mels, n_freqs), dtype=np.float32)

    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        for f in range(left, center):
            if center != left:
                weights[i - 1, f] = (f - left) / (center - left)
        for f in range(center, right):
            if right != center:
                weights[i - 1, f] = (right - f) / (right - center)

        enorm = 2.0 / (hz_points[i + 1] - hz_points[i - 1])
        weights[i - 1] *= enorm

    return weights


_MEL_FILTERS = None


def log_mel_spectrogram(
    audio: np.ndarray,
    n_mels: int = N_MELS,
    padding: int = 0,
) -> np.ndarray:
    """
    Compute log-mel spectrogram.
    Returns:
        np.ndarray of shape (1, n_mels, n_frames) with float32 values.
    """
    global _MEL_FILTERS
    if _MEL_FILTERS is None:
        _MEL_FILTERS = get_mel_filters(SAMPLE_RATE, N_FFT, n_mels)

    if padding > 0:
        audio = np.pad(audio, (0, padding))

    # STFT with Hanning window
    window = np.hanning(N_FFT).astype(np.float32)

    n_samples = len(audio)
    num_frames = 1 + (n_samples - N_FFT) // HOP_LENGTH if n_samples >= N_FFT else 0
    if num_frames <= 0:
        return np.zeros((1, n_mels, N_FRAMES), dtype=np.float32)

    strides = (audio.strides[0] * HOP_LENGTH, audio.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        audio, shape=(num_frames, N_FFT), strides=strides
    )

    windowed_frames = frames * window
    fft_spec = np.fft.rfft(windowed_frames, n=N_FFT, axis=-1)
    magnitudes = np.abs(fft_spec[:, :-1]) ** 2

    mel_spec = np.dot(magnitudes, _MEL_FILTERS[:, :-1].T)

    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    transposed = log_spec.T[np.newaxis, :, :]

    if transposed.shape[2] < N_FRAMES:
        pad_width = ((0, 0), (0, 0), (0, N_FRAMES - transposed.shape[2]))
        transposed = np.pad(transposed, pad_width, mode="constant")
    elif transposed.shape[2] > N_FRAMES:
        transposed = transposed[:, :, :N_FRAMES]

    return transposed.astype(np.float32)
