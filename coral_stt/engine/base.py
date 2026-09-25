"""Base abstract class for Speech-to-Text transcription engines."""

from abc import ABC, abstractmethod
from typing import Optional


class STTEngine(ABC):
    """Abstract base class for STT transcription engines."""

    @abstractmethod
    def load(self) -> None:
        """Load and initialize model weights/runners."""
        pass

    @abstractmethod
    def transcribe(
        self,
        audio_pcm: bytes,
        sample_rate: int = 16000,
        language: Optional[str] = "en",
    ) -> str:
        """
        Transcribe raw PCM audio to text.
        
        Args:
            audio_pcm: Raw 16-bit mono PCM bytes.
            sample_rate: Audio sampling rate (usually 16000).
            language: Target ISO language code or None for auto-detect.
            
        Returns:
            Transcribed text.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the model in use."""
        pass
