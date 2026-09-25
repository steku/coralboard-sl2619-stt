"""Torq NPU Speech-to-Text inference engine."""

from .base import STTEngine
from .torq_engine import TorqSTTEngine

__all__ = ["STTEngine", "TorqSTTEngine"]
