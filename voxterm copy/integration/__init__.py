"""
VoxTerm Integration Module

Provides adapters for integrating various voice engines with VoxTerm.
"""

from .base import BaseIntegration
from .voice_engine import VoiceEngineAdapter

__all__ = ['BaseIntegration', 'VoiceEngineAdapter']