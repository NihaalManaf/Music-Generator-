"""
Music Generator Package

A music generation toolkit using PCA and SVD for decomposing and 
recombining musical elements to create novel compositions.
"""

__version__ = "0.1.0"
__author__ = "NihaalManaf"

from .music_generator import MusicGenerator
from .decomposer import MusicDecomposer
from .audio_processor import AudioProcessor
from .midi_processor import MIDIProcessor

__all__ = [
    'MusicGenerator',
    'MusicDecomposer', 
    'AudioProcessor',
    'MIDIProcessor',
]
