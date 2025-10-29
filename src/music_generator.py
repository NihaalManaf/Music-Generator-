"""
Main Music Generator Class

This module provides the high-level interface for music generation using PCA/SVD.
"""

import numpy as np
from typing import List, Optional, Union
from pathlib import Path

from .decomposer import MusicDecomposer
from .audio_processor import AudioProcessor
from .midi_processor import MIDIProcessor


class MusicGenerator:
    """
    Main class for generating music using PCA/SVD decomposition.
    
    This class orchestrates the entire music generation pipeline:
    1. Load multiple music files
    2. Convert to numerical representations (spectrograms/piano rolls)
    3. Decompose using PCA or SVD
    4. Combine components from different songs
    5. Reconstruct and save generated music
    
    Example:
        >>> generator = MusicGenerator()
        >>> generator.load_songs(['song1.wav', 'song2.wav'])
        >>> generator.decompose(n_components=10)
        >>> new_music = generator.generate(weights=[0.5, 0.5])
        >>> generator.save('output.wav')
    """
    
    def __init__(self, method: str = 'pca', sample_rate: int = 22050):
        """
        Initialize the Music Generator.
        
        Args:
            method: Decomposition method - 'pca' or 'svd'
            sample_rate: Audio sample rate in Hz
        """
        self.method = method
        self.sample_rate = sample_rate
        
        # Initialize processors
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        self.midi_processor = MIDIProcessor()
        self.decomposer = MusicDecomposer(method=method)
        
        # Storage for loaded songs and components
        self.songs = []
        self.song_representations = []
        self.components = []
        self.generated_music = None
        
    def load_songs(self, filepaths: List[Union[str, Path]]) -> None:
        """
        Load multiple music files for analysis.
        
        Args:
            filepaths: List of paths to audio/MIDI files
            
        Raises:
            ValueError: If no valid files are loaded
        """
        self.songs = []
        self.song_representations = []
        
        for filepath in filepaths:
            filepath = Path(filepath)
            
            # Determine file type and load accordingly
            if filepath.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg']:
                # Load as audio
                spectrogram = self.audio_processor.load_and_convert(str(filepath))
                self.song_representations.append(spectrogram)
                self.songs.append(str(filepath))
                
            elif filepath.suffix.lower() in ['.mid', '.midi']:
                # Load as MIDI
                piano_roll = self.midi_processor.load_and_convert(str(filepath))
                self.song_representations.append(piano_roll)
                self.songs.append(str(filepath))
                
            else:
                print(f"Warning: Unsupported file format {filepath.suffix}")
                
        if not self.songs:
            raise ValueError("No valid music files loaded")
            
        print(f"Loaded {len(self.songs)} songs successfully")
        
    def decompose(self, n_components: int = 10) -> None:
        """
        Decompose loaded songs into principal components.
        
        Args:
            n_components: Number of components to extract
        """
        if not self.song_representations:
            raise ValueError("No songs loaded. Call load_songs() first.")
            
        self.components = []
        
        for i, representation in enumerate(self.song_representations):
            print(f"Decomposing song {i+1}/{len(self.song_representations)}...")
            
            if self.method == 'pca':
                components = self.decomposer.decompose_pca(
                    representation, 
                    n_components=n_components
                )
            else:  # SVD
                U, sigma, Vt = self.decomposer.decompose_svd(
                    representation,
                    n_components=n_components
                )
                components = {'U': U, 'sigma': sigma, 'Vt': Vt}
                
            self.components.append(components)
            
        print(f"Decomposition complete: {n_components} components extracted")
        
    def generate(self, weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Generate new music by combining components from loaded songs.
        
        Args:
            weights: Weights for blending songs (should sum to 1.0)
                    If None, uses equal weights
                    
        Returns:
            Generated music representation (spectrogram or piano roll)
        """
        if not self.components:
            raise ValueError("No components available. Call decompose() first.")
            
        # Use equal weights if none provided
        if weights is None:
            weights = [1.0 / len(self.components)] * len(self.components)
            
        if len(weights) != len(self.components):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of songs ({len(self.components)})"
            )
            
        # Normalize weights to sum to 1.0
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Combine components based on method
        print("Generating new music from components...")
        self.generated_music = self.decomposer.reconstruct_blended(
            self.components,
            weights
        )
        
        return self.generated_music
        
    def save(self, output_path: Union[str, Path], 
             format: str = 'wav') -> None:
        """
        Save generated music to file.
        
        Args:
            output_path: Path where to save the file
            format: Output format - 'wav', 'mp3', or 'midi'
        """
        if self.generated_music is None:
            raise ValueError("No music generated. Call generate() first.")
            
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format in ['wav', 'mp3']:
            # Convert spectrogram back to audio
            audio = self.audio_processor.spectrogram_to_audio(
                self.generated_music
            )
            self.audio_processor.save_audio(str(output_path), audio)
            
        elif format == 'midi':
            # Convert piano roll to MIDI
            self.midi_processor.piano_roll_to_midi(
                self.generated_music,
                str(output_path)
            )
            
        print(f"Saved generated music to {output_path}")
        
    def analyze_components(self, song_index: int = 0) -> None:
        """
        Visualize the principal components of a song.
        
        Args:
            song_index: Index of the song to analyze
        """
        if not self.components:
            raise ValueError("No components available. Call decompose() first.")
            
        if song_index >= len(self.components):
            raise ValueError(f"Song index {song_index} out of range")
            
        self.decomposer.plot_components(self.components[song_index])
        self.decomposer.plot_variance_explained()
