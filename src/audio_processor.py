"""
Audio Processing Module

Handles loading, converting, and saving audio files.
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Tuple
from pathlib import Path


class AudioProcessor:
    """
    Processes audio files for music generation.
    
    Handles:
    - Loading audio files
    - Converting audio to spectrograms
    - Converting spectrograms back to audio
    - Saving audio files
    """
    
    def __init__(self, sample_rate: int = 22050, n_fft: int = 2048,
                 hop_length: int = 512, n_mels: int = 128):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate: Target sample rate in Hz
            n_fft: FFT window size
            hop_length: Number of samples between frames
            n_mels: Number of mel bands
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
    def load_audio(self, filepath: str, duration: Optional[float] = None) -> np.ndarray:
        """
        Load an audio file.
        
        Args:
            filepath: Path to audio file
            duration: Maximum duration to load (seconds), None for full file
            
        Returns:
            Audio time series as numpy array
        """
        audio, sr = librosa.load(filepath, sr=self.sample_rate, duration=duration)
        return audio
        
    def audio_to_spectrogram(self, audio: np.ndarray, 
                            mel: bool = True) -> np.ndarray:
        """
        Convert audio to spectrogram representation.
        
        Args:
            audio: Audio time series
            mel: If True, compute mel spectrogram; else linear spectrogram
            
        Returns:
            Spectrogram (frequency x time)
        """
        if mel:
            # Compute mel spectrogram
            spectrogram = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            # Convert to log scale (dB)
            spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        else:
            # Compute STFT
            stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
            spectrogram_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
            
        return spectrogram_db
        
    def spectrogram_to_audio(self, spectrogram: np.ndarray,
                            mel: bool = True) -> np.ndarray:
        """
        Convert spectrogram back to audio (using Griffin-Lim algorithm).
        
        Args:
            spectrogram: Spectrogram in dB scale
            mel: If True, interpret as mel spectrogram
            
        Returns:
            Reconstructed audio time series
        """
        # Convert from dB back to power/amplitude
        if mel:
            spectrogram_power = librosa.db_to_power(spectrogram)
            # Use inverse mel filterbank (approximation)
            audio = librosa.feature.inverse.mel_to_audio(
                spectrogram_power,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
        else:
            spectrogram_amp = librosa.db_to_amplitude(spectrogram)
            # Use Griffin-Lim to estimate phase and reconstruct
            audio = librosa.griffinlim(
                spectrogram_amp,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
        return audio
        
    def load_and_convert(self, filepath: str, 
                        duration: Optional[float] = None) -> np.ndarray:
        """
        Load audio file and convert to spectrogram.
        
        Args:
            filepath: Path to audio file
            duration: Maximum duration to load (seconds)
            
        Returns:
            Spectrogram representation
        """
        audio = self.load_audio(filepath, duration=duration)
        spectrogram = self.audio_to_spectrogram(audio)
        return spectrogram
        
    def save_audio(self, filepath: str, audio: np.ndarray) -> None:
        """
        Save audio to file.
        
        Args:
            filepath: Output file path
            audio: Audio time series
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        sf.write(str(filepath), audio, self.sample_rate)
        print(f"Audio saved to {filepath}")
        
    def get_chromagram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute chromagram (pitch class representation).
        
        Args:
            audio: Audio time series
            
        Returns:
            Chromagram (12 x time)
        """
        chromagram = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return chromagram
        
    def get_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """
        Compute MFCCs (Mel-frequency cepstral coefficients).
        
        Args:
            audio: Audio time series
            n_mfcc: Number of MFCCs to compute
            
        Returns:
            MFCC features (n_mfcc x time)
        """
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return mfccs
        
    def visualize_spectrogram(self, spectrogram: np.ndarray, 
                            title: str = "Spectrogram") -> None:
        """
        Display a spectrogram.
        
        Args:
            spectrogram: Spectrogram to visualize
            title: Plot title
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(
            spectrogram,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            x_axis='time',
            y_axis='mel'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.show()
