"""
Visualization Utilities

Helper functions for visualizing music data and analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


def plot_waveform(audio: np.ndarray, sample_rate: int = 22050, 
                 title: str = "Waveform") -> None:
    """
    Plot audio waveform.
    
    Args:
        audio: Audio time series
        sample_rate: Sample rate in Hz
        title: Plot title
    """
    plt.figure(figsize=(14, 4))
    time_axis = np.arange(len(audio)) / sample_rate
    plt.plot(time_axis, audio)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_comparison(original: np.ndarray, reconstructed: np.ndarray,
                   sample_rate: int = 22050) -> None:
    """
    Plot original vs reconstructed audio/spectrogram.
    
    Args:
        original: Original data
        reconstructed: Reconstructed data
        sample_rate: Sample rate
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Original
    axes[0].imshow(original, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Original')
    axes[0].set_ylabel('Frequency/Pitch')
    
    # Reconstructed
    axes[1].imshow(reconstructed, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Reconstructed')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Frequency/Pitch')
    
    plt.tight_layout()
    plt.show()


def plot_component_contribution(variances: np.ndarray, n_display: int = 20) -> None:
    """
    Plot component contribution to variance.
    
    Args:
        variances: Variance explained by each component
        n_display: Number of components to display
    """
    variances = variances[:n_display]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    axes[0].bar(range(len(variances)), variances)
    axes[0].set_xlabel('Component Index')
    axes[0].set_ylabel('Variance Explained')
    axes[0].set_title('Component Contributions')
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative plot
    cumsum = np.cumsum(variances)
    axes[1].plot(cumsum, 'o-', linewidth=2)
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Variance')
    axes[1].set_title('Cumulative Variance Explained')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()


def plot_blending_weights(weights: List[float], song_names: Optional[List[str]] = None) -> None:
    """
    Visualize blending weights for multiple songs.
    
    Args:
        weights: Weight for each song
        song_names: Optional names for songs
    """
    if song_names is None:
        song_names = [f'Song {i+1}' for i in range(len(weights))]
        
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(weights)))
    bars = plt.bar(song_names, weights, color=colors)
    plt.xlabel('Song')
    plt.ylabel('Weight')
    plt.title('Blending Weights')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def create_analysis_report(songs: List[str], components: List, 
                          variance_explained: np.ndarray) -> None:
    """
    Create a comprehensive analysis report.
    
    Args:
        songs: List of song names/paths
        components: List of component data
        variance_explained: Variance explained by components
    """
    print("=" * 60)
    print("MUSIC DECOMPOSITION ANALYSIS REPORT")
    print("=" * 60)
    print(f"\nNumber of songs analyzed: {len(songs)}")
    print("\nSongs:")
    for i, song in enumerate(songs, 1):
        print(f"  {i}. {song}")
    
    print(f"\nNumber of components extracted: {len(variance_explained)}")
    print(f"Total variance explained: {np.sum(variance_explained):.2%}")
    print(f"Variance by top 5 components: {np.sum(variance_explained[:5]):.2%}")
    print(f"Variance by top 10 components: {np.sum(variance_explained[:10]):.2%}")
    
    print("\nTop 5 components:")
    for i in range(min(5, len(variance_explained))):
        print(f"  Component {i+1}: {variance_explained[i]:.2%}")
    
    print("=" * 60)
