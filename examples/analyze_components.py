"""
Analyze Components Example

This example demonstrates how to analyze and visualize the principal
components extracted from music.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from decomposer import MusicDecomposer
from audio_processor import AudioProcessor
from visualizer import plot_component_contribution, create_analysis_report


def analyze_single_song(filepath: str, method: str = 'pca'):
    """
    Analyze a single song in detail.
    
    Args:
        filepath: Path to audio file
        method: 'pca' or 'svd'
    """
    print(f"\nAnalyzing: {Path(filepath).name}")
    print("-" * 60)
    
    # Load and convert to spectrogram
    processor = AudioProcessor(sample_rate=22050)
    spectrogram = processor.load_and_convert(filepath)
    print(f"Spectrogram shape: {spectrogram.shape}")
    
    # Decompose
    decomposer = MusicDecomposer(method=method)
    n_components = 20
    
    if method == 'pca':
        components = decomposer.decompose_pca(spectrogram, n_components=n_components)
        print(f"PCA components: {components['components'].shape}")
        print(f"Transformed data: {components['transformed'].shape}")
        
    else:  # SVD
        U, sigma, Vt = decomposer.decompose_svd(spectrogram, n_components=n_components)
        print(f"U shape: {U.shape}")
        print(f"Sigma shape: {sigma.shape}")
        print(f"Vt shape: {Vt.shape}")
        components = (U, sigma, Vt)
    
    # Show variance explained
    print(f"\nVariance explained by components:")
    for i, var in enumerate(decomposer.variance_explained[:10], 1):
        print(f"  Component {i:2d}: {var:6.2%}")
    
    cumulative = sum(decomposer.variance_explained)
    print(f"\nTotal variance explained: {cumulative:.2%}")
    
    # Visualize
    print("\nGenerating visualizations...")
    decomposer.plot_components(components, n_display=5)
    decomposer.plot_variance_explained()
    plot_component_contribution(decomposer.variance_explained)
    
    # Test reconstruction
    print("\nTesting reconstruction...")
    if method == 'pca':
        reconstructed = decomposer.reconstruct_pca(components, n_components=10)
    else:
        reconstructed = decomposer.reconstruct_svd(U, sigma, Vt, n_components=10)
    
    print(f"Reconstructed shape: {reconstructed.shape}")
    print("✅ Reconstruction successful!")
    
    return decomposer.variance_explained


def main():
    """
    Main analysis routine.
    """
    print("=" * 60)
    print("COMPONENT ANALYSIS EXAMPLE")
    print("=" * 60)
    
    # Find available songs
    input_dir = Path('../data/input')
    audio_files = []
    
    for ext in ['*.wav', '*.mp3', '*.flac']:
        audio_files.extend(input_dir.glob(ext))
    
    if not audio_files:
        print("\n⚠️  No audio files found in data/input/")
        print("Please add some audio files to analyze")
        print("\nSupported formats: WAV, MP3, FLAC")
        return
    
    print(f"\nFound {len(audio_files)} audio file(s)")
    
    # Analyze each file
    all_variances = []
    song_names = []
    
    for audio_file in audio_files[:3]:  # Limit to first 3
        try:
            variance = analyze_single_song(str(audio_file), method='pca')
            all_variances.append(variance)
            song_names.append(audio_file.name)
        except Exception as e:
            print(f"\n❌ Error analyzing {audio_file.name}: {e}")
            continue
    
    # Create summary report
    if all_variances:
        print("\n" + "=" * 60)
        print("SUMMARY REPORT")
        print("=" * 60)
        
        for name, variance in zip(song_names, all_variances):
            print(f"\n{name}:")
            print(f"  Top component variance: {variance[0]:.2%}")
            print(f"  Top 5 components: {sum(variance[:5]):.2%}")
            print(f"  Top 10 components: {sum(variance[:10]):.2%}")
        
        print("\n" + "=" * 60)
        print("✅ Analysis complete!")
        print("=" * 60)


if __name__ == '__main__':
    main()
