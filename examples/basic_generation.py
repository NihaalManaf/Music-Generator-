"""
Basic Music Generation Example

This example demonstrates the basic workflow of the Music Generator:
1. Load music files
2. Decompose using PCA/SVD
3. Generate new music
4. Save the result
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from music_generator import MusicGenerator


def main():
    """
    Basic example of music generation.
    
    Note: This is a skeleton example. You'll need to provide actual music files
    in the data/input/ directory to run this.
    """
    print("=" * 60)
    print("BASIC MUSIC GENERATION EXAMPLE")
    print("=" * 60)
    
    # Initialize the generator
    print("\n1. Initializing Music Generator...")
    generator = MusicGenerator(method='pca', sample_rate=22050)
    
    # Example song paths (update these with actual files)
    song_files = [
        '../data/input/song1.wav',
        '../data/input/song2.wav',
        '../data/input/song3.wav',
    ]
    
    # Check if files exist
    existing_files = [f for f in song_files if Path(f).exists()]
    
    if not existing_files:
        print("\n⚠️  No input files found!")
        print("Please add some .wav or .mp3 files to the data/input/ directory")
        print("\nExample files you can use:")
        print("  - Short music clips (5-30 seconds work well)")
        print("  - WAV, MP3, or MIDI files")
        print("  - At least 2 files for blending")
        return
    
    try:
        # Load songs
        print(f"\n2. Loading {len(existing_files)} songs...")
        generator.load_songs(existing_files)
        
        # Decompose into components
        print("\n3. Decomposing music into principal components...")
        n_components = 10
        generator.decompose(n_components=n_components)
        
        # Analyze components
        print("\n4. Analyzing components...")
        generator.analyze_components(song_index=0)
        
        # Generate new music with equal blending
        print("\n5. Generating new music...")
        weights = [1.0 / len(existing_files)] * len(existing_files)
        print(f"   Blending weights: {weights}")
        new_music = generator.generate(weights=weights)
        
        # Save result
        output_path = '../data/output/generated_basic.wav'
        print(f"\n6. Saving generated music to {output_path}...")
        generator.save(output_path, format='wav')
        
        print("\n" + "=" * 60)
        print("✅ Generation complete!")
        print(f"Output saved to: {output_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Installed all dependencies (pip install -r requirements.txt)")
        print("  2. Added music files to data/input/")
        print("  3. Audio files are in supported formats (WAV, MP3)")


if __name__ == '__main__':
    main()
