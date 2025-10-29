"""
Blend Songs Example

This example shows how to blend multiple songs with custom weights
to create new musical compositions.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from music_generator import MusicGenerator
from visualizer import plot_blending_weights


def main():
    """
    Example of blending multiple songs with custom weights.
    """
    print("=" * 60)
    print("SONG BLENDING EXAMPLE")
    print("=" * 60)
    
    # Initialize generator
    print("\n1. Initializing Music Generator with SVD method...")
    generator = MusicGenerator(method='svd', sample_rate=22050)
    
    # Define songs to blend
    song_files = [
        '../data/input/classical.wav',
        '../data/input/jazz.wav',
        '../data/input/electronic.wav',
    ]
    
    # Check which files exist
    existing_files = []
    song_names = []
    for f in song_files:
        if Path(f).exists():
            existing_files.append(f)
            song_names.append(Path(f).stem)
    
    if len(existing_files) < 2:
        print("\n⚠️  Need at least 2 music files to blend!")
        print("Please add music files to data/input/")
        print("\nSuggested file names:")
        for f in song_files:
            print(f"  - {f}")
        return
    
    try:
        # Load songs
        print(f"\n2. Loading {len(existing_files)} songs for blending...")
        for name in song_names:
            print(f"   - {name}")
        generator.load_songs(existing_files)
        
        # Decompose
        print("\n3. Decomposing songs (SVD)...")
        n_components = 15
        generator.decompose(n_components=n_components)
        
        # Define different blending scenarios
        blending_scenarios = [
            {
                'name': 'Equal Blend',
                'weights': [1.0 / len(existing_files)] * len(existing_files),
                'output': 'blend_equal.wav'
            },
            {
                'name': 'Dominant First',
                'weights': [0.7] + [0.3 / (len(existing_files) - 1)] * (len(existing_files) - 1),
                'output': 'blend_dominant_first.wav'
            },
            {
                'name': 'Dominant Last',
                'weights': [0.15] * (len(existing_files) - 1) + [0.7],
                'output': 'blend_dominant_last.wav'
            },
        ]
        
        # Generate music for each scenario
        print("\n4. Generating blended music...")
        for i, scenario in enumerate(blending_scenarios, 1):
            print(f"\n   Scenario {i}: {scenario['name']}")
            print(f"   Weights: {scenario['weights']}")
            
            # Visualize weights
            plot_blending_weights(scenario['weights'], song_names)
            
            # Generate
            new_music = generator.generate(weights=scenario['weights'])
            
            # Save
            output_path = f"../data/output/{scenario['output']}"
            generator.save(output_path, format='wav')
            print(f"   ✅ Saved to: {output_path}")
        
        print("\n" + "=" * 60)
        print("✅ All blends generated successfully!")
        print("Check the data/output/ directory for results")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
