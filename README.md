# Music Generator using PCA and SVD

## Overview

This project applies Principal Component Analysis (PCA) and Singular Value Decomposition (SVD) to music generation, inspired by the concept of combining facial features to create new faces. Instead of faces, we decompose and recombine musical elements to generate novel compositions!

## Concept

### The Inspiration
Just as PCA/SVD can decompose face images into principal components (eigenfaces) and recombine them to create new faces, we can apply the same mathematical techniques to music:

- **Face Generation**: Decompose multiple face images → Extract principal components → Combine components → Generate new face
- **Music Generation**: Decompose multiple music pieces → Extract principal components → Combine components → Generate new music

### How It Works

#### 1. **Music Decomposition**
Music can be represented as numerical data in various forms:
- **Spectrograms**: Time-frequency representations of audio signals
- **Piano Roll**: MIDI note sequences over time
- **Audio Waveforms**: Raw amplitude values

#### 2. **PCA/SVD Analysis**
- **SVD (Singular Value Decomposition)**: Factorizes the music matrix into U, Σ, and V components
  - `M = U × Σ × V^T`
  - U: Left singular vectors (temporal patterns)
  - Σ: Singular values (importance of each component)
  - V^T: Right singular vectors (frequency/note patterns)

- **PCA (Principal Component Analysis)**: Identifies the directions of maximum variance
  - Reduces dimensionality while preserving essential musical characteristics
  - Components capture melodic, harmonic, and rhythmic patterns

#### 3. **Music Generation**
- Combine principal components from different songs
- Adjust weights to create variations
- Reconstruct music from the combined components
- Generate novel compositions with blended characteristics

## Mathematical Foundation

### SVD Decomposition
```
Music Matrix M (m×n) = U (m×r) × Σ (r×r) × V^T (r×n)
```
Where:
- m = time steps
- n = frequency bins or notes
- r = rank (number of components)

### PCA Transformation
```
X_transformed = X × Principal_Components
X_reconstructed = X_transformed × Principal_Components^T
```

## Project Structure

```
Music-Generator-/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── src/                         # Source code
│   ├── __init__.py
│   ├── music_generator.py       # Main generator class
│   ├── decomposer.py            # PCA/SVD decomposition logic
│   ├── audio_processor.py       # Audio loading and processing
│   ├── midi_processor.py        # MIDI handling
│   └── visualizer.py            # Visualization utilities
├── examples/                    # Example scripts
│   ├── basic_generation.py      # Basic usage example
│   ├── blend_songs.py           # Blend multiple songs
│   └── analyze_components.py    # Analyze principal components
├── notebooks/                   # Jupyter notebooks
│   └── exploration.ipynb        # Interactive exploration
├── tests/                       # Unit tests
│   └── test_decomposer.py
└── data/                        # Data directory (gitignored)
    ├── input/                   # Input music files
    ├── output/                  # Generated music
    └── models/                  # Saved models/components
```

## Installation

```bash
# Clone the repository
git clone https://github.com/NihaalManaf/Music-Generator-.git
cd Music-Generator-

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from src.music_generator import MusicGenerator

# Initialize the generator
generator = MusicGenerator()

# Load and decompose music files
generator.load_songs(['song1.wav', 'song2.wav', 'song3.wav'])

# Extract principal components
generator.decompose(n_components=10)

# Generate new music by blending components
new_music = generator.generate(weights=[0.4, 0.3, 0.3])

# Save the result
generator.save('output/generated_music.wav')
```

### Advanced Usage

```python
from src.decomposer import MusicDecomposer
from src.audio_processor import AudioProcessor

# Load audio
processor = AudioProcessor()
spectrogram = processor.load_and_convert('song.wav')

# Decompose using SVD
decomposer = MusicDecomposer(method='svd')
U, sigma, Vt = decomposer.decompose(spectrogram, n_components=20)

# Analyze components
decomposer.plot_components()
decomposer.plot_variance_explained()

# Reconstruct with selected components
reconstructed = decomposer.reconstruct(components=[0, 1, 2, 5, 7])

# Convert back to audio
audio = processor.spectrogram_to_audio(reconstructed)
processor.save_audio('output.wav', audio)
```

## Features

- [x] **Multiple Input Formats**: Support for WAV, MP3, MIDI files
- [x] **Flexible Decomposition**: PCA and SVD methods
- [x] **Component Analysis**: Visualize and analyze musical components
- [x] **Customizable Generation**: Blend songs with custom weights
- [x] **Audio Visualization**: Spectrograms, waveforms, component plots
- [x] **MIDI Support**: Work with symbolic music representation

## Use Cases

1. **Music Blending**: Combine characteristics of multiple songs
2. **Style Transfer**: Extract style from one song, apply to another
3. **Music Completion**: Reconstruct incomplete or damaged audio
4. **Compression**: Reduce music data while preserving quality
5. **Feature Extraction**: Identify key musical patterns and motifs

## Technical Details

### Representation Methods

1. **Mel Spectrogram**: Time-frequency representation on mel scale (matches human perception)
2. **Chromagram**: Represents pitch classes over time
3. **Piano Roll**: Binary matrix for MIDI (time × pitch)

### Dimensionality Reduction

- **Input**: High-dimensional music matrix (e.g., 1000 time steps × 128 frequency bins)
- **Components**: Reduced to k components (e.g., 10-50) that capture 90%+ variance
- **Output**: Reconstruction from components maintains musical coherence

## Dependencies

Core libraries used:
- `numpy`: Numerical computations and matrix operations
- `scipy`: Scientific computing and optimization
- `librosa`: Music and audio analysis
- `scikit-learn`: PCA and machine learning utilities
- `matplotlib`: Visualization
- `soundfile`: Audio file I/O
- `mido`: MIDI file processing

## Examples

See the `examples/` directory for:
- `basic_generation.py`: Generate music from scratch
- `blend_songs.py`: Blend two or more songs
- `analyze_components.py`: Visualize principal components

## Future Enhancements

- [ ] Deep learning integration (Autoencoders, VAEs)
- [ ] Real-time generation
- [ ] Web interface for interactive music generation
- [ ] Support for more audio formats
- [ ] Preset blending profiles
- [ ] Advanced time-alignment for better blending

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - feel free to use this project for learning and experimentation!

## References

- Eigenfaces for Recognition (Turk & Pentland, 1991)
- Music Information Retrieval (MIR) techniques
- Librosa documentation: https://librosa.org/
- PCA/SVD in signal processing

## Acknowledgments

Inspired by the fascinating application of PCA/SVD to facial recognition, extended to the creative domain of music generation!
