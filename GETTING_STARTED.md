# Getting Started with Music Generator

## Quick Start Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/NihaalManaf/Music-Generator-.git
cd Music-Generator-

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Music Files

Add some music files to the `data/input/` directory:
- Supported formats: WAV, MP3, FLAC, MIDI
- Recommended: 2-5 songs, 5-30 seconds each
- Shorter clips work better for initial experiments

### 3. Run Your First Example

```bash
# Basic generation
python examples/basic_generation.py

# Blend multiple songs
python examples/blend_songs.py

# Analyze components
python examples/analyze_components.py
```

### 4. Interactive Exploration

Launch Jupyter notebook for interactive experimentation:

```bash
jupyter notebook notebooks/exploration.ipynb
```

## Project Structure Overview

```
Music-Generator-/
├── src/                         # Core library code
│   ├── music_generator.py       # Main interface
│   ├── decomposer.py            # PCA/SVD algorithms
│   ├── audio_processor.py       # Audio handling
│   ├── midi_processor.py        # MIDI handling
│   └── visualizer.py            # Visualization tools
│
├── examples/                    # Example scripts
│   ├── basic_generation.py      # Simple generation
│   ├── blend_songs.py           # Blend multiple songs
│   └── analyze_components.py    # Component analysis
│
├── notebooks/                   # Jupyter notebooks
│   └── exploration.ipynb        # Interactive exploration
│
├── tests/                       # Unit tests
│   └── test_decomposer.py       # Decomposer tests
│
└── data/                        # Data directory (gitignored)
    ├── input/                   # Your input music files
    ├── output/                  # Generated music
    └── models/                  # Saved components
```

## Core Concepts

### How It Works

1. **Load Music**: Import audio/MIDI files
2. **Convert to Numbers**: Transform to spectrograms or piano rolls
3. **Decompose**: Apply PCA or SVD to extract principal components
4. **Combine**: Blend components from different songs with custom weights
5. **Reconstruct**: Convert back to audio and save

### Code Example

```python
from src.music_generator import MusicGenerator

# Initialize
generator = MusicGenerator(method='pca')

# Load songs
generator.load_songs(['song1.wav', 'song2.wav', 'song3.wav'])

# Extract components
generator.decompose(n_components=10)

# Generate new music (blend with custom weights)
new_music = generator.generate(weights=[0.5, 0.3, 0.2])

# Save result
generator.save('output/my_creation.wav')
```

## Next Steps

### Experiment Ideas

1. **Try Different Components**: Test with 5, 10, 20 components
2. **Custom Blending**: Adjust weights to favor certain songs
3. **Compare Methods**: Try both PCA and SVD
4. **Visualize**: Analyze which components capture which features
5. **MIDI vs Audio**: Compare symbolic vs audio representations

### Advanced Techniques

- Extract specific musical features (rhythm, melody, harmony)
- Use different audio representations (MFCC, chromagram)
- Implement time-alignment for better blending
- Add temporal dynamics
- Create style transfer between genres

## Troubleshooting

### Common Issues

**Import errors**: Make sure you've installed all requirements
```bash
pip install -r requirements.txt
```

**Audio files not found**: Check paths in example scripts
- Files should be in `data/input/`
- Update paths in example scripts if needed

**Out of memory**: Try:
- Shorter audio clips (5-10 seconds)
- Fewer components (5-10)
- Lower sample rate

**Poor quality output**: 
- Use more components (15-20)
- Use higher quality input files
- Try SVD instead of PCA

## Resources

### Learn More About PCA/SVD
- [Principal Component Analysis - Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis)
- [Singular Value Decomposition - Wikipedia](https://en.wikipedia.org/wiki/Singular_value_decomposition)
- [Eigenfaces - Original Paper](http://www.face-rec.org/algorithms/PCA/jcn.pdf)

### Music Information Retrieval
- [Librosa Documentation](https://librosa.org/)
- [ISMIR (Music Information Retrieval)](https://www.ismir.net/)

## Contributing

Feel free to:
- Report bugs or issues
- Suggest new features
- Submit pull requests
- Share your generated music!

## License

MIT License - See LICENSE file

---

**Have fun creating music with math! 🎵**
