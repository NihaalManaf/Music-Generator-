"""
Tests for MusicDecomposer

Basic unit tests for the decomposition functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pytest
from decomposer import MusicDecomposer


def test_decomposer_initialization():
    """Test that decomposer initializes correctly."""
    decomposer_pca = MusicDecomposer(method='pca')
    assert decomposer_pca.method == 'pca'
    
    decomposer_svd = MusicDecomposer(method='svd')
    assert decomposer_svd.method == 'svd'
    
    # Test invalid method
    with pytest.raises(ValueError):
        MusicDecomposer(method='invalid')


def test_pca_decomposition():
    """Test PCA decomposition on random data."""
    # Create synthetic data
    np.random.seed(42)
    data = np.random.randn(100, 50)
    
    decomposer = MusicDecomposer(method='pca')
    components = decomposer.decompose_pca(data, n_components=10)
    
    # Check components
    assert 'components' in components
    assert 'transformed' in components
    assert 'mean' in components
    
    # Check shapes
    assert components['components'].shape == (10, 50)
    assert components['transformed'].shape == (100, 10)
    assert components['mean'].shape == (50,)
    
    # Check variance explained
    assert decomposer.variance_explained is not None
    assert len(decomposer.variance_explained) == 10
    assert np.all(decomposer.variance_explained >= 0)
    assert np.all(decomposer.variance_explained <= 1)


def test_svd_decomposition():
    """Test SVD decomposition on random data."""
    # Create synthetic data
    np.random.seed(42)
    data = np.random.randn(100, 50)
    
    decomposer = MusicDecomposer(method='svd')
    U, sigma, Vt = decomposer.decompose_svd(data, n_components=10)
    
    # Check shapes
    assert U.shape == (100, 10)
    assert sigma.shape == (10,)
    assert Vt.shape == (10, 50)
    
    # Check singular values are positive and decreasing
    assert np.all(sigma >= 0)
    assert np.all(sigma[:-1] >= sigma[1:])  # Decreasing order


def test_pca_reconstruction():
    """Test PCA reconstruction."""
    # Create synthetic data with known structure
    np.random.seed(42)
    data = np.random.randn(100, 50)
    
    decomposer = MusicDecomposer(method='pca')
    components = decomposer.decompose_pca(data, n_components=10)
    
    # Reconstruct
    reconstructed = decomposer.reconstruct_pca(components, n_components=10)
    
    # Check shape
    assert reconstructed.shape == data.shape
    
    # Reconstruction should be similar to original (with some loss)
    # We can't expect perfect reconstruction with reduced components
    mse = np.mean((data - reconstructed) ** 2)
    assert mse < np.mean(data ** 2)  # MSE should be less than data variance


def test_svd_reconstruction():
    """Test SVD reconstruction."""
    # Create synthetic data
    np.random.seed(42)
    data = np.random.randn(100, 50)
    
    decomposer = MusicDecomposer(method='svd')
    U, sigma, Vt = decomposer.decompose_svd(data, n_components=10)
    
    # Reconstruct
    reconstructed = decomposer.reconstruct_svd(U, sigma, Vt, n_components=10)
    
    # Check shape
    assert reconstructed.shape == data.shape


def test_blended_reconstruction():
    """Test blending multiple components."""
    # Create synthetic data for multiple "songs"
    np.random.seed(42)
    data1 = np.random.randn(100, 50)
    data2 = np.random.randn(100, 50)
    
    decomposer = MusicDecomposer(method='pca')
    
    # Decompose both
    comp1 = decomposer.decompose_pca(data1, n_components=5)
    comp2 = decomposer.decompose_pca(data2, n_components=5)
    
    # Blend
    weights = np.array([0.5, 0.5])
    blended = decomposer.reconstruct_blended([comp1, comp2], weights)
    
    # Check that result has reasonable shape
    assert blended.shape[1] == 50  # Features dimension preserved


if __name__ == '__main__':
    # Run tests
    test_decomposer_initialization()
    print("✅ Initialization test passed")
    
    test_pca_decomposition()
    print("✅ PCA decomposition test passed")
    
    test_svd_decomposition()
    print("✅ SVD decomposition test passed")
    
    test_pca_reconstruction()
    print("✅ PCA reconstruction test passed")
    
    test_svd_reconstruction()
    print("✅ SVD reconstruction test passed")
    
    test_blended_reconstruction()
    print("✅ Blended reconstruction test passed")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
