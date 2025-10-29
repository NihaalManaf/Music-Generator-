"""
Music Decomposer Module

Handles PCA and SVD decomposition of music representations.
"""

import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import svd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union


class MusicDecomposer:
    """
    Decomposes music representations using PCA or SVD.
    
    This class provides methods for:
    - Decomposing music matrices into principal components
    - Reconstructing music from components
    - Analyzing variance explained by components
    - Visualizing components
    """
    
    def __init__(self, method: str = 'pca'):
        """
        Initialize the decomposer.
        
        Args:
            method: Decomposition method - 'pca' or 'svd'
        """
        self.method = method.lower()
        self.pca_model = None
        self.variance_explained = None
        
        if self.method not in ['pca', 'svd']:
            raise ValueError("Method must be 'pca' or 'svd'")
            
    def decompose_pca(self, data: np.ndarray, 
                      n_components: int = 10) -> Dict[str, np.ndarray]:
        """
        Decompose data using PCA.
        
        Args:
            data: Input music matrix (time x features)
            n_components: Number of principal components to extract
            
        Returns:
            Dictionary containing:
                - 'components': Principal components
                - 'transformed': Transformed data
                - 'mean': Mean of the data
        """
        # Reshape if necessary (flatten to 2D)
        original_shape = data.shape
        if len(data.shape) > 2:
            data_2d = data.reshape(data.shape[0], -1)
        else:
            data_2d = data
            
        # Apply PCA
        self.pca_model = PCA(n_components=n_components)
        transformed = self.pca_model.fit_transform(data_2d)
        
        # Store variance explained
        self.variance_explained = self.pca_model.explained_variance_ratio_
        
        return {
            'components': self.pca_model.components_,
            'transformed': transformed,
            'mean': self.pca_model.mean_,
            'original_shape': original_shape
        }
        
    def decompose_svd(self, data: np.ndarray,
                      n_components: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompose data using SVD.
        
        Args:
            data: Input music matrix (time x features)
            n_components: Number of singular values to keep
            
        Returns:
            Tuple of (U, sigma, Vt) where:
                - U: Left singular vectors (time patterns)
                - sigma: Singular values (importance)
                - Vt: Right singular vectors (frequency patterns)
        """
        # Reshape if necessary
        original_shape = data.shape
        if len(data.shape) > 2:
            data_2d = data.reshape(data.shape[0], -1)
        else:
            data_2d = data
            
        # Perform SVD
        U, sigma, Vt = svd(data_2d, full_matrices=False)
        
        # Keep only top n_components
        U = U[:, :n_components]
        sigma = sigma[:n_components]
        Vt = Vt[:n_components, :]
        
        # Calculate variance explained
        total_variance = np.sum(sigma ** 2)
        self.variance_explained = (sigma ** 2) / total_variance
        
        return U, sigma, Vt
        
    def reconstruct_pca(self, components_dict: Dict[str, np.ndarray],
                        n_components: Optional[int] = None) -> np.ndarray:
        """
        Reconstruct data from PCA components.
        
        Args:
            components_dict: Dictionary from decompose_pca
            n_components: Number of components to use (None = all)
            
        Returns:
            Reconstructed data
        """
        if n_components is None:
            n_components = components_dict['transformed'].shape[1]
            
        # Use subset of components
        transformed = components_dict['transformed'][:, :n_components]
        components = components_dict['components'][:n_components, :]
        
        # Reconstruct
        reconstructed = transformed @ components + components_dict['mean']
        
        # Reshape to original shape if needed
        if 'original_shape' in components_dict and len(components_dict['original_shape']) > 2:
            reconstructed = reconstructed.reshape(components_dict['original_shape'])
            
        return reconstructed
        
    def reconstruct_svd(self, U: np.ndarray, sigma: np.ndarray, 
                        Vt: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
        """
        Reconstruct data from SVD components.
        
        Args:
            U: Left singular vectors
            sigma: Singular values
            Vt: Right singular vectors
            n_components: Number of components to use (None = all)
            
        Returns:
            Reconstructed data
        """
        if n_components is None:
            n_components = len(sigma)
            
        # Use subset of components
        U_subset = U[:, :n_components]
        sigma_subset = sigma[:n_components]
        Vt_subset = Vt[:n_components, :]
        
        # Reconstruct: M = U @ diag(sigma) @ Vt
        reconstructed = U_subset @ np.diag(sigma_subset) @ Vt_subset
        
        return reconstructed
        
    def reconstruct_blended(self, components_list: List[Union[Dict, Tuple]],
                           weights: np.ndarray) -> np.ndarray:
        """
        Blend components from multiple songs to create new music.
        
        Args:
            components_list: List of component dictionaries/tuples
            weights: Weight for each song (should sum to 1.0)
            
        Returns:
            Blended music representation
        """
        if self.method == 'pca':
            # Blend PCA components
            blended_transformed = None
            
            for i, (comp_dict, weight) in enumerate(zip(components_list, weights)):
                if blended_transformed is None:
                    blended_transformed = weight * comp_dict['transformed']
                else:
                    # Align shapes if needed
                    min_len = min(blended_transformed.shape[0], 
                                 comp_dict['transformed'].shape[0])
                    blended_transformed = (
                        blended_transformed[:min_len] + 
                        weight * comp_dict['transformed'][:min_len]
                    )
                    
            # Use first song's components for reconstruction (could be improved)
            result = blended_transformed @ components_list[0]['components']
            result += components_list[0]['mean']
            
            return result
            
        else:  # SVD
            # Blend SVD components
            blended_U = None
            blended_sigma = None
            blended_Vt = None
            
            for comp_dict, weight in zip(components_list, weights):
                U, sigma, Vt = comp_dict['U'], comp_dict['sigma'], comp_dict['Vt']
                
                if blended_U is None:
                    blended_U = weight * U
                    blended_sigma = weight * sigma
                    blended_Vt = weight * Vt
                else:
                    # Align shapes and blend
                    min_u = min(blended_U.shape[0], U.shape[0])
                    min_vt = min(blended_Vt.shape[1], Vt.shape[1])
                    
                    blended_U[:min_u] += weight * U[:min_u]
                    blended_sigma += weight * sigma
                    blended_Vt[:, :min_vt] += weight * Vt[:, :min_vt]
                    
            # Reconstruct from blended components
            result = blended_U @ np.diag(blended_sigma) @ blended_Vt
            
            return result
            
    def plot_components(self, components: Union[Dict, Tuple], 
                       n_display: int = 5) -> None:
        """
        Visualize principal components.
        
        Args:
            components: Component dictionary or tuple
            n_display: Number of components to display
        """
        plt.figure(figsize=(15, 8))
        
        if isinstance(components, dict) and 'components' in components:
            # PCA components
            comp_array = components['components'][:n_display]
            
            for i, component in enumerate(comp_array):
                plt.subplot(n_display, 1, i + 1)
                plt.plot(component)
                plt.title(f'Principal Component {i+1}')
                plt.ylabel('Amplitude')
                
        else:
            # SVD components
            U, sigma, Vt = components
            
            # Plot singular values
            plt.subplot(2, 1, 1)
            plt.plot(sigma[:n_display], 'o-')
            plt.title('Singular Values')
            plt.ylabel('Value')
            plt.xlabel('Component')
            
            # Plot first few right singular vectors
            plt.subplot(2, 1, 2)
            for i in range(min(n_display, Vt.shape[0])):
                plt.plot(Vt[i, :100], label=f'V{i+1}')  # Plot first 100 values
            plt.title('Right Singular Vectors (first 100 values)')
            plt.legend()
            
        plt.tight_layout()
        plt.show()
        
    def plot_variance_explained(self) -> None:
        """
        Plot variance explained by components.
        """
        if self.variance_explained is None:
            print("No variance data available. Run decomposition first.")
            return
            
        plt.figure(figsize=(12, 5))
        
        # Individual variance
        plt.subplot(1, 2, 1)
        plt.bar(range(len(self.variance_explained)), self.variance_explained)
        plt.xlabel('Component')
        plt.ylabel('Variance Explained Ratio')
        plt.title('Variance Explained by Each Component')
        
        # Cumulative variance
        plt.subplot(1, 2, 2)
        cumsum = np.cumsum(self.variance_explained)
        plt.plot(cumsum, 'o-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Variance Explained')
        plt.title('Cumulative Variance Explained')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
