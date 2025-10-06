"""
PCA-based Band Selection for Hyperspectral Data

This module implements PCA (Principal Component Analysis) for:
1. Dimensionality reduction (459 bands → fewer essential bands)
2. Noise reduction through variance-based filtering
3. Feature extraction and visualization

PCA benefits for hyperspectral classification:
- Removes redundant spectral information
- Reduces computational cost
- Can improve generalization by filtering noise
- Retains most variance with fewer components
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json
from PIL import Image
from typing import Tuple, Dict, List, Optional
import pickle


class PCABandSelector:
    """
    PCA-based band selection for hyperspectral images.

    Workflow:
    1. Load normalized hypercube (after preprocessing)
    2. Fit PCA on training data
    3. Transform both training and test data
    4. Analyze variance explained and noise reduction
    """

    def __init__(self, n_components: Optional[int] = None,
                 variance_threshold: float = 0.99,
                 standardize: bool = True):
        """
        Args:
            n_components: Number of PCA components (None = auto from variance)
            variance_threshold: Cumulative variance to retain (e.g., 0.99 = 99%)
            standardize: Whether to standardize features before PCA
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.standardize = standardize

        self.pca = None
        self.scaler = None
        self.n_components_selected = None
        self.explained_variance_ratio = None
        self.wavelengths = None

    def fit(self, hypercube: np.ndarray, wavelengths: List[float] = None):
        """
        Fit PCA on training hypercube.

        Args:
            hypercube: Shape (n_bands, height, width) or (n_samples, n_bands)
            wavelengths: Wavelength values for each band
        """
        self.wavelengths = wavelengths

        # Reshape to (n_samples, n_features)
        if hypercube.ndim == 3:
            n_bands, height, width = hypercube.shape
            X = hypercube.reshape(n_bands, -1).T  # (n_pixels, n_bands)
        else:
            X = hypercube
            n_bands = X.shape[1]

        print(f"\n{'='*80}")
        print(f"PCA FITTING")
        print(f"{'='*80}")
        print(f"Input shape: {X.shape} (samples, bands)")
        print(f"Number of bands: {n_bands}")

        # Standardize if requested
        if self.standardize:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
            print(f"✓ Data standardized (mean=0, std=1)")

        # Determine n_components
        if self.n_components is None:
            # Use variance threshold to determine components
            pca_full = PCA()
            pca_full.fit(X)

            cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
            self.n_components_selected = np.argmax(cumsum_variance >= self.variance_threshold) + 1

            print(f"✓ Auto-selected {self.n_components_selected} components "
                  f"(>{self.variance_threshold*100:.0f}% variance)")
        else:
            self.n_components_selected = self.n_components
            print(f"✓ Using {self.n_components_selected} components (user-specified)")

        # Fit final PCA
        self.pca = PCA(n_components=self.n_components_selected)
        self.pca.fit(X)

        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        total_variance = np.sum(self.explained_variance_ratio)

        print(f"\nPCA Results:")
        print(f"  Components: {n_bands} → {self.n_components_selected}")
        print(f"  Reduction ratio: {self.n_components_selected/n_bands:.2%}")
        print(f"  Variance explained: {total_variance*100:.2f}%")
        print(f"  Variance lost: {(1-total_variance)*100:.2f}%")

        # Top components variance
        print(f"\nTop 5 components:")
        for i in range(min(5, len(self.explained_variance_ratio))):
            print(f"  PC{i+1}: {self.explained_variance_ratio[i]*100:.2f}%")

    def transform(self, hypercube: np.ndarray) -> np.ndarray:
        """
        Transform hypercube to PCA space.

        Args:
            hypercube: Shape (n_bands, height, width)

        Returns:
            Reduced hypercube of shape (n_components, height, width)
        """
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit() first.")

        n_bands, height, width = hypercube.shape

        # Reshape to (n_samples, n_features)
        X = hypercube.reshape(n_bands, -1).T

        # Standardize if needed
        if self.standardize and self.scaler is not None:
            X = self.scaler.transform(X)

        # Transform
        X_transformed = self.pca.transform(X)

        # Reshape back to (n_components, height, width)
        reduced_cube = X_transformed.T.reshape(self.n_components_selected, height, width)

        return reduced_cube

    def inverse_transform(self, reduced_cube: np.ndarray) -> np.ndarray:
        """
        Reconstruct original hypercube from PCA space.

        Useful for:
        - Visualizing what information is retained
        - Measuring reconstruction error (noise)

        Args:
            reduced_cube: Shape (n_components, height, width)

        Returns:
            Reconstructed hypercube of shape (n_bands, height, width)
        """
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit() first.")

        n_components, height, width = reduced_cube.shape

        # Reshape to (n_samples, n_components)
        X_reduced = reduced_cube.reshape(n_components, -1).T

        # Inverse transform
        X_reconstructed = self.pca.inverse_transform(X_reduced)

        # Inverse standardization if needed
        if self.standardize and self.scaler is not None:
            X_reconstructed = self.scaler.inverse_transform(X_reconstructed)

        # Reshape back to (n_bands, height, width)
        n_bands = self.pca.n_features_in_
        reconstructed_cube = X_reconstructed.T.reshape(n_bands, height, width)

        return reconstructed_cube

    def calculate_reconstruction_error(self, original_cube: np.ndarray,
                                       reduced_cube: np.ndarray) -> Dict:
        """
        Calculate reconstruction error to assess information loss.

        Lower error = better PCA compression = less information loss

        Args:
            original_cube: Original hypercube (n_bands, H, W)
            reduced_cube: PCA-reduced hypercube (n_components, H, W)

        Returns:
            Dictionary with error metrics
        """
        reconstructed = self.inverse_transform(reduced_cube)

        # Calculate errors
        mse = np.mean((original_cube - reconstructed) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(original_cube - reconstructed))

        # Normalized error (relative to signal range)
        signal_range = original_cube.max() - original_cube.min()
        normalized_rmse = rmse / signal_range if signal_range > 0 else 0

        # Per-band error
        per_band_error = np.mean((original_cube - reconstructed) ** 2, axis=(1, 2))

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'normalized_rmse': normalized_rmse,
            'per_band_error': per_band_error,
            'mean_per_band_error': np.mean(per_band_error),
            'max_per_band_error': np.max(per_band_error),
            'reconstruction_quality': 1 - normalized_rmse  # 1 = perfect, 0 = worst
        }

    def visualize_variance_explained(self, output_path: str = None):
        """Visualize cumulative variance explained by components."""
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit() first.")

        cumsum_variance = np.cumsum(self.explained_variance_ratio)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Individual variance
        ax = axes[0]
        ax.bar(range(1, len(self.explained_variance_ratio) + 1),
               self.explained_variance_ratio * 100,
               alpha=0.7, color='steelblue')
        ax.set_xlabel('Principal Component', fontsize=12)
        ax.set_ylabel('Variance Explained (%)', fontsize=12)
        ax.set_title('Variance Explained by Each Component', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Cumulative variance
        ax = axes[1]
        ax.plot(range(1, len(cumsum_variance) + 1), cumsum_variance * 100,
                linewidth=2, marker='o', markersize=4, color='darkgreen')
        ax.axhline(y=95, color='red', linestyle='--', label='95% threshold')
        ax.axhline(y=99, color='orange', linestyle='--', label='99% threshold')
        ax.set_xlabel('Number of Components', fontsize=12)
        ax.set_ylabel('Cumulative Variance Explained (%)', fontsize=12)
        ax.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Variance plot saved to: {output_path}")

        plt.show()

    def visualize_principal_components(self, output_path: str = None, n_show: int = 6):
        """Visualize top principal components as spatial maps."""
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit() first.")

        n_components = min(n_show, self.pca.n_components_)
        n_rows = (n_components + 2) // 3

        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_components > 1 else [axes]

        for i in range(n_components):
            ax = axes[i]

            # Get component loadings (weights for each original band)
            loadings = self.pca.components_[i]

            # Plot as bar chart
            x = range(len(loadings))
            ax.bar(x, loadings, alpha=0.7, color='steelblue')
            ax.set_title(f'PC{i+1} ({self.explained_variance_ratio[i]*100:.1f}% variance)',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Band Index' if self.wavelengths is None else 'Wavelength (nm)',
                         fontsize=10)
            ax.set_ylabel('Loading', fontsize=10)
            ax.grid(axis='y', alpha=0.3)

            # Add wavelength ticks if available
            if self.wavelengths is not None and len(self.wavelengths) == len(loadings):
                n_ticks = 10
                step = max(1, len(loadings) // n_ticks)
                tick_pos = range(0, len(loadings), step)
                tick_labels = [f"{self.wavelengths[i]:.0f}" for i in tick_pos]
                ax.set_xticks(tick_pos)
                ax.set_xticklabels(tick_labels, rotation=45)

        # Hide unused subplots
        for i in range(n_components, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Component visualization saved to: {output_path}")

        plt.show()

    def visualize_noise_reduction(self, original_cube: np.ndarray,
                                  reduced_cube: np.ndarray,
                                  band_idx: int = None,
                                  output_path: str = None):
        """
        Visualize noise reduction effect of PCA.

        Shows: Original → PCA → Reconstructed → Noise Removed
        """
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit() first.")

        reconstructed = self.inverse_transform(reduced_cube)
        noise = original_cube - reconstructed

        # Auto-select middle band if not specified
        if band_idx is None:
            band_idx = original_cube.shape[0] // 2

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Original
        ax = axes[0]
        im = ax.imshow(original_cube[band_idx], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Original Band {band_idx}\n' +
                    (f'({self.wavelengths[band_idx]:.0f} nm)' if self.wavelengths else ''),
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Reconstructed
        ax = axes[1]
        im = ax.imshow(reconstructed[band_idx], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'PCA Reconstructed\n({self.n_components_selected} components)',
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Noise removed
        ax = axes[2]
        im = ax.imshow(np.abs(noise[band_idx]), cmap='hot', vmin=0, vmax=0.1)
        ax.set_title('Noise Removed by PCA\n(High-frequency noise)',
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Histogram comparison
        ax = axes[3]
        ax.hist(original_cube[band_idx].flatten(), bins=50, alpha=0.5,
               label='Original', color='blue', density=True)
        ax.hist(reconstructed[band_idx].flatten(), bins=50, alpha=0.5,
               label='Reconstructed', color='green', density=True)
        ax.set_xlabel('Intensity', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Intensity Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Noise reduction visualization saved to: {output_path}")

        plt.show()

    def save_model(self, filepath: str):
        """Save PCA model for inference."""
        model_data = {
            'pca': self.pca,
            'scaler': self.scaler,
            'n_components_selected': self.n_components_selected,
            'explained_variance_ratio': self.explained_variance_ratio,
            'wavelengths': self.wavelengths,
            'standardize': self.standardize,
            'variance_threshold': self.variance_threshold
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n✓ PCA model saved to: {filepath}")

    def load_model(self, filepath: str):
        """Load PCA model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.pca = model_data['pca']
        self.scaler = model_data['scaler']
        self.n_components_selected = model_data['n_components_selected']
        self.explained_variance_ratio = model_data['explained_variance_ratio']
        self.wavelengths = model_data.get('wavelengths')
        self.standardize = model_data.get('standardize', True)
        self.variance_threshold = model_data.get('variance_threshold', 0.99)

        print(f"✓ PCA model loaded from: {filepath}")
        print(f"  Components: {self.n_components_selected}")
        print(f"  Variance explained: {np.sum(self.explained_variance_ratio)*100:.2f}%")


def load_normalized_hypercube(dataset_path: str) -> Tuple[np.ndarray, List[float]]:
    """
    Load pre-normalized hypercube from dataset.

    Assumes normalization has already been done (see preprocessing.py)
    """
    dataset_path = Path(dataset_path)

    # Load wavelengths
    header_path = dataset_path / 'header.json'
    with open(header_path, 'r') as f:
        header = json.load(f)
    wavelengths = header['wavelength (nm)']

    # Load all bands
    bands = []
    for i in range(1, len(wavelengths) + 1):
        img_path = dataset_path / f'ImagesStack{i:03d}.png'
        if img_path.exists():
            img = np.array(Image.open(img_path).convert('L'), dtype=np.float32) / 255.0
            bands.append(img)

    hypercube = np.stack(bands, axis=0)

    return hypercube, wavelengths


def analyze_pca_for_noise_reduction(original_cube: np.ndarray,
                                    reduced_cube: np.ndarray,
                                    pca_selector: PCABandSelector) -> Dict:
    """
    Comprehensive analysis of PCA's noise reduction capability.

    Returns:
        Dictionary with noise analysis metrics
    """
    reconstructed = pca_selector.inverse_transform(reduced_cube)
    noise = original_cube - reconstructed

    # Signal and noise statistics
    signal_power = np.mean(original_cube ** 2)
    noise_power = np.mean(noise ** 2)
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

    # Spatial noise analysis
    noise_std = np.std(noise, axis=(1, 2))  # Per-band noise std

    # Frequency analysis (high-freq noise reduction)
    noise_freq = np.abs(np.fft.fft2(noise, axes=(1, 2)))
    high_freq_noise = np.mean(noise_freq[:, -10:, -10:])

    return {
        'signal_power': signal_power,
        'noise_power': noise_power,
        'snr_db': snr_db,
        'noise_std_mean': np.mean(noise_std),
        'noise_std_max': np.max(noise_std),
        'high_freq_noise': high_freq_noise,
        'noise_spatial_pattern': noise_std,  # Per-band
    }


def main():
    """
    Demonstrate PCA band selection and noise reduction.
    """
    print("\n" + "="*80)
    print("PCA BAND SELECTION FOR HYPERSPECTRAL DATA")
    print("="*80)

    # Load normalized training data
    print("\n[1/6] Loading normalized hypercube...")
    hypercube, wavelengths = load_normalized_hypercube('training_dataset')
    print(f"✓ Loaded: {hypercube.shape} (bands, height, width)")
    print(f"✓ Wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")

    # Test different component counts
    component_options = [50, 100, 150, 200]

    print(f"\n[2/6] Testing PCA with different component counts...")

    for n_comp in component_options:
        print(f"\n--- Testing n_components = {n_comp} ---")

        # Fit PCA
        pca_selector = PCABandSelector(n_components=n_comp, standardize=True)
        pca_selector.fit(hypercube, wavelengths=wavelengths)

        # Transform
        reduced_cube = pca_selector.transform(hypercube)
        print(f"✓ Transformed: {hypercube.shape} → {reduced_cube.shape}")

        # Calculate reconstruction error
        error_metrics = pca_selector.calculate_reconstruction_error(hypercube, reduced_cube)
        print(f"\nReconstruction Quality:")
        print(f"  RMSE: {error_metrics['rmse']:.6f}")
        print(f"  Normalized RMSE: {error_metrics['normalized_rmse']:.4f}")
        print(f"  Reconstruction Quality: {error_metrics['reconstruction_quality']*100:.2f}%")

        # Noise reduction analysis
        noise_analysis = analyze_pca_for_noise_reduction(hypercube, reduced_cube, pca_selector)
        print(f"\nNoise Reduction:")
        print(f"  SNR: {noise_analysis['snr_db']:.2f} dB")
        print(f"  Noise power: {noise_analysis['noise_power']:.6f}")

    # Use optimal components (99% variance)
    print(f"\n[3/6] Fitting PCA with auto-selected components (99% variance)...")
    pca_selector = PCABandSelector(n_components=None, variance_threshold=0.99, standardize=True)
    pca_selector.fit(hypercube, wavelengths=wavelengths)

    # Transform
    print(f"\n[4/6] Transforming hypercube...")
    reduced_cube = pca_selector.transform(hypercube)

    # Visualizations
    print(f"\n[5/6] Creating visualizations...")

    pca_selector.visualize_variance_explained(output_path='pca_variance_explained.png')
    pca_selector.visualize_principal_components(output_path='pca_components.png', n_show=9)
    pca_selector.visualize_noise_reduction(hypercube, reduced_cube,
                                          output_path='pca_noise_reduction.png')

    # Save model
    print(f"\n[6/6] Saving PCA model...")
    pca_selector.save_model('pca_model.pkl')

    # Summary
    print(f"\n{'='*80}")
    print("PCA ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nKey Findings:")
    print(f"  Original bands: {hypercube.shape[0]}")
    print(f"  PCA components: {pca_selector.n_components_selected}")
    print(f"  Reduction: {(1 - pca_selector.n_components_selected/hypercube.shape[0])*100:.1f}%")
    print(f"  Variance retained: {np.sum(pca_selector.explained_variance_ratio)*100:.2f}%")

    error_metrics = pca_selector.calculate_reconstruction_error(hypercube, reduced_cube)
    print(f"  Reconstruction quality: {error_metrics['reconstruction_quality']*100:.2f}%")

    print(f"\nGenerated files:")
    print(f"  • pca_variance_explained.png - Variance analysis")
    print(f"  • pca_components.png - Principal component loadings")
    print(f"  • pca_noise_reduction.png - Noise reduction visualization")
    print(f"  • pca_model.pkl - Trained PCA model")

    print(f"\n{'='*80}")
    print("DOES PCA REDUCE NOISE AFTER PREDICTION?")
    print(f"{'='*80}")
    print("""
YES, PCA can reduce noise in predictions through multiple mechanisms:

1. VARIANCE-BASED FILTERING:
   - PCA components are ordered by variance (PC1 > PC2 > ... > PCn)
   - High-variance components = signal (meaningful patterns)
   - Low-variance components = noise (random fluctuations)
   - By keeping top components, we filter out low-variance noise

2. DIMENSIONALITY REDUCTION:
   - Original: 459 bands (high-dimensional, prone to overfitting)
   - PCA: ~100-200 components (reduced dimensionality)
   - Lower dimensions → better generalization → less noisy predictions

3. SPECTRAL SMOOTHING:
   - PCA reconstruction averages correlated bands
   - Acts as a form of spectral filtering
   - Removes high-frequency spectral noise

4. IMPROVED MODEL TRAINING:
   - Fewer features → faster training
   - Less overfitting to noisy bands
   - Better convergence

WHEN DOES PCA HELP?
✓ When spectral bands are highly correlated (redundant)
✓ When dataset has noise in higher-frequency bands
✓ When model tends to overfit (too many features)
✓ When computational efficiency is important

WHEN MIGHT PCA NOT HELP?
✗ If all bands contain unique information (rare in hyperspectral)
✗ If critical features are in low-variance components
✗ If preprocessing already removed most noise

RECOMMENDATION:
1. Train model WITHOUT PCA first (baseline)
2. Train model WITH PCA (various component counts: 50, 100, 150, 200)
3. Compare validation accuracy and prediction smoothness
4. Choose configuration with best accuracy/efficiency tradeoff

Based on your 459 bands:
- Expected PCA components for 99% variance: ~100-200
- Expected reduction: 60-80%
- Expected noise reduction: Moderate to significant
- Recommended: Start with 150 components
    """)


if __name__ == '__main__':
    main()
