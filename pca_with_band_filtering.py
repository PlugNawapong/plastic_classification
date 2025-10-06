"""
Enhanced PCA with Band Quality Pre-filtering

This module implements a two-stage approach:
1. Filter out noisy/low-quality bands BEFORE PCA (saves time!)
2. Apply PCA on clean bands only

Benefits:
- Faster PCA fitting (fewer bands to process)
- Better PCA quality (no noise contamination)
- More interpretable components
- Adjustable quality thresholds

Usage:
    selector = PCAWithBandFiltering(
        snr_threshold=10.0,        # Minimum SNR
        variance_threshold=0.001,   # Minimum variance
        saturation_threshold=5.0,   # Max % saturated pixels
        keep_percentage=80.0        # Or use percentile-based filtering
    )
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
from tqdm import tqdm


class BandQualityFilter:
    """Filter bands based on quality metrics before PCA."""

    def __init__(self,
                 snr_threshold: float = None,
                 variance_threshold: float = None,
                 saturation_threshold: float = None,
                 darkness_threshold: float = None,
                 keep_percentage: float = None):
        """
        Args:
            snr_threshold: Minimum Signal-to-Noise Ratio (e.g., 10.0)
            variance_threshold: Minimum variance (e.g., 0.001)
            saturation_threshold: Maximum % of saturated pixels (e.g., 5.0)
            darkness_threshold: Maximum % of dark pixels (e.g., 5.0)
            keep_percentage: Keep top X% of bands by SNR (e.g., 80.0)
                           If set, overrides individual thresholds
        """
        self.snr_threshold = snr_threshold
        self.variance_threshold = variance_threshold
        self.saturation_threshold = saturation_threshold
        self.darkness_threshold = darkness_threshold
        self.keep_percentage = keep_percentage

        self.band_metrics = None
        self.good_band_indices = None
        self.filtered_wavelengths = None

    def calculate_band_metrics(self, hypercube: np.ndarray, wavelengths: List[float] = None):
        """Calculate quality metrics for each band."""
        n_bands = hypercube.shape[0]
        metrics = []

        print(f"\nCalculating quality metrics for {n_bands} bands...")

        for i in tqdm(range(n_bands), desc="Band quality analysis"):
            band = hypercube[i]

            # SNR: signal-to-noise ratio
            mean_val = np.mean(band)
            std_val = np.std(band)
            snr = mean_val / (std_val + 1e-8)

            # Variance: information content
            variance = np.var(band)

            # Saturation: percentage of pixels at max value
            saturation_pct = (np.sum(band >= 0.98) / band.size) * 100

            # Darkness: percentage of pixels near zero
            darkness_pct = (np.sum(band <= 0.02) / band.size) * 100

            metrics.append({
                'band_idx': i,
                'wavelength': wavelengths[i] if wavelengths else i,
                'snr': snr,
                'variance': variance,
                'mean': mean_val,
                'std': std_val,
                'saturation_pct': saturation_pct,
                'darkness_pct': darkness_pct
            })

        self.band_metrics = metrics
        return metrics

    def filter_bands(self, hypercube: np.ndarray, wavelengths: List[float] = None) -> Tuple[np.ndarray, List[int], List[float]]:
        """
        Filter bands based on quality criteria.

        Returns:
            filtered_hypercube: Hypercube with only good bands
            good_indices: Indices of good bands
            filtered_wavelengths: Wavelengths of good bands
        """
        if self.band_metrics is None:
            self.calculate_band_metrics(hypercube, wavelengths)

        n_bands = len(self.band_metrics)

        print(f"\n{'='*80}")
        print(f"BAND QUALITY FILTERING")
        print(f"{'='*80}")

        # Method 1: Percentile-based (keeps top X% by SNR)
        if self.keep_percentage is not None:
            snr_values = [m['snr'] for m in self.band_metrics]
            percentile = 100 - self.keep_percentage
            snr_cutoff = np.percentile(snr_values, percentile)

            good_indices = [i for i, m in enumerate(self.band_metrics) if m['snr'] >= snr_cutoff]

            print(f"Filter method: Percentile-based (keep top {self.keep_percentage}%)")
            print(f"SNR cutoff: {snr_cutoff:.2f}")

        # Method 2: Threshold-based
        else:
            good_indices = list(range(n_bands))

            print(f"Filter method: Threshold-based")

            # Apply SNR threshold
            if self.snr_threshold is not None:
                before = len(good_indices)
                good_indices = [i for i in good_indices
                              if self.band_metrics[i]['snr'] >= self.snr_threshold]
                removed = before - len(good_indices)
                print(f"  SNR threshold ({self.snr_threshold}): Removed {removed} bands")

            # Apply variance threshold
            if self.variance_threshold is not None:
                before = len(good_indices)
                good_indices = [i for i in good_indices
                              if self.band_metrics[i]['variance'] >= self.variance_threshold]
                removed = before - len(good_indices)
                print(f"  Variance threshold ({self.variance_threshold}): Removed {removed} bands")

            # Apply saturation threshold
            if self.saturation_threshold is not None:
                before = len(good_indices)
                good_indices = [i for i in good_indices
                              if self.band_metrics[i]['saturation_pct'] <= self.saturation_threshold]
                removed = before - len(good_indices)
                print(f"  Saturation threshold ({self.saturation_threshold}%): Removed {removed} bands")

            # Apply darkness threshold
            if self.darkness_threshold is not None:
                before = len(good_indices)
                good_indices = [i for i in good_indices
                              if self.band_metrics[i]['darkness_pct'] <= self.darkness_threshold]
                removed = before - len(good_indices)
                print(f"  Darkness threshold ({self.darkness_threshold}%): Removed {removed} bands")

        # Results
        self.good_band_indices = good_indices
        filtered_hypercube = hypercube[good_indices]

        if wavelengths:
            self.filtered_wavelengths = [wavelengths[i] for i in good_indices]
        else:
            self.filtered_wavelengths = good_indices

        print(f"\nFiltering Results:")
        print(f"  Original bands: {n_bands}")
        print(f"  Filtered bands: {len(good_indices)}")
        print(f"  Removed bands: {n_bands - len(good_indices)} ({(n_bands - len(good_indices))/n_bands*100:.1f}%)")
        print(f"  Wavelength range: {self.filtered_wavelengths[0]:.1f} - {self.filtered_wavelengths[-1]:.1f} nm")

        # Quality stats
        good_metrics = [self.band_metrics[i] for i in good_indices]
        avg_snr = np.mean([m['snr'] for m in good_metrics])
        avg_var = np.mean([m['variance'] for m in good_metrics])

        print(f"\nQuality of filtered bands:")
        print(f"  Average SNR: {avg_snr:.2f}")
        print(f"  Average variance: {avg_var:.6f}")

        return filtered_hypercube, good_indices, self.filtered_wavelengths

    def visualize_filtering(self, wavelengths: List[float], output_path: str = None):
        """Visualize which bands were kept vs removed."""
        if self.band_metrics is None:
            raise ValueError("Run filter_bands() first")

        n_bands = len(self.band_metrics)
        good_set = set(self.good_band_indices)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # SNR per band
        ax = axes[0, 0]
        snr_values = [m['snr'] for m in self.band_metrics]
        colors = ['green' if i in good_set else 'red' for i in range(n_bands)]
        ax.scatter(wavelengths, snr_values, c=colors, alpha=0.6, s=10)
        if self.snr_threshold:
            ax.axhline(self.snr_threshold, color='orange', linestyle='--', label=f'Threshold: {self.snr_threshold}')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('SNR')
        ax.set_title('Signal-to-Noise Ratio per Band\n(Green=Kept, Red=Removed)', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # Variance per band
        ax = axes[0, 1]
        var_values = [m['variance'] for m in self.band_metrics]
        ax.scatter(wavelengths, var_values, c=colors, alpha=0.6, s=10)
        if self.variance_threshold:
            ax.axhline(self.variance_threshold, color='orange', linestyle='--', label=f'Threshold: {self.variance_threshold}')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Variance')
        ax.set_title('Variance per Band', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # Saturation
        ax = axes[1, 0]
        sat_values = [m['saturation_pct'] for m in self.band_metrics]
        ax.scatter(wavelengths, sat_values, c=colors, alpha=0.6, s=10)
        if self.saturation_threshold:
            ax.axhline(self.saturation_threshold, color='orange', linestyle='--', label=f'Threshold: {self.saturation_threshold}%')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Saturation (%)')
        ax.set_title('Pixel Saturation per Band', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # Distribution
        ax = axes[1, 1]
        ax.axis('off')

        summary_text = f"""
Band Filtering Summary

Total bands: {n_bands}
Kept: {len(self.good_band_indices)} ({len(self.good_band_indices)/n_bands*100:.1f}%)
Removed: {n_bands - len(self.good_band_indices)} ({(n_bands - len(self.good_band_indices))/n_bands*100:.1f}%)

Filtering Criteria:
"""

        if self.keep_percentage:
            summary_text += f"  • Keep top {self.keep_percentage}% by SNR\n"
        else:
            if self.snr_threshold:
                summary_text += f"  • SNR ≥ {self.snr_threshold}\n"
            if self.variance_threshold:
                summary_text += f"  • Variance ≥ {self.variance_threshold}\n"
            if self.saturation_threshold:
                summary_text += f"  • Saturation ≤ {self.saturation_threshold}%\n"
            if self.darkness_threshold:
                summary_text += f"  • Darkness ≤ {self.darkness_threshold}%\n"

        good_metrics = [self.band_metrics[i] for i in self.good_band_indices]
        summary_text += f"""
Quality of Kept Bands:
  • Average SNR: {np.mean([m['snr'] for m in good_metrics]):.2f}
  • Average Variance: {np.mean([m['variance'] for m in good_metrics]):.6f}
  • Min SNR: {np.min([m['snr'] for m in good_metrics]):.2f}
  • Max SNR: {np.max([m['snr'] for m in good_metrics]):.2f}
        """

        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='center', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Filtering visualization saved to: {output_path}")

        plt.show()


class PCAWithBandFiltering:
    """PCA with automatic band quality pre-filtering."""

    def __init__(self,
                 # Band filtering parameters
                 snr_threshold: float = None,
                 variance_threshold: float = None,
                 saturation_threshold: float = 5.0,
                 darkness_threshold: float = 5.0,
                 keep_percentage: float = 80.0,
                 # PCA parameters
                 n_components: int = None,
                 pca_variance_threshold: float = 0.99,
                 standardize: bool = True):
        """
        Args:
            Band filtering (choose one method):
              snr_threshold: Minimum SNR (e.g., 10.0)
              variance_threshold: Minimum variance (e.g., 0.001)
              saturation_threshold: Max % saturated pixels (e.g., 5.0)
              darkness_threshold: Max % dark pixels (e.g., 5.0)
              keep_percentage: Keep top X% by SNR (e.g., 80.0) - RECOMMENDED

            PCA parameters:
              n_components: Number of PCA components (None = auto)
              pca_variance_threshold: Variance to retain (e.g., 0.99)
              standardize: Standardize before PCA
        """
        # Band filter
        self.band_filter = BandQualityFilter(
            snr_threshold=snr_threshold,
            variance_threshold=variance_threshold,
            saturation_threshold=saturation_threshold,
            darkness_threshold=darkness_threshold,
            keep_percentage=keep_percentage
        )

        # PCA parameters
        self.n_components = n_components
        self.pca_variance_threshold = pca_variance_threshold
        self.standardize = standardize

        # Models
        self.pca = None
        self.scaler = None
        self.n_components_selected = None
        self.explained_variance_ratio = None

        # Band mapping
        self.original_wavelengths = None
        self.filtered_wavelengths = None
        self.good_band_indices = None
        self.n_original_bands = None
        self.n_filtered_bands = None

    def fit(self, hypercube: np.ndarray, wavelengths: List[float] = None):
        """
        Fit PCA on filtered bands.

        Args:
            hypercube: Shape (n_bands, height, width)
            wavelengths: Wavelength values for each band
        """
        self.original_wavelengths = wavelengths
        self.n_original_bands = hypercube.shape[0]

        print(f"\n{'='*80}")
        print(f"PCA WITH BAND FILTERING")
        print(f"{'='*80}")
        print(f"Original bands: {self.n_original_bands}")

        # Step 1: Filter noisy bands
        filtered_hypercube, good_indices, filtered_wavelengths = self.band_filter.filter_bands(
            hypercube, wavelengths
        )

        self.good_band_indices = good_indices
        self.filtered_wavelengths = filtered_wavelengths
        self.n_filtered_bands = len(good_indices)

        # Step 2: Apply PCA on clean bands only
        print(f"\n{'='*80}")
        print(f"PCA FITTING ON FILTERED BANDS")
        print(f"{'='*80}")

        n_bands, height, width = filtered_hypercube.shape
        X = filtered_hypercube.reshape(n_bands, -1).T  # (n_pixels, n_bands)

        print(f"Input shape: {X.shape} (samples, bands)")

        # Standardize
        if self.standardize:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
            print(f"✓ Data standardized")

        # Determine n_components
        if self.n_components is None:
            pca_full = PCA()
            pca_full.fit(X)
            cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
            self.n_components_selected = np.argmax(cumsum_variance >= self.pca_variance_threshold) + 1
            print(f"✓ Auto-selected {self.n_components_selected} components "
                  f"(>{self.pca_variance_threshold*100:.0f}% variance)")
        else:
            self.n_components_selected = self.n_components
            print(f"✓ Using {self.n_components_selected} components (user-specified)")

        # Fit PCA
        self.pca = PCA(n_components=self.n_components_selected)
        self.pca.fit(X)
        self.explained_variance_ratio = self.pca.explained_variance_ratio_

        total_variance = np.sum(self.explained_variance_ratio)

        print(f"\n{'='*80}")
        print(f"FINAL RESULTS")
        print(f"{'='*80}")
        print(f"Dimensionality reduction:")
        print(f"  {self.n_original_bands} bands → {self.n_filtered_bands} filtered → {self.n_components_selected} PCA components")
        print(f"  Total reduction: {(1 - self.n_components_selected/self.n_original_bands)*100:.1f}%")
        print(f"\nPCA variance explained: {total_variance*100:.2f}%")
        print(f"Time savings: ~{(1 - self.n_filtered_bands/self.n_original_bands)*100:.0f}% faster PCA fitting")

        return self

    def transform(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Transform spectrum to PCA space.

        Args:
            spectrum: Full spectrum (n_original_bands,)

        Returns:
            PCA components (n_components,)
        """
        # Filter to good bands
        spectrum_filtered = spectrum[self.good_band_indices]

        # Reshape and standardize
        spectrum_2d = spectrum_filtered.reshape(1, -1)
        if self.standardize and self.scaler:
            spectrum_2d = self.scaler.transform(spectrum_2d)

        # PCA transform
        spectrum_pca = self.pca.transform(spectrum_2d)

        return spectrum_pca.flatten()

    def save(self, filepath: str):
        """Save model."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'band_filter': self.band_filter,
                'pca': self.pca,
                'scaler': self.scaler,
                'n_components_selected': self.n_components_selected,
                'explained_variance_ratio': self.explained_variance_ratio,
                'good_band_indices': self.good_band_indices,
                'filtered_wavelengths': self.filtered_wavelengths,
                'original_wavelengths': self.original_wavelengths,
                'n_original_bands': self.n_original_bands,
                'n_filtered_bands': self.n_filtered_bands,
            }, f)
        print(f"\n✓ Model saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Load model."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        obj = cls()
        obj.band_filter = data['band_filter']
        obj.pca = data['pca']
        obj.scaler = data['scaler']
        obj.n_components_selected = data['n_components_selected']
        obj.explained_variance_ratio = data['explained_variance_ratio']
        obj.good_band_indices = data['good_band_indices']
        obj.filtered_wavelengths = data['filtered_wavelengths']
        obj.original_wavelengths = data['original_wavelengths']
        obj.n_original_bands = data['n_original_bands']
        obj.n_filtered_bands = data['n_filtered_bands']

        print(f"✓ Model loaded from: {filepath}")
        print(f"  {obj.n_original_bands} bands → {obj.n_filtered_bands} filtered → {obj.n_components_selected} PCA components")

        return obj


def load_hypercube(dataset_path: str) -> Tuple[np.ndarray, List[float]]:
    """Load normalized hypercube."""
    dataset_path = Path(dataset_path)

    with open(dataset_path / 'header.json', 'r') as f:
        header = json.load(f)
    wavelengths = header['wavelength (nm)']

    print(f"Loading {len(wavelengths)} bands...")
    bands = []
    for i in tqdm(range(1, len(wavelengths) + 1), desc='Loading bands'):
        img_path = dataset_path / f'ImagesStack{i:03d}.png'
        if img_path.exists():
            img = np.array(Image.open(img_path).convert('L'), dtype=np.float32) / 255.0
            bands.append(img)

    hypercube = np.stack(bands, axis=0)
    print(f"✓ Hypercube loaded: {hypercube.shape}")

    return hypercube, wavelengths


def main():
    """Demonstrate band filtering + PCA."""

    print("\n" + "="*80)
    print("PCA WITH BAND QUALITY PRE-FILTERING")
    print("="*80)

    # Load data
    print("\n[1/5] Loading hypercube...")
    hypercube, wavelengths = load_hypercube('training_dataset')

    # Test different filtering strategies
    configs = [
        {
            'name': 'Conservative (keep 90%)',
            'params': {'keep_percentage': 90.0, 'n_components': None, 'pca_variance_threshold': 0.99}
        },
        {
            'name': 'Balanced (keep 80%)',
            'params': {'keep_percentage': 80.0, 'n_components': None, 'pca_variance_threshold': 0.99}
        },
        {
            'name': 'Aggressive (keep 70%)',
            'params': {'keep_percentage': 70.0, 'n_components': None, 'pca_variance_threshold': 0.99}
        },
    ]

    results = []

    for config in configs:
        print(f"\n{'='*80}")
        print(f"Testing: {config['name']}")
        print(f"{'='*80}")

        selector = PCAWithBandFiltering(**config['params'])
        selector.fit(hypercube, wavelengths)

        # Visualize
        selector.band_filter.visualize_filtering(
            wavelengths,
            output_path=f"band_filtering_{config['name'].replace(' ', '_')}.png"
        )

        results.append({
            'name': config['name'],
            'original_bands': selector.n_original_bands,
            'filtered_bands': selector.n_filtered_bands,
            'pca_components': selector.n_components_selected,
            'total_reduction': (1 - selector.n_components_selected/selector.n_original_bands) * 100
        })

    # Summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Strategy':<25} {'Original':<10} {'Filtered':<10} {'PCA':<10} {'Total Reduction':<15}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:<25} {r['original_bands']:<10} {r['filtered_bands']:<10} "
              f"{r['pca_components']:<10} {r['total_reduction']:.1f}%")

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("""
Start with 'Balanced (keep 80%)':
- Removes obviously noisy bands
- Keeps high-quality bands
- Faster PCA fitting
- Good balance of speed and quality

Adjust parameters based on your needs:
- More speed: Use 70% (aggressive)
- More quality: Use 90% (conservative)
- Custom thresholds: Set snr_threshold, variance_threshold manually
    """)


if __name__ == '__main__':
    main()
