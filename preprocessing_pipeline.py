"""
Complete Preprocessing Pipeline with Noise Removal and Step-by-Step Visualization

This pipeline helps identify and remove noise from hyperspectral images:
1. Load raw bands
2. Filter out noisy bands (based on quality metrics)
3. Apply denoising (spatial filtering)
4. Apply normalization (band-wise with brightness boost)
5. Visualize each step to identify where noise comes from
"""

import numpy as np
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, gaussian_filter
from typing import Tuple, List, Dict
import cv2


class NoiseRemovalPreprocessor:
    """Preprocessing with noise removal and visualization"""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.wavelengths = self._load_wavelengths()
        self.n_bands = len(self.wavelengths)

    def _load_wavelengths(self) -> List[float]:
        """Load wavelengths from header.json"""
        header_path = self.dataset_path / 'header.json'
        with open(header_path, 'r') as f:
            header = json.load(f)
        return header['wavelength (nm)']

    def load_hypercube(self) -> np.ndarray:
        """Load all spectral bands from dataset"""
        bands = []
        for band_idx in range(1, self.n_bands + 1):  # Bands are 1-indexed (001-458)
            img_path = self.dataset_path / f'ImagesStack{band_idx:03d}.png'
            if img_path.exists():
                img = np.array(Image.open(img_path).convert('L'), dtype=np.float32)
                bands.append(img)
            else:
                print(f"Warning: Band {band_idx} not found at {img_path}")
                # If band doesn't exist, use zeros
                if bands:
                    bands.append(np.zeros_like(bands[0]))

        if not bands:
            raise ValueError(f"No bands found in {self.dataset_path}")

        return np.stack(bands, axis=0)  # Shape: (n_bands, H, W)

    def calculate_band_quality(self, hypercube: np.ndarray) -> List[Dict]:
        """Calculate quality metrics for each band"""
        metrics = []

        for band_idx in range(hypercube.shape[0]):
            band = hypercube[band_idx]

            # SNR: signal-to-noise ratio
            mean_val = np.mean(band)
            std_val = np.std(band)
            snr = mean_val / (std_val + 1e-8)

            # Variance: information content
            variance = np.var(band)

            # Saturation: percentage of pixels at max value
            saturation_pct = (np.sum(band >= 250) / band.size) * 100

            # Darkness: percentage of pixels near zero
            darkness_pct = (np.sum(band <= 5) / band.size) * 100

            metrics.append({
                'band_idx': band_idx,
                'wavelength': self.wavelengths[band_idx],
                'snr': snr,
                'variance': variance,
                'mean': mean_val,
                'std': std_val,
                'saturation_pct': saturation_pct,
                'darkness_pct': darkness_pct
            })

        return metrics

    def filter_noisy_bands(self, hypercube: np.ndarray,
                          keep_percentage: float = 75.0) -> Tuple[np.ndarray, List[int]]:
        """
        Filter out noisy bands based on SNR percentile

        Returns:
            clean_hypercube: Only clean bands
            clean_indices: Indices of clean bands
        """
        metrics = self.calculate_band_quality(hypercube)

        # Use SNR percentile
        snr_values = [m['snr'] for m in metrics]
        percentile = 100 - keep_percentage
        snr_threshold = np.percentile(snr_values, percentile)

        # Select clean bands
        clean_indices = [i for i, m in enumerate(metrics) if m['snr'] >= snr_threshold]
        clean_hypercube = hypercube[clean_indices]

        print(f"\n{'='*80}")
        print(f"BAND FILTERING RESULTS")
        print(f"{'='*80}")
        print(f"Total bands:           {len(metrics)}")
        print(f"Clean bands:           {len(clean_indices)} ({len(clean_indices)/len(metrics)*100:.1f}%)")
        print(f"SNR threshold:         {snr_threshold:.2f}")
        print(f"Wavelength range:      {self.wavelengths[clean_indices[0]]:.1f} - {self.wavelengths[clean_indices[-1]]:.1f} nm")

        return clean_hypercube, clean_indices

    def denoise_spatial(self, hypercube: np.ndarray, method: str = 'median') -> np.ndarray:
        """
        Apply spatial denoising to remove salt-and-pepper and Gaussian noise

        Methods:
        - 'median': Median filter (good for salt-and-pepper noise)
        - 'gaussian': Gaussian filter (good for Gaussian noise)
        - 'bilateral': Bilateral filter (edge-preserving)
        - 'nlm': Non-local means (best quality, slowest)
        """
        denoised = np.zeros_like(hypercube)

        if method == 'median':
            # Median filter: preserves edges, removes salt-and-pepper
            for i in range(hypercube.shape[0]):
                denoised[i] = median_filter(hypercube[i], size=3)

        elif method == 'gaussian':
            # Gaussian filter: smooth noise, blurs edges
            for i in range(hypercube.shape[0]):
                denoised[i] = gaussian_filter(hypercube[i], sigma=1.0)

        elif method == 'bilateral':
            # Bilateral filter: edge-preserving smoothing
            for i in range(hypercube.shape[0]):
                band_uint8 = np.clip(hypercube[i], 0, 255).astype(np.uint8)
                denoised[i] = cv2.bilateralFilter(band_uint8, d=5, sigmaColor=50, sigmaSpace=50)

        elif method == 'nlm':
            # Non-local means: best quality, preserves details
            for i in range(hypercube.shape[0]):
                band_uint8 = np.clip(hypercube[i], 0, 255).astype(np.uint8)
                denoised[i] = cv2.fastNlMeansDenoising(band_uint8, h=10)

        else:
            raise ValueError(f"Unknown denoising method: {method}")

        return denoised

    def normalize_bandwise(self, hypercube: np.ndarray,
                          brightness_boost: bool = True,
                          percentile_clip: bool = True) -> np.ndarray:
        """
        Band-wise normalization with brightness boost and percentile clipping
        """
        normalized = np.zeros_like(hypercube)

        for i in range(hypercube.shape[0]):
            band = hypercube[i].copy()

            # Step 1: Brightness boost (scale to full dynamic range)
            if brightness_boost:
                min_val = np.min(band)
                max_val = np.max(band)
                if max_val > min_val:
                    band = ((band - min_val) / (max_val - min_val)) * 255.0

            # Step 2: Percentile clipping (remove outliers)
            if percentile_clip:
                p1 = np.percentile(band, 1)
                p99 = np.percentile(band, 99)
                band = np.clip(band, p1, p99)

            # Step 3: Normalize to [0, 1]
            min_val = np.min(band)
            max_val = np.max(band)
            if max_val > min_val:
                band = (band - min_val) / (max_val - min_val)

            normalized[i] = band

        return normalized

    def process_with_steps(self, keep_percentage: float = 75.0,
                           denoise_method: str = 'median') -> Dict:
        """
        Complete preprocessing pipeline with intermediate results

        Returns dict with all intermediate steps for visualization
        """
        print(f"\n{'='*80}")
        print(f"PREPROCESSING PIPELINE: {self.dataset_path.name}")
        print(f"{'='*80}")

        # Step 1: Load raw data
        print("\n[Step 1/5] Loading raw hypercube...")
        raw_hypercube = self.load_hypercube()
        print(f"  ✓ Loaded {raw_hypercube.shape[0]} bands, shape: {raw_hypercube.shape}")

        # Step 2: Filter noisy bands
        print(f"\n[Step 2/5] Filtering noisy bands (keeping top {keep_percentage}%)...")
        clean_hypercube, clean_indices = self.filter_noisy_bands(raw_hypercube, keep_percentage)

        # Step 3: Denoise spatially
        print(f"\n[Step 3/5] Applying {denoise_method} denoising...")
        denoised_hypercube = self.denoise_spatial(clean_hypercube, method=denoise_method)
        print(f"  ✓ Applied {denoise_method} filter")

        # Step 4: Normalize
        print("\n[Step 4/5] Normalizing band-wise...")
        normalized_hypercube = self.normalize_bandwise(denoised_hypercube,
                                                       brightness_boost=True,
                                                       percentile_clip=True)
        print(f"  ✓ Normalized {normalized_hypercube.shape[0]} bands")

        # Step 5: Calculate quality improvement
        print("\n[Step 5/5] Calculating quality metrics...")
        raw_metrics = self.calculate_band_quality(raw_hypercube)
        final_metrics = self.calculate_band_quality(normalized_hypercube)

        raw_snr = np.mean([m['snr'] for m in raw_metrics])
        final_snr = np.mean([m['snr'] for m in final_metrics])

        print(f"  ✓ Average SNR improvement: {raw_snr:.2f} → {final_snr:.2f}")

        return {
            'dataset_name': self.dataset_path.name,
            'step1_raw': raw_hypercube,
            'step2_filtered': clean_hypercube,
            'step3_denoised': denoised_hypercube,
            'step4_normalized': normalized_hypercube,
            'clean_indices': clean_indices,
            'raw_metrics': raw_metrics,
            'final_metrics': final_metrics,
            'denoise_method': denoise_method
        }

    def visualize_steps(self, results: Dict, band_to_show: int = None,
                       output_path: str = None):
        """
        Visualize preprocessing steps side-by-side

        Args:
            results: Output from process_with_steps()
            band_to_show: Which band index to visualize (after filtering)
            output_path: Where to save the visualization
        """
        if band_to_show is None:
            band_to_show = len(results['clean_indices']) - 1  # Last band (noisiest)

        # Get the original band index
        orig_band_idx = results['clean_indices'][band_to_show]
        wavelength = self.wavelengths[orig_band_idx]

        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Preprocessing Pipeline - {results['dataset_name']}\n"
                    f"Band {orig_band_idx} ({wavelength:.1f} nm)",
                    fontsize=16, fontweight='bold')

        # Step 1: Raw
        ax = axes[0, 0]
        im1 = ax.imshow(results['step1_raw'][orig_band_idx], cmap='gray', vmin=0, vmax=255)
        ax.set_title('Step 1: Raw Band\n(Original 8-bit)', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

        # Step 2: After filtering (same as raw for this band)
        ax = axes[0, 1]
        im2 = ax.imshow(results['step2_filtered'][band_to_show], cmap='gray', vmin=0, vmax=255)
        ax.set_title('Step 2: After Band Filtering\n(Noisy bands removed)', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

        # Step 3: Denoised
        ax = axes[0, 2]
        im3 = ax.imshow(results['step3_denoised'][band_to_show], cmap='gray', vmin=0, vmax=255)
        ax.set_title('Step 3: Spatial Denoising\n(Noise smoothed)', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

        # Step 4: Normalized
        ax = axes[1, 0]
        im4 = ax.imshow(results['step4_normalized'][band_to_show], cmap='gray', vmin=0, vmax=1)
        ax.set_title('Step 4: Normalized\n(Band-wise [0,1])', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im4, ax=ax, fraction=0.046, pad=0.04)

        # Difference: Raw vs Denoised
        ax = axes[1, 1]
        diff_denoise = np.abs(results['step2_filtered'][band_to_show].astype(float) -
                              results['step3_denoised'][band_to_show].astype(float))
        im5 = ax.imshow(diff_denoise, cmap='hot', vmin=0, vmax=50)
        ax.set_title('Noise Removed by Denoising\n(Absolute difference)', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im5, ax=ax, fraction=0.046, pad=0.04)

        # Quality metrics comparison
        ax = axes[1, 2]
        ax.axis('off')

        # Get metrics for this band
        raw_metric = results['raw_metrics'][orig_band_idx]
        final_metric = results['final_metrics'][band_to_show]

        metrics_text = f"""
Quality Metrics:

Raw Band:
  SNR:        {raw_metric['snr']:.2f}
  Variance:   {raw_metric['variance']:.1f}
  Mean:       {raw_metric['mean']:.1f}
  Saturation: {raw_metric['saturation_pct']:.1f}%
  Darkness:   {raw_metric['darkness_pct']:.1f}%

Final (Normalized):
  SNR:        {final_metric['snr']:.2f}
  Variance:   {final_metric['variance']:.3f}
  Mean:       {final_metric['mean']:.3f}

Improvement:
  SNR:        {((final_metric['snr']/raw_metric['snr'])-1)*100:+.1f}%
        """

        ax.text(0.1, 0.5, metrics_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='center', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Visualization saved to: {output_path}")

        plt.show()

        return fig

    def compare_denoising_methods(self, band_idx: int = None,
                                 output_path: str = None):
        """Compare different denoising methods side-by-side"""

        print(f"\n{'='*80}")
        print(f"COMPARING DENOISING METHODS: {self.dataset_path.name}")
        print(f"{'='*80}")

        # Load and filter
        raw_hypercube = self.load_hypercube()
        clean_hypercube, clean_indices = self.filter_noisy_bands(raw_hypercube, keep_percentage=75)

        if band_idx is None:
            band_idx = len(clean_indices) - 1  # Last band (noisiest)

        orig_band_idx = clean_indices[band_idx]
        wavelength = self.wavelengths[orig_band_idx]
        band = clean_hypercube[band_idx]

        # Apply different methods
        methods = ['median', 'gaussian', 'bilateral', 'nlm']
        denoised_bands = {}

        for method in methods:
            print(f"  Testing {method} filter...")
            if method == 'median':
                denoised_bands[method] = median_filter(band, size=3)
            elif method == 'gaussian':
                denoised_bands[method] = gaussian_filter(band, sigma=1.0)
            elif method == 'bilateral':
                band_uint8 = np.clip(band, 0, 255).astype(np.uint8)
                denoised_bands[method] = cv2.bilateralFilter(band_uint8, d=5, sigmaColor=50, sigmaSpace=50)
            elif method == 'nlm':
                band_uint8 = np.clip(band, 0, 255).astype(np.uint8)
                denoised_bands[method] = cv2.fastNlMeansDenoising(band_uint8, h=10)

        # Visualize
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Denoising Methods Comparison - {self.dataset_path.name}\n"
                    f"Band {orig_band_idx} ({wavelength:.1f} nm)",
                    fontsize=16, fontweight='bold')

        # Original
        ax = axes[0, 0]
        im = ax.imshow(band, cmap='gray', vmin=0, vmax=255)
        ax.set_title('Original (Filtered)', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Denoised versions
        for idx, method in enumerate(methods):
            row = (idx + 1) // 3
            col = (idx + 1) % 3
            ax = axes[row, col]

            im = ax.imshow(denoised_bands[method], cmap='gray', vmin=0, vmax=255)

            # Calculate noise reduction
            noise = np.abs(band.astype(float) - denoised_bands[method].astype(float))
            noise_mean = np.mean(noise)

            ax.set_title(f'{method.capitalize()}\n(Noise removed: {noise_mean:.2f})',
                        fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide last subplot
        axes[1, 2].axis('off')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Comparison saved to: {output_path}")

        plt.show()

        return fig


def main():
    """Example usage - Focus on preprocessing only"""

    # Initialize preprocessor for training dataset
    preprocessor = NoiseRemovalPreprocessor('training_dataset')

    print(f"Dataset: training_dataset")
    print(f"Total bands: {preprocessor.n_bands}")
    print(f"Wavelength range: {preprocessor.wavelengths[0]:.1f} - {preprocessor.wavelengths[-1]:.1f} nm")

    # Run complete pipeline with median filter (standard method)
    results = preprocessor.process_with_steps(
        keep_percentage=75.0,
        denoise_method='median'  # Standard noise removal method
    )

    # Visualize steps - will show the LAST (noisiest) band
    print("\n" + "="*80)
    print("Generating visualization for the noisiest band (last band)...")
    print("="*80)

    preprocessor.visualize_steps(
        results=results,
        band_to_show=None,  # Auto-select last (noisiest) band
        output_path=f'preprocessing_steps_training.png'
    )

    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"✓ Visualization saved to: preprocessing_steps_training.png")
    print(f"✓ Showing band {results['clean_indices'][-1]} ({preprocessor.wavelengths[results['clean_indices'][-1]]:.1f} nm)")
    print(f"✓ This is the last (typically noisiest) band after filtering")
    print(f"\nNext steps:")
    print(f"  1. Open preprocessing_steps_training.png")
    print(f"  2. Check the 'Noise Removed' panel (bottom middle)")
    print(f"  3. Hot colors = high noise removed")
    print(f"  4. Compare raw vs normalized quality")
    print(f"\nOptional: Compare denoising methods")
    print(f"  python -c \"from preprocessing_pipeline import NoiseRemovalPreprocessor; p = NoiseRemovalPreprocessor('training_dataset'); p.compare_denoising_methods()\"


if __name__ == '__main__':
    main()
