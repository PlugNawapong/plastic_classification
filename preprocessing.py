"""
Advanced preprocessing techniques for hyperspectral imaging.

Includes:
1. Paper's simplified normalization (baseline)
2. Brightness enhancement + percentile normalization (improved)
3. Band-wise normalization
4. PCA dimensionality reduction (optional)
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple, Optional


class HyperspectralPreprocessor:
    """
    Preprocessing pipeline for hyperspectral data.

    Supports multiple normalization strategies to enhance classification performance.
    """

    def __init__(
        self,
        method='percentile',
        brightness_boost=True,
        percentile_low=1,
        percentile_high=99,
        band_wise=True,
        pca_components=None
    ):
        """
        Args:
            method: 'simple' (paper's method) or 'percentile' (improved)
            brightness_boost: Whether to boost brightness to max before normalization
            percentile_low: Lower percentile for clipping (default: 1%)
            percentile_high: Upper percentile for clipping (default: 99%)
            band_wise: Whether to normalize each band independently
            pca_components: Number of PCA components (None = no PCA)
        """
        self.method = method
        self.brightness_boost = brightness_boost
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.band_wise = band_wise
        self.pca_components = pca_components
        self.pca_model = None

    def normalize_simple(self, spectral_cube: np.ndarray) -> np.ndarray:
        """
        Simple normalization (from paper).

        Normalize by global maximum value across all bands.

        Args:
            spectral_cube: Array of shape (n_bands, height, width)

        Returns:
            Normalized cube in [0, 1]
        """
        max_val = spectral_cube.max()
        if max_val > 0:
            return spectral_cube / max_val
        return spectral_cube

    def normalize_percentile(
        self,
        spectral_cube: np.ndarray,
        band_wise: bool = True
    ) -> np.ndarray:
        """
        Percentile-based normalization (improved method).

        1. Clip outliers using percentiles
        2. Normalize to [0, 1] range
        3. Can be applied per-band or globally

        Args:
            spectral_cube: Array of shape (n_bands, height, width)
            band_wise: If True, normalize each band independently

        Returns:
            Normalized cube in [0, 1]
        """
        normalized = spectral_cube.copy()

        if band_wise:
            # Normalize each spectral band independently
            for i in range(spectral_cube.shape[0]):
                band = spectral_cube[i]

                # Calculate percentiles for this band
                p_low = np.percentile(band, self.percentile_low)
                p_high = np.percentile(band, self.percentile_high)

                # Clip and normalize
                band_clipped = np.clip(band, p_low, p_high)

                if p_high > p_low:
                    normalized[i] = (band_clipped - p_low) / (p_high - p_low)
                else:
                    normalized[i] = band_clipped
        else:
            # Global normalization across all bands
            p_low = np.percentile(spectral_cube, self.percentile_low)
            p_high = np.percentile(spectral_cube, self.percentile_high)

            clipped = np.clip(spectral_cube, p_low, p_high)

            if p_high > p_low:
                normalized = (clipped - p_low) / (p_high - p_low)
            else:
                normalized = clipped

        return normalized

    def boost_brightness(self, spectral_cube: np.ndarray) -> np.ndarray:
        """
        Boost brightness by scaling to maximum possible value.

        Enhances signal before normalization, improving contrast.

        Args:
            spectral_cube: Array of shape (n_bands, height, width)

        Returns:
            Brightness-boosted cube
        """
        # Find current max value
        current_max = spectral_cube.max()

        # Assuming 16-bit data (0-65535) or 8-bit (0-255)
        # Scale to utilize full dynamic range
        if current_max < 256:
            target_max = 255.0
        elif current_max < 4096:
            target_max = 4095.0
        else:
            target_max = 65535.0

        if current_max > 0:
            boosted = spectral_cube * (target_max / current_max)
        else:
            boosted = spectral_cube

        return boosted

    def apply_pca(
        self,
        spectral_cube: np.ndarray,
        fit: bool = True
    ) -> np.ndarray:
        """
        Apply PCA for dimensionality reduction.

        PCA can reduce spectral bands while preserving most information.
        Useful when computational efficiency is critical.

        Args:
            spectral_cube: Array of shape (n_bands, height, width)
            fit: Whether to fit PCA model (True for training, False for inference)

        Returns:
            Reduced cube of shape (n_components, height, width)
        """
        if self.pca_components is None:
            return spectral_cube

        n_bands, height, width = spectral_cube.shape

        # Reshape to (n_pixels, n_bands)
        pixels = spectral_cube.reshape(n_bands, -1).T

        if fit:
            # Fit PCA model
            self.pca_model = PCA(n_components=self.pca_components)
            reduced_pixels = self.pca_model.fit_transform(pixels)

            variance_explained = self.pca_model.explained_variance_ratio_.sum()
            print(f"PCA: {n_bands} → {self.pca_components} bands")
            print(f"Variance explained: {variance_explained*100:.2f}%")
        else:
            # Use existing PCA model
            if self.pca_model is None:
                raise ValueError("PCA model not fitted. Call with fit=True first.")
            reduced_pixels = self.pca_model.transform(pixels)

        # Reshape back to (n_components, height, width)
        reduced_cube = reduced_pixels.T.reshape(self.pca_components, height, width)

        return reduced_cube

    def preprocess(
        self,
        spectral_cube: np.ndarray,
        fit_pca: bool = True
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline.

        Pipeline:
        1. Brightness boost (optional)
        2. Normalization (simple or percentile)
        3. PCA reduction (optional)

        Args:
            spectral_cube: Raw spectral cube (n_bands, height, width)
            fit_pca: Whether to fit PCA (True for training, False for inference)

        Returns:
            Preprocessed spectral cube
        """
        processed = spectral_cube.copy()

        # Step 1: Brightness boost
        if self.brightness_boost:
            processed = self.boost_brightness(processed)
            print("✓ Brightness boosted")

        # Step 2: Normalization
        if self.method == 'simple':
            processed = self.normalize_simple(processed)
            print("✓ Simple normalization applied")
        elif self.method == 'percentile':
            processed = self.normalize_percentile(processed, band_wise=self.band_wise)
            mode = "band-wise" if self.band_wise else "global"
            print(f"✓ Percentile normalization applied ({mode})")

        # Step 3: PCA (optional)
        if self.pca_components is not None:
            processed = self.apply_pca(processed, fit=fit_pca)
            print(f"✓ PCA applied: {spectral_cube.shape[0]} → {processed.shape[0]} bands")

        return processed


def compare_preprocessing_methods(spectral_cube: np.ndarray) -> dict:
    """
    Compare different preprocessing methods on a sample cube.

    Args:
        spectral_cube: Sample spectral cube

    Returns:
        Dictionary with preprocessed cubes from different methods
    """
    results = {}

    # Method 1: Paper's simple normalization
    preprocessor1 = HyperspectralPreprocessor(
        method='simple',
        brightness_boost=False
    )
    results['simple'] = preprocessor1.preprocess(spectral_cube.copy())

    # Method 2: Percentile normalization (global)
    preprocessor2 = HyperspectralPreprocessor(
        method='percentile',
        brightness_boost=True,
        band_wise=False
    )
    results['percentile_global'] = preprocessor2.preprocess(spectral_cube.copy())

    # Method 3: Percentile normalization (band-wise) - YOUR SUGGESTION
    preprocessor3 = HyperspectralPreprocessor(
        method='percentile',
        brightness_boost=True,
        band_wise=True
    )
    results['percentile_bandwise'] = preprocessor3.preprocess(spectral_cube.copy())

    # Method 4: With PCA reduction
    preprocessor4 = HyperspectralPreprocessor(
        method='percentile',
        brightness_boost=True,
        band_wise=True,
        pca_components=50  # Reduce 459 → 50 bands
    )
    results['percentile_bandwise_pca'] = preprocessor4.preprocess(spectral_cube.copy())

    return results


# =============================================================================
# PAPER'S POSTPROCESSING
# =============================================================================

def apply_postprocessing(
    prediction_map: np.ndarray,
    median_kernel: int = 5,
    morph_kernel: int = 3
) -> np.ndarray:
    """
    Post-processing from the paper.

    Pipeline:
    1. Median filter (size 5) - removes salt-and-pepper noise
    2. Morphological opening - removes small isolated regions
    3. Morphological closing - fills small gaps

    This reduces classification errors at object boundaries.

    Args:
        prediction_map: Class prediction map (H, W)
        median_kernel: Kernel size for median filtering
        morph_kernel: Kernel size for morphological operations

    Returns:
        Cleaned prediction map
    """
    from scipy.ndimage import median_filter, binary_opening, binary_closing

    # Step 1: Median filter
    processed = median_filter(prediction_map, size=median_kernel)

    # Step 2 & 3: Morphological operations per class
    unique_classes = np.unique(processed)

    for class_id in unique_classes:
        if class_id == 0:  # Skip background
            continue

        # Binary mask for this class
        mask = (processed == class_id).astype(np.uint8)

        # Opening: erosion → dilation (removes small objects)
        mask = binary_opening(mask, structure=np.ones((morph_kernel, morph_kernel)))

        # Closing: dilation → erosion (fills small holes)
        mask = binary_closing(mask, structure=np.ones((morph_kernel, morph_kernel)))

        # Update map
        processed[mask.astype(bool)] = class_id

    return processed


# =============================================================================
# IS PCA NECESSARY?
# =============================================================================

def is_pca_necessary(n_bands: int, dataset_size: int, model_params: int) -> dict:
    """
    Analyze whether PCA is necessary for your use case.

    PCA Pros:
    - Reduces computational cost (fewer bands = faster training/inference)
    - Reduces memory usage
    - Can improve generalization (removes noise)
    - Typical: 90-95% variance retained with 50-100 components vs 459

    PCA Cons:
    - Loss of spectral information (even if small)
    - Extra preprocessing step (complexity)
    - Requires fitting on training data
    - May reduce peak accuracy

    Args:
        n_bands: Number of spectral bands (459 in your case)
        dataset_size: Number of training samples
        model_params: Model parameter count

    Returns:
        Recommendation dictionary
    """
    recommendation = {
        'necessary': False,
        'recommended': False,
        'reasons': [],
        'alternatives': []
    }

    # Analysis
    if n_bands > 200:
        recommendation['reasons'].append(
            f"High dimensionality ({n_bands} bands) → PCA could reduce by 5-10x"
        )
        recommendation['recommended'] = True

    if dataset_size < n_bands * 100:
        recommendation['reasons'].append(
            f"Limited data ({dataset_size} samples, {n_bands} features) → Risk of overfitting"
        )
        recommendation['recommended'] = True

    if model_params > 1000000:
        recommendation['reasons'].append(
            f"Large model ({model_params:,} params) → PCA could speed up training"
        )

    # Alternatives
    recommendation['alternatives'] = [
        "1. Try training WITHOUT PCA first (baseline)",
        "2. If accuracy is good (>95%), PCA not needed",
        "3. If training is slow, try PCA with 50-100 components",
        "4. Compare: Full bands vs PCA - pick best accuracy/speed tradeoff"
    ]

    return recommendation


if __name__ == '__main__':
    """Example usage and comparison."""

    # Simulate a small spectral cube
    np.random.seed(42)
    spectral_cube = np.random.rand(459, 100, 100) * 4095  # 16-bit data

    print("="*70)
    print("PREPROCESSING COMPARISON")
    print("="*70)

    # Compare methods
    results = compare_preprocessing_methods(spectral_cube)

    print("\nResults:")
    for method, cube in results.items():
        print(f"\n{method}:")
        print(f"  Shape: {cube.shape}")
        print(f"  Range: [{cube.min():.4f}, {cube.max():.4f}]")
        print(f"  Mean: {cube.mean():.4f}")
        print(f"  Std: {cube.std():.4f}")

    # PCA analysis
    print("\n" + "="*70)
    print("PCA NECESSITY ANALYSIS")
    print("="*70)

    analysis = is_pca_necessary(
        n_bands=459,
        dataset_size=100000,  # Approximate pixel count
        model_params=1000000
    )

    print(f"\nNecessary: {analysis['necessary']}")
    print(f"Recommended: {analysis['recommended']}")
    print("\nReasons:")
    for reason in analysis['reasons']:
        print(f"  • {reason}")

    print("\nRecommended approach:")
    for alt in analysis['alternatives']:
        print(f"  {alt}")
