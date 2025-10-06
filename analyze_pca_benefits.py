"""
Quick PCA Analysis: Should You Use PCA?

This script analyzes your hyperspectral data to determine if PCA will help:
1. Spectral correlation analysis
2. Variance distribution
3. Noise estimation
4. Recommendations
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from PIL import Image
from scipy.stats import pearsonr
import seaborn as sns


def load_hypercube(dataset_path: str):
    """Load normalized hypercube."""
    dataset_path = Path(dataset_path)

    with open(dataset_path / 'header.json', 'r') as f:
        header = json.load(f)
    wavelengths = header['wavelength (nm)']

    bands = []
    for i in range(1, len(wavelengths) + 1):
        img_path = dataset_path / f'ImagesStack{i:03d}.png'
        if img_path.exists():
            img = np.array(Image.open(img_path).convert('L'), dtype=np.float32) / 255.0
            bands.append(img)

    hypercube = np.stack(bands, axis=0)
    return hypercube, wavelengths


def calculate_band_correlation_matrix(hypercube: np.ndarray):
    """Calculate correlation matrix between spectral bands."""
    n_bands, height, width = hypercube.shape

    # Reshape to (n_bands, n_pixels)
    bands_flat = hypercube.reshape(n_bands, -1)

    # Calculate correlation matrix
    print("Calculating correlation matrix...")
    corr_matrix = np.corrcoef(bands_flat)

    # Statistics
    # Get upper triangle (excluding diagonal)
    upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    mean_corr = np.mean(upper_triangle)
    median_corr = np.median(upper_triangle)
    high_corr_pct = np.sum(upper_triangle > 0.9) / len(upper_triangle) * 100

    print(f"\nBand Correlation Statistics:")
    print(f"  Mean correlation: {mean_corr:.3f}")
    print(f"  Median correlation: {median_corr:.3f}")
    print(f"  High correlation (>0.9): {high_corr_pct:.1f}%")

    if mean_corr > 0.8:
        print(f"  → HIGH correlation: PCA will be very effective! ✓")
    elif mean_corr > 0.6:
        print(f"  → MODERATE correlation: PCA may help ✓")
    else:
        print(f"  → LOW correlation: PCA may not help much")

    return corr_matrix, mean_corr, high_corr_pct


def estimate_noise_level(hypercube: np.ndarray):
    """Estimate noise level using local variance."""
    n_bands = hypercube.shape[0]

    # Sample random patches
    n_patches = 100
    patch_size = 5
    noise_estimates = []

    for _ in range(n_patches):
        # Random location
        h, w = hypercube.shape[1:]
        y = np.random.randint(0, h - patch_size)
        x = np.random.randint(0, w - patch_size)

        # Extract patch from random band
        band_idx = np.random.randint(0, n_bands)
        patch = hypercube[band_idx, y:y+patch_size, x:x+patch_size]

        # Local variance as noise estimate
        noise_estimates.append(np.var(patch))

    noise_level = np.median(noise_estimates)
    signal_level = np.median(np.var(hypercube, axis=(1, 2)))

    snr = signal_level / (noise_level + 1e-10)
    snr_db = 10 * np.log10(snr)

    print(f"\nNoise Estimation:")
    print(f"  Estimated noise level: {noise_level:.6f}")
    print(f"  Estimated signal level: {signal_level:.6f}")
    print(f"  SNR: {snr_db:.2f} dB")

    if snr_db < 20:
        print(f"  → NOISY data: PCA will help reduce noise! ✓")
    elif snr_db < 30:
        print(f"  → MODERATE noise: PCA may help ✓")
    else:
        print(f"  → LOW noise: PCA won't help much with noise")

    return noise_level, snr_db


def analyze_variance_distribution(hypercube: np.ndarray):
    """Analyze how variance is distributed across bands."""
    # Per-band variance
    band_variance = np.var(hypercube, axis=(1, 2))

    # Statistics
    max_var = np.max(band_variance)
    min_var = np.min(band_variance)
    mean_var = np.mean(band_variance)
    std_var = np.std(band_variance)

    # Ratio of max to min variance
    variance_ratio = max_var / (min_var + 1e-10)

    print(f"\nVariance Distribution:")
    print(f"  Max variance: {max_var:.6f}")
    print(f"  Min variance: {min_var:.6f}")
    print(f"  Mean variance: {mean_var:.6f}")
    print(f"  Std variance: {std_var:.6f}")
    print(f"  Max/Min ratio: {variance_ratio:.1f}x")

    if variance_ratio > 100:
        print(f"  → HIGH variance range: Some bands are much more informative ✓")
        print(f"  → PCA will effectively filter low-variance (noisy) bands ✓")
    elif variance_ratio > 10:
        print(f"  → MODERATE variance range: PCA may help ✓")
    else:
        print(f"  → UNIFORM variance: All bands contribute similarly")

    return band_variance, variance_ratio


def estimate_intrinsic_dimensionality(hypercube: np.ndarray):
    """Estimate intrinsic dimensionality using PCA variance analysis."""
    from sklearn.decomposition import PCA

    n_bands, h, w = hypercube.shape
    X = hypercube.reshape(n_bands, -1).T

    # Fit full PCA
    print("\nEstimating intrinsic dimensionality...")
    pca = PCA()
    pca.fit(X)

    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find components needed for different thresholds
    n_95 = np.argmax(cumsum_variance >= 0.95) + 1
    n_99 = np.argmax(cumsum_variance >= 0.99) + 1
    n_999 = np.argmax(cumsum_variance >= 0.999) + 1

    print(f"  Components for 95% variance: {n_95} ({n_95/n_bands*100:.1f}% of original)")
    print(f"  Components for 99% variance: {n_99} ({n_99/n_bands*100:.1f}% of original)")
    print(f"  Components for 99.9% variance: {n_999} ({n_999/n_bands*100:.1f}% of original)")

    reduction_95 = (1 - n_95/n_bands) * 100
    reduction_99 = (1 - n_99/n_bands) * 100

    if reduction_99 > 50:
        print(f"\n  → PCA can reduce dimensions by {reduction_99:.0f}% with minimal loss! ✓")
        print(f"  → Strongly recommended to use PCA ✓✓✓")
    elif reduction_99 > 30:
        print(f"\n  → PCA can reduce dimensions by {reduction_99:.0f}% ✓")
        print(f"  → Recommended to use PCA ✓✓")
    else:
        print(f"\n  → PCA reduction would be modest ({reduction_99:.0f}%)")
        print(f"  → May not be worth the complexity")

    return pca, n_95, n_99, n_999, cumsum_variance


def visualize_analysis(hypercube, wavelengths, corr_matrix, band_variance,
                       pca, cumsum_variance, output_path='pca_analysis.png'):
    """Create comprehensive visualization of PCA analysis."""
    fig = plt.figure(figsize=(20, 12))

    # 1. Correlation heatmap (sample for visualization)
    ax1 = plt.subplot(2, 3, 1)
    # Sample every 10th band for visualization clarity
    sample_indices = range(0, len(wavelengths), 10)
    corr_sample = corr_matrix[np.ix_(sample_indices, sample_indices)]
    im1 = ax1.imshow(corr_sample, cmap='RdYlGn', vmin=-1, vmax=1)
    ax1.set_title('Band Correlation Matrix\n(Sampled every 10th band)', fontweight='bold')
    ax1.set_xlabel('Band Index')
    ax1.set_ylabel('Band Index')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # 2. Correlation distribution
    ax2 = plt.subplot(2, 3, 2)
    upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    ax2.hist(upper_triangle, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(np.mean(upper_triangle), color='red', linestyle='--',
                label=f'Mean: {np.mean(upper_triangle):.2f}')
    ax2.set_xlabel('Correlation Coefficient')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Band Correlations', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Variance per band
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(wavelengths, band_variance, linewidth=1.5, color='darkgreen')
    ax3.set_xlabel('Wavelength (nm)')
    ax3.set_ylabel('Variance')
    ax3.set_title('Variance per Spectral Band', fontweight='bold')
    ax3.grid(alpha=0.3)

    # 4. PCA variance explained (individual)
    ax4 = plt.subplot(2, 3, 4)
    n_show = min(50, len(pca.explained_variance_ratio_))
    ax4.bar(range(1, n_show + 1), pca.explained_variance_ratio_[:n_show] * 100,
            alpha=0.7, color='steelblue')
    ax4.set_xlabel('Principal Component')
    ax4.set_ylabel('Variance Explained (%)')
    ax4.set_title('Variance Explained by Each PC\n(First 50 components)', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    # 5. Cumulative variance
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(range(1, len(cumsum_variance) + 1), cumsum_variance * 100,
            linewidth=2, color='darkblue')
    ax5.axhline(95, color='red', linestyle='--', label='95% threshold')
    ax5.axhline(99, color='orange', linestyle='--', label='99% threshold')
    ax5.axhline(99.9, color='green', linestyle='--', label='99.9% threshold')
    ax5.set_xlabel('Number of Components')
    ax5.set_ylabel('Cumulative Variance Explained (%)')
    ax5.set_title('Cumulative Variance Explained', fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)
    ax5.set_xlim(0, min(300, len(cumsum_variance)))

    # 6. Recommendation summary (text)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # Calculate recommendations
    mean_corr = np.mean(upper_triangle)
    n_99 = np.argmax(cumsum_variance >= 0.99) + 1
    reduction_99 = (1 - n_99/len(wavelengths)) * 100

    recommendation = "PCA RECOMMENDATION\n\n"

    if mean_corr > 0.7 and reduction_99 > 40:
        recommendation += "✓✓✓ HIGHLY RECOMMENDED\n\n"
        recommendation += f"Reasons:\n"
        recommendation += f"• High band correlation ({mean_corr:.2f})\n"
        recommendation += f"• Large dimensionality reduction ({reduction_99:.0f}%)\n"
        recommendation += f"• Expected noise reduction\n"
        recommendation += f"• Faster training & inference\n\n"
        recommendation += f"Suggested config:\n"
        recommendation += f"• Start with {n_99} components (99% var.)\n"
        recommendation += f"• Try {n_99//2}, {n_99}, {int(n_99*1.5)} and compare\n"
    elif mean_corr > 0.5 or reduction_99 > 30:
        recommendation += "✓✓ RECOMMENDED\n\n"
        recommendation += f"Reasons:\n"
        recommendation += f"• Moderate correlation ({mean_corr:.2f})\n"
        recommendation += f"• Decent reduction ({reduction_99:.0f}%)\n\n"
        recommendation += f"Suggested config:\n"
        recommendation += f"• Try {n_99} components (99% var.)\n"
        recommendation += f"• Compare with baseline (no PCA)\n"
    else:
        recommendation += "? OPTIONAL\n\n"
        recommendation += f"PCA may not provide significant benefits:\n"
        recommendation += f"• Lower correlation ({mean_corr:.2f})\n"
        recommendation += f"• Modest reduction ({reduction_99:.0f}%)\n\n"
        recommendation += f"Suggestion:\n"
        recommendation += f"• Train baseline first\n"
        recommendation += f"• Try PCA only if overfitting occurs\n"

    ax6.text(0.1, 0.5, recommendation, transform=ax6.transAxes,
            fontsize=11, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle('PCA Analysis for Hyperspectral Band Selection',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Analysis visualization saved to: {output_path}")

    plt.show()


def main():
    """Run complete PCA analysis."""
    print("\n" + "="*80)
    print("PCA BENEFIT ANALYSIS FOR HYPERSPECTRAL DATA")
    print("="*80)

    # Load data
    print("\n[1/6] Loading hypercube...")
    hypercube, wavelengths = load_hypercube('training_dataset')
    print(f"✓ Loaded: {hypercube.shape} (bands, height, width)")
    print(f"✓ Wavelengths: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")

    # Correlation analysis
    print(f"\n[2/6] Analyzing band correlations...")
    corr_matrix, mean_corr, high_corr_pct = calculate_band_correlation_matrix(hypercube)

    # Noise estimation
    print(f"\n[3/6] Estimating noise level...")
    noise_level, snr_db = estimate_noise_level(hypercube)

    # Variance distribution
    print(f"\n[4/6] Analyzing variance distribution...")
    band_variance, variance_ratio = analyze_variance_distribution(hypercube)

    # Intrinsic dimensionality
    print(f"\n[5/6] Estimating intrinsic dimensionality...")
    pca, n_95, n_99, n_999, cumsum_variance = estimate_intrinsic_dimensionality(hypercube)

    # Visualization
    print(f"\n[6/6] Creating visualization...")
    visualize_analysis(hypercube, wavelengths, corr_matrix, band_variance,
                      pca, cumsum_variance)

    # Final recommendation
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)

    reduction_99 = (1 - n_99/len(wavelengths)) * 100

    if mean_corr > 0.7 and reduction_99 > 40:
        print("\n✓✓✓ PCA IS HIGHLY RECOMMENDED\n")
        print("Benefits you will get:")
        print(f"  1. Dimension reduction: {len(wavelengths)} → {n_99} bands ({reduction_99:.0f}% reduction)")
        print(f"  2. Noise reduction: High band correlation indicates redundant noise")
        print(f"  3. Faster training: ~{reduction_99:.0f}% fewer features to process")
        print(f"  4. Better generalization: Reduced overfitting risk")
        print(f"  5. Memory savings: ~{reduction_99:.0f}% less memory usage")
        print(f"\nRecommended approach:")
        print(f"  python train_with_pca.py --compare --compare-configs {n_99//2} {n_99} {int(n_99*1.5)}")
        print(f"\nThis will compare:")
        print(f"  • Baseline (no PCA)")
        print(f"  • PCA with {n_99//2} components (conservative)")
        print(f"  • PCA with {n_99} components (99% variance)")
        print(f"  • PCA with {int(n_99*1.5)} components (safe)")

    elif mean_corr > 0.5 or reduction_99 > 30:
        print("\n✓✓ PCA IS RECOMMENDED\n")
        print("Expected benefits:")
        print(f"  1. Dimension reduction: {len(wavelengths)} → {n_99} bands ({reduction_99:.0f}% reduction)")
        print(f"  2. Some noise reduction")
        print(f"  3. Faster training")
        print(f"\nRecommended approach:")
        print(f"  python train_with_pca.py --compare --compare-configs {n_99}")
        print(f"\nCompare baseline vs PCA-{n_99} to see actual benefit")

    else:
        print("\n? PCA IS OPTIONAL\n")
        print("Analysis shows:")
        print(f"  • Low band correlation ({mean_corr:.2f})")
        print(f"  • Modest reduction potential ({reduction_99:.0f}%)")
        print(f"\nRecommended approach:")
        print(f"  1. Train baseline model first (no PCA)")
        print(f"  2. If you see overfitting, try PCA with {n_99} components")
        print(f"\nCommand:")
        print(f"  python train_with_pca.py  # baseline")
        print(f"  python train_with_pca.py --pca-components {n_99}  # if needed")

    print("\n" + "="*80)
    print("\nGenerated file:")
    print("  • pca_analysis.png - Complete PCA benefit analysis")
    print("\nNext steps:")
    print("  1. Review pca_analysis.png")
    print("  2. Run suggested training command")
    print("  3. Compare results in pca_training_comparison.json")
    print("="*80)


if __name__ == '__main__':
    main()
