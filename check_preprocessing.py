"""
Simple Preprocessing Checker - Focus on verifying preprocessing quality

This script shows you exactly what happens to your data during preprocessing:
1. Load raw bands
2. Filter noisy bands
3. Apply median denoising
4. Apply band-wise normalization
5. Show before/after comparison for the NOISIEST band

No prediction, no complex analysis - just preprocessing verification.
"""

import numpy as np
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from typing import List


def load_wavelengths(dataset_path: str) -> List[float]:
    """Load wavelengths from header.json"""
    with open(Path(dataset_path) / 'header.json', 'r') as f:
        header = json.load(f)
    return header['wavelength (nm)']


def load_all_bands(dataset_path: str, n_bands: int) -> np.ndarray:
    """Load all spectral bands"""
    bands = []
    for band_idx in range(1, n_bands + 1):
        img_path = Path(dataset_path) / f'ImagesStack{band_idx:03d}.png'
        if img_path.exists():
            img = np.array(Image.open(img_path).convert('L'), dtype=np.float32)
            bands.append(img)
    return np.stack(bands, axis=0)


def calculate_snr(band: np.ndarray) -> float:
    """Calculate Signal-to-Noise Ratio"""
    mean_val = np.mean(band)
    std_val = np.std(band)
    return mean_val / (std_val + 1e-8)


def filter_noisy_bands(hypercube: np.ndarray, keep_percentage: float = 75.0):
    """Filter out noisy bands based on SNR"""
    snr_values = [calculate_snr(hypercube[i]) for i in range(hypercube.shape[0])]
    percentile = 100 - keep_percentage
    snr_threshold = np.percentile(snr_values, percentile)

    clean_indices = [i for i, snr in enumerate(snr_values) if snr >= snr_threshold]
    return hypercube[clean_indices], clean_indices, snr_threshold


def denoise_median(hypercube: np.ndarray) -> np.ndarray:
    """Apply median filter to remove salt-and-pepper noise"""
    denoised = np.zeros_like(hypercube)
    for i in range(hypercube.shape[0]):
        denoised[i] = median_filter(hypercube[i], size=3)
    return denoised


def normalize_bandwise(hypercube: np.ndarray) -> np.ndarray:
    """Band-wise normalization with brightness boost and percentile clipping"""
    normalized = np.zeros_like(hypercube)

    for i in range(hypercube.shape[0]):
        band = hypercube[i].copy()

        # Step 1: Brightness boost
        min_val = np.min(band)
        max_val = np.max(band)
        if max_val > min_val:
            band = ((band - min_val) / (max_val - min_val)) * 255.0

        # Step 2: Percentile clipping
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


def visualize_preprocessing(dataset_path: str, output_path: str = 'preprocessing_check.png'):
    """
    Visualize preprocessing steps for the noisiest band
    """
    print("="*80)
    print("PREPROCESSING VERIFICATION")
    print("="*80)

    # Load data
    print("\n[1/5] Loading wavelengths...")
    wavelengths = load_wavelengths(dataset_path)
    n_bands = len(wavelengths)
    print(f"  ✓ Found {n_bands} wavelengths ({wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm)")

    print("\n[2/5] Loading all spectral bands...")
    raw_hypercube = load_all_bands(dataset_path, n_bands)
    print(f"  ✓ Loaded {raw_hypercube.shape[0]} bands, shape: {raw_hypercube.shape}")

    # Filter noisy bands
    print("\n[3/5] Filtering noisy bands (keeping top 75%)...")
    clean_hypercube, clean_indices, snr_threshold = filter_noisy_bands(raw_hypercube, 75.0)
    print(f"  ✓ Clean bands: {len(clean_indices)}/{n_bands} ({len(clean_indices)/n_bands*100:.1f}%)")
    print(f"  ✓ SNR threshold: {snr_threshold:.2f}")
    print(f"  ✓ Wavelength range: {wavelengths[clean_indices[0]]:.1f} - {wavelengths[clean_indices[-1]]:.1f} nm")

    # Select the last (noisiest) band for visualization
    band_idx = len(clean_indices) - 1  # Last band
    orig_band_idx = clean_indices[band_idx]
    wavelength = wavelengths[orig_band_idx]

    print(f"\n[4/5] Processing band {orig_band_idx} ({wavelength:.1f} nm) - LAST BAND (noisiest)...")

    # Get the specific band through each step
    raw_band = raw_hypercube[orig_band_idx]

    # After filtering (still same as raw for this band)
    filtered_band = clean_hypercube[band_idx]

    # After denoising
    denoised_band = median_filter(filtered_band, size=3)

    # Path 1: Normalize WITHOUT denoising
    normalized_without_denoise = filtered_band.copy()
    # Brightness boost
    min_val, max_val = np.min(normalized_without_denoise), np.max(normalized_without_denoise)
    if max_val > min_val:
        normalized_without_denoise = ((normalized_without_denoise - min_val) / (max_val - min_val)) * 255.0
    # Percentile clip
    p1, p99 = np.percentile(normalized_without_denoise, 1), np.percentile(normalized_without_denoise, 99)
    normalized_without_denoise = np.clip(normalized_without_denoise, p1, p99)
    # Normalize
    min_val, max_val = np.min(normalized_without_denoise), np.max(normalized_without_denoise)
    if max_val > min_val:
        normalized_without_denoise = (normalized_without_denoise - min_val) / (max_val - min_val)

    # Path 2: Normalize WITH denoising
    normalized_with_denoise = denoised_band.copy()
    # Brightness boost
    min_val, max_val = np.min(normalized_with_denoise), np.max(normalized_with_denoise)
    if max_val > min_val:
        normalized_with_denoise = ((normalized_with_denoise - min_val) / (max_val - min_val)) * 255.0
    # Percentile clip
    p1, p99 = np.percentile(normalized_with_denoise, 1), np.percentile(normalized_with_denoise, 99)
    normalized_with_denoise = np.clip(normalized_with_denoise, p1, p99)
    # Normalize
    min_val, max_val = np.min(normalized_with_denoise), np.max(normalized_with_denoise)
    if max_val > min_val:
        normalized_with_denoise = (normalized_with_denoise - min_val) / (max_val - min_val)

    # Calculate metrics
    raw_snr = calculate_snr(raw_band)
    denoised_snr = calculate_snr(denoised_band)
    normalized_without_denoise_snr = calculate_snr(normalized_without_denoise)
    normalized_with_denoise_snr = calculate_snr(normalized_with_denoise)

    print(f"  ✓ SNR: Raw={raw_snr:.2f}, Denoised={denoised_snr:.2f}")
    print(f"  ✓ SNR after normalization: Without denoise={normalized_without_denoise_snr:.2f}, With denoise={normalized_with_denoise_snr:.2f}")

    # Visualize
    print(f"\n[5/5] Creating visualization...")

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle(f"Preprocessing Verification - {Path(dataset_path).name}\n"
                f"Band {orig_band_idx} ({wavelength:.1f} nm) - Last band after filtering",
                fontsize=16, fontweight='bold')

    # 1. Raw
    ax = axes[0, 0]
    im1 = ax.imshow(raw_band, cmap='gray', vmin=0, vmax=255)
    ax.set_title(f'Step 1: Raw Band\nSNR: {raw_snr:.2f}', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    # 2. Filtered (same as raw for visualization)
    ax = axes[0, 1]
    im2 = ax.imshow(filtered_band, cmap='gray', vmin=0, vmax=255)
    ax.set_title(f'Step 2: After Band Filtering\n(Removed {n_bands - len(clean_indices)} noisy bands)',
                fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    # 3. Denoised
    ax = axes[0, 2]
    im3 = ax.imshow(denoised_band, cmap='gray', vmin=0, vmax=255)
    ax.set_title(f'Step 3: Median Denoising\nSNR: {denoised_snr:.2f}',
                fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

    # 4. Noise removed
    ax = axes[0, 3]
    noise = np.abs(filtered_band.astype(float) - denoised_band.astype(float))
    im4 = ax.imshow(noise, cmap='hot', vmin=0, vmax=50)
    ax.set_title(f'Noise Removed by Denoising\nMean: {np.mean(noise):.2f}',
                fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im4, ax=ax, fraction=0.046, pad=0.04)

    # 5. Normalized WITHOUT denoising
    ax = axes[1, 0]
    im5 = ax.imshow(normalized_without_denoise, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Normalized WITHOUT Denoise\nSNR: {normalized_without_denoise_snr:.2f}',
                fontsize=12, fontweight='bold', color='darkred')
    ax.axis('off')
    plt.colorbar(im5, ax=ax, fraction=0.046, pad=0.04)

    # 6. Normalized WITH denoising
    ax = axes[1, 1]
    im6 = ax.imshow(normalized_with_denoise, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Normalized WITH Denoise\nSNR: {normalized_with_denoise_snr:.2f}',
                fontsize=12, fontweight='bold', color='darkgreen')
    ax.axis('off')
    plt.colorbar(im6, ax=ax, fraction=0.046, pad=0.04)

    # 7. Difference between normalized versions
    ax = axes[1, 2]
    norm_diff = np.abs(normalized_without_denoise - normalized_with_denoise)
    im7 = ax.imshow(norm_diff, cmap='hot', vmin=0, vmax=0.2)
    ax.set_title(f'Difference After Normalization\nMean: {np.mean(norm_diff):.4f}',
                fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im7, ax=ax, fraction=0.046, pad=0.04)

    # 8. Summary statistics
    ax = axes[1, 3]
    ax.axis('off')

    comparison_text = f"""
PREPROCESSING SUMMARY

Dataset: {Path(dataset_path).name}
Total bands: {n_bands}
Clean bands: {len(clean_indices)} ({len(clean_indices)/n_bands*100:.1f}%)

Showing: Band {orig_band_idx} / {n_bands}
Wavelength: {wavelength:.1f} nm
(Last band after filtering)

Quality Metrics:
  Raw SNR:              {raw_snr:.2f}
  Denoised SNR:         {denoised_snr:.2f}

After Normalization:
  WITHOUT denoise SNR:  {normalized_without_denoise_snr:.2f}
  WITH denoise SNR:     {normalized_with_denoise_snr:.2f}

Noise Impact:
  Before normalize:     {np.mean(noise):.2f} (mean diff)
  After normalize:      {np.mean(norm_diff):.4f} (mean diff)
  Reduction:            {(1 - np.mean(norm_diff)/(np.mean(noise)/255))*100:.1f}%

COMPARISON:
  Panel 5 (RED):   Without denoising
  Panel 6 (GREEN): With denoising
  Panel 7 (DIFF):  Impact of denoising
                   after normalization

Hot colors in Panel 7 =
Denoising still matters after normalization

✓ Check if difference is significant
    """

    ax.text(0.1, 0.5, comparison_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='center', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved to: {output_path}")

    plt.show()

    # Final summary
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE ✓")
    print("="*80)
    print(f"Output: {output_path}")
    print(f"\n8-PANEL LAYOUT:")
    print(f"  Top row:    Raw → Filtered → Denoised → Noise Removed")
    print(f"  Bottom row: Normalized (NO denoise) → Normalized (WITH denoise) → Difference → Stats")
    print(f"\nKEY COMPARISON (Bottom row):")
    print(f"  • Panel 5 (RED title):   Normalized WITHOUT denoising")
    print(f"  • Panel 6 (GREEN title): Normalized WITH denoising")
    print(f"  • Panel 7 (DIFF):        Impact of denoising after normalization")
    print(f"\nWhat to check:")
    print(f"  1. Panel 4: Noise removed before normalization (mean: {np.mean(noise):.2f})")
    print(f"  2. Panel 7: Difference after normalization (mean: {np.mean(norm_diff):.4f})")
    print(f"  3. Compare Panel 5 vs Panel 6 visually")
    print(f"  4. Hot colors in Panel 7 = denoising still makes a difference")
    print(f"\nConclusion:")
    if np.mean(norm_diff) > 0.01:
        print(f"  ✓ Denoising has SIGNIFICANT impact even after normalization")
        print(f"  → Recommend using denoising in training pipeline")
    else:
        print(f"  • Denoising has MINIMAL impact after normalization")
        print(f"  → Normalization alone may be sufficient")
    print(f"\nThis band ({wavelength:.1f} nm) is the last one after filtering,")
    print(f"typically one of the noisier bands that still passed quality checks.")


if __name__ == '__main__':
    import sys

    # Allow specifying dataset as argument
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'training_dataset'

    visualize_preprocessing(dataset, f'preprocessing_check_{Path(dataset).name}.png')
