"""
Export spectral bands with wavelength information and noise filtering.

This script:
1. Calculates wavelength for each band (900-1700nm, 224 or 459 bands)
2. Detects and filters noisy bands using SNR analysis
3. Exports only clean bands with wavelength in filename
4. Generates detailed quality report
"""

import os
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
from preprocessing import HyperspectralPreprocessor


def calculate_wavelength(band_index, n_total_bands, wl_min=900, wl_max=1700):
    """
    Calculate wavelength for a given band index.

    Args:
        band_index: Band number (0-based)
        n_total_bands: Total number of spectral bands
        wl_min: Minimum wavelength (nm)
        wl_max: Maximum wavelength (nm)

    Returns:
        Wavelength in nanometers
    """
    # Linear interpolation across wavelength range
    wavelength = wl_min + (wl_max - wl_min) * (band_index / (n_total_bands - 1))
    return round(wavelength, 1)


def calculate_snr(band):
    """
    Calculate Signal-to-Noise Ratio (SNR) for a spectral band.

    Args:
        band: 2D array representing spectral band

    Returns:
        SNR value (higher is better)
    """
    # Signal: mean intensity
    signal = np.mean(band)

    # Noise: standard deviation of pixel values
    noise = np.std(band)

    # Avoid division by zero
    if noise < 1e-6:
        return 0.0

    snr = signal / noise
    return snr


def detect_noisy_bands(spectral_cube, snr_threshold=5.0, variance_threshold=10.0):
    """
    Detect noisy bands using multiple criteria.

    Args:
        spectral_cube: 3D array (n_bands, height, width)
        snr_threshold: Minimum SNR to be considered clean
        variance_threshold: Minimum variance (too low = dead band)

    Returns:
        Dictionary with quality metrics for each band
    """
    n_bands, height, width = spectral_cube.shape

    quality_metrics = []

    print("\nAnalyzing band quality...")
    for i in tqdm(range(n_bands), desc="Analyzing"):
        band = spectral_cube[i]

        # Calculate metrics
        mean_val = np.mean(band)
        std_val = np.std(band)
        variance = np.var(band)
        snr = calculate_snr(band)

        # Calculate pixel saturation
        max_possible = 65535 if band.max() > 4095 else 4095
        saturated_pixels = np.sum(band >= max_possible * 0.99)
        saturation_pct = 100 * saturated_pixels / (height * width)

        # Determine if band is clean
        is_clean = True
        issues = []

        if snr < snr_threshold:
            is_clean = False
            issues.append(f"Low SNR ({snr:.2f})")

        if variance < variance_threshold:
            is_clean = False
            issues.append(f"Low variance ({variance:.2f})")

        if saturation_pct > 10:  # More than 10% saturated
            is_clean = False
            issues.append(f"High saturation ({saturation_pct:.1f}%)")

        if mean_val < 10:  # Nearly all zeros
            is_clean = False
            issues.append("Near-zero signal")

        quality_metrics.append({
            'band_index': i,
            'mean': mean_val,
            'std': std_val,
            'variance': variance,
            'snr': snr,
            'saturation_pct': saturation_pct,
            'is_clean': is_clean,
            'issues': issues
        })

    return quality_metrics


def export_clean_bands(
    input_dir='training_dataset',
    output_dir='clean_normalized_bands',
    method='percentile',
    brightness_boost=True,
    band_wise=True,
    snr_threshold=5.0,
    variance_threshold=10.0,
    wl_min=900,
    wl_max=1700
):
    """
    Export only clean (non-noisy) normalized bands with wavelength names.

    Args:
        input_dir: Directory containing original spectral bands
        output_dir: Directory to save clean bands
        method: Normalization method
        brightness_boost: Apply brightness boost
        band_wise: Band-wise normalization
        snr_threshold: Minimum SNR for clean bands
        variance_threshold: Minimum variance
        wl_min: Minimum wavelength (nm)
        wl_max: Maximum wavelength (nm)
    """

    print("\n" + "="*90)
    print("EXPORTING CLEAN NORMALIZED BANDS WITH WAVELENGTH INFORMATION")
    print("="*90)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load spectral bands
    print(f"\n1. Loading spectral bands from: {input_dir}")
    image_files = sorted(glob.glob(os.path.join(input_dir, 'ImagesStack*.png')))

    if len(image_files) == 0:
        raise FileNotFoundError(f"No ImagesStack*.png files found in {input_dir}")

    n_bands = len(image_files)
    print(f"   Found {n_bands} spectral bands")

    # Load first image for dimensions
    first_img = np.array(Image.open(image_files[0]))
    height, width = first_img.shape
    print(f"   Image dimensions: {height} x {width}")
    print(f"   Wavelength range: {wl_min}-{wl_max} nm")

    # Pre-allocate array
    spectral_cube = np.zeros((n_bands, height, width), dtype=np.float32)

    # Load all bands
    print(f"\n2. Loading all {n_bands} bands...")
    for i, img_path in enumerate(tqdm(image_files, desc="   Loading")):
        img = np.array(Image.open(img_path))
        spectral_cube[i] = img.astype(np.float32)

    # Detect noisy bands
    print(f"\n3. Detecting noisy bands (SNR threshold: {snr_threshold}, Variance threshold: {variance_threshold})...")
    quality_metrics = detect_noisy_bands(spectral_cube, snr_threshold, variance_threshold)

    clean_bands = [m for m in quality_metrics if m['is_clean']]
    noisy_bands = [m for m in quality_metrics if not m['is_clean']]

    print(f"\n   Analysis complete:")
    print(f"   ✓ Clean bands: {len(clean_bands)}/{n_bands} ({100*len(clean_bands)/n_bands:.1f}%)")
    print(f"   ✗ Noisy bands: {len(noisy_bands)}/{n_bands} ({100*len(noisy_bands)/n_bands:.1f}%)")

    if len(noisy_bands) > 0:
        print(f"\n   Top 10 noisiest bands:")
        sorted_noisy = sorted(noisy_bands, key=lambda x: x['snr'])[:10]
        for m in sorted_noisy:
            wl = calculate_wavelength(m['band_index'], n_bands, wl_min, wl_max)
            issues_str = ", ".join(m['issues'])
            print(f"     Band {m['band_index']+1:3d} ({wl:.1f}nm): {issues_str}")

    # Apply normalization to ALL bands (including noisy, for comparison)
    print(f"\n4. Applying band-wise normalization...")
    preprocessor = HyperspectralPreprocessor(
        method=method,
        brightness_boost=brightness_boost,
        band_wise=band_wise,
        percentile_low=1,
        percentile_high=99
    )
    normalized_cube = preprocessor.preprocess(spectral_cube, fit_pca=False)

    # Export only CLEAN bands
    print(f"\n5. Exporting {len(clean_bands)} clean bands to: {output_dir}")

    exported_files = []
    clean_export_count = 0

    for metric in tqdm(quality_metrics, desc="   Exporting"):
        i = metric['band_index']
        wavelength = calculate_wavelength(i, n_bands, wl_min, wl_max)

        # Create filename with wavelength
        # Format: Band_001_900.0nm_normalized.png
        filename = f"Band_{i+1:03d}_{wavelength:.1f}nm_normalized.png"

        if metric['is_clean']:
            # Export clean band
            output_path = os.path.join(output_dir, filename)
            normalized_band = normalized_cube[i]
            normalized_8bit = (normalized_band * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(normalized_8bit).save(output_path)

            exported_files.append({
                'band_index': i + 1,
                'wavelength': wavelength,
                'filename': filename,
                'snr': metric['snr'],
                'variance': metric['variance'],
                'is_clean': True
            })
            clean_export_count += 1
        else:
            # Mark as skipped (noisy)
            exported_files.append({
                'band_index': i + 1,
                'wavelength': wavelength,
                'filename': filename,
                'snr': metric['snr'],
                'variance': metric['variance'],
                'is_clean': False,
                'issues': metric['issues']
            })

    print(f"\n   ✓ Exported {clean_export_count} clean bands")
    print(f"   ✗ Skipped {len(noisy_bands)} noisy bands")

    # Generate reports
    print(f"\n6. Generating quality reports...")
    generate_quality_report(quality_metrics, exported_files, output_dir, n_bands, wl_min, wl_max)
    generate_band_list(exported_files, output_dir)

    # Generate comparison visualization
    print(f"\n7. Generating visualizations...")
    visualize_clean_vs_noisy(spectral_cube, normalized_cube, quality_metrics,
                            output_dir, n_bands, wl_min, wl_max)

    print(f"\n{'='*90}")
    print("EXPORT COMPLETE!")
    print(f"{'='*90}")
    print(f"\nClean bands exported to: {output_dir}/")
    print(f"Quality report: {output_dir}/quality_report.txt")
    print(f"Clean band list: {output_dir}/clean_bands_list.txt")
    print(f"Comparison visualization: {output_dir}/clean_vs_noisy_comparison.png")
    print(f"\nTotal clean bands: {clean_export_count}/{n_bands}")


def generate_quality_report(quality_metrics, exported_files, output_dir, n_bands, wl_min, wl_max):
    """Generate detailed quality report."""

    report_path = os.path.join(output_dir, 'quality_report.txt')

    with open(report_path, 'w') as f:
        f.write("="*90 + "\n")
        f.write("SPECTRAL BAND QUALITY REPORT\n")
        f.write("="*90 + "\n\n")

        f.write("CONFIGURATION:\n")
        f.write(f"  Total bands: {n_bands}\n")
        f.write(f"  Wavelength range: {wl_min}-{wl_max} nm\n")
        f.write(f"  Wavelength step: {(wl_max-wl_min)/(n_bands-1):.2f} nm\n\n")

        clean_count = sum(1 for m in quality_metrics if m['is_clean'])
        f.write(f"SUMMARY:\n")
        f.write(f"  Clean bands: {clean_count}/{n_bands} ({100*clean_count/n_bands:.1f}%)\n")
        f.write(f"  Noisy bands: {n_bands-clean_count}/{n_bands} ({100*(n_bands-clean_count)/n_bands:.1f}%)\n\n")

        f.write("="*90 + "\n")
        f.write("CLEAN BANDS (EXPORTED)\n")
        f.write("="*90 + "\n\n")

        f.write(f"{'Band':<6} {'Wavelength':<12} {'SNR':<10} {'Variance':<12} {'Filename':<40}\n")
        f.write("-"*90 + "\n")

        for item in exported_files:
            if item['is_clean']:
                f.write(f"{item['band_index']:<6} {item['wavelength']:.1f} nm{'':<4} "
                       f"{item['snr']:<10.2f} {item['variance']:<12.2f} {item['filename']:<40}\n")

        f.write("\n" + "="*90 + "\n")
        f.write("NOISY BANDS (SKIPPED)\n")
        f.write("="*90 + "\n\n")

        f.write(f"{'Band':<6} {'Wavelength':<12} {'SNR':<10} {'Variance':<12} {'Issues':<40}\n")
        f.write("-"*90 + "\n")

        for item in exported_files:
            if not item['is_clean']:
                issues_str = ", ".join(item['issues'][:2])  # First 2 issues
                f.write(f"{item['band_index']:<6} {item['wavelength']:.1f} nm{'':<4} "
                       f"{item['snr']:<10.2f} {item['variance']:<12.2f} {issues_str:<40}\n")

        f.write("\n" + "="*90 + "\n")
        f.write("QUALITY STATISTICS\n")
        f.write("="*90 + "\n\n")

        snr_values = [m['snr'] for m in quality_metrics if m['is_clean']]
        if snr_values:
            f.write(f"Clean bands SNR statistics:\n")
            f.write(f"  Mean: {np.mean(snr_values):.2f}\n")
            f.write(f"  Median: {np.median(snr_values):.2f}\n")
            f.write(f"  Min: {np.min(snr_values):.2f}\n")
            f.write(f"  Max: {np.max(snr_values):.2f}\n\n")

        f.write("RECOMMENDATION:\n")
        f.write(f"  Use only the {clean_count} clean bands for training.\n")
        f.write(f"  Update model input to accept {clean_count} spectral bands instead of {n_bands}.\n")
        f.write(f"  See clean_bands_list.txt for list of clean band indices.\n")

    print(f"   ✓ Quality report saved: {report_path}")


def generate_band_list(exported_files, output_dir):
    """Generate clean band index list for easy use."""

    list_path = os.path.join(output_dir, 'clean_bands_list.txt')

    clean_indices = [item['band_index'] for item in exported_files if item['is_clean']]
    clean_wavelengths = [item['wavelength'] for item in exported_files if item['is_clean']]

    with open(list_path, 'w') as f:
        f.write("# Clean Band Indices (1-based)\n")
        f.write("# Format: Band_Index, Wavelength (nm)\n\n")

        for idx, wl in zip(clean_indices, clean_wavelengths):
            f.write(f"{idx}, {wl:.1f}\n")

        f.write(f"\n# Total clean bands: {len(clean_indices)}\n")
        f.write(f"\n# Python list (0-based indices):\n")
        f.write("clean_band_indices = [\n")

        # Format as Python list
        python_indices = [idx - 1 for idx in clean_indices]
        for i in range(0, len(python_indices), 10):
            chunk = python_indices[i:i+10]
            f.write("    " + ", ".join(map(str, chunk)) + ",\n")
        f.write("]\n")

    print(f"   ✓ Band list saved: {list_path}")


def visualize_clean_vs_noisy(original_cube, normalized_cube, quality_metrics,
                             output_dir, n_bands, wl_min, wl_max):
    """Create visualization comparing clean vs noisy bands."""

    import matplotlib.pyplot as plt

    # Select samples: 2 clean, 2 noisy
    clean_bands = [m for m in quality_metrics if m['is_clean']]
    noisy_bands = [m for m in quality_metrics if not m['is_clean']]

    if len(clean_bands) >= 2 and len(noisy_bands) >= 2:
        # Pick diverse samples
        clean_samples = [
            clean_bands[len(clean_bands)//4],  # 25%
            clean_bands[3*len(clean_bands)//4]  # 75%
        ]
        noisy_samples = [
            noisy_bands[0],  # First noisy
            noisy_bands[len(noisy_bands)//2]  # Middle noisy
        ]

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        # Clean bands
        for i, metric in enumerate(clean_samples):
            idx = metric['band_index']
            wl = calculate_wavelength(idx, n_bands, wl_min, wl_max)

            # Original
            ax = axes[0, i*2]
            ax.imshow(original_cube[idx], cmap='gray')
            ax.set_title(f'Clean Band {idx+1}\n{wl:.1f}nm (Original)\nSNR: {metric["snr"]:.1f}',
                        fontsize=10, color='green', fontweight='bold')
            ax.axis('off')

            # Normalized
            ax = axes[0, i*2+1]
            ax.imshow(normalized_cube[idx], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Clean Band {idx+1}\n{wl:.1f}nm (Normalized)\nSNR: {metric["snr"]:.1f}',
                        fontsize=10, color='green', fontweight='bold')
            ax.axis('off')

        # Noisy bands
        for i, metric in enumerate(noisy_samples):
            idx = metric['band_index']
            wl = calculate_wavelength(idx, n_bands, wl_min, wl_max)

            # Original
            ax = axes[1, i*2]
            ax.imshow(original_cube[idx], cmap='gray')
            issues = ", ".join(metric['issues'][:1])
            ax.set_title(f'Noisy Band {idx+1}\n{wl:.1f}nm (Original)\n{issues}',
                        fontsize=10, color='red')
            ax.axis('off')

            # Normalized
            ax = axes[1, i*2+1]
            ax.imshow(normalized_cube[idx], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Noisy Band {idx+1}\n{wl:.1f}nm (Normalized)\nSNR: {metric["snr"]:.1f}',
                        fontsize=10, color='red')
            ax.axis('off')

        fig.suptitle('Clean vs Noisy Bands Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()

        vis_path = os.path.join(output_dir, 'clean_vs_noisy_comparison.png')
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   ✓ Visualization saved: {vis_path}")


if __name__ == '__main__':
    """Main execution."""

    # Export clean bands with wavelength information
    export_clean_bands(
        input_dir='training_dataset',
        output_dir='clean_normalized_bands',
        method='percentile',
        brightness_boost=True,
        band_wise=True,
        snr_threshold=5.0,        # Adjust this to be more/less strict
        variance_threshold=10.0,   # Adjust this to be more/less strict
        wl_min=900,               # Starting wavelength (nm)
        wl_max=1700               # Ending wavelength (nm)
    )

    print("\n" + "="*90)
    print("USAGE INSTRUCTIONS")
    print("="*90)
    print("""
The clean bands have been exported with wavelength information:
  - Filename format: Band_XXX_WWWW.Wnm_normalized.png
  - Example: Band_001_900.0nm_normalized.png
  - Example: Band_230_1300.5nm_normalized.png

To adjust noise filtering:
  - Increase snr_threshold (e.g., 10.0) for stricter filtering
  - Decrease snr_threshold (e.g., 3.0) for more lenient filtering

Check quality_report.txt for detailed analysis of each band.
Use clean_bands_list.txt to update your model to use only clean bands.
""")
