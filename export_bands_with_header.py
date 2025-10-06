"""
Export spectral bands with actual wavelength from header.json and noise filtering.

This script:
1. Reads wavelength information from header.json
2. Detects and filters noisy bands using SNR analysis
3. Exports only clean bands with actual wavelength in filename
4. Generates detailed quality report
"""

import os
import json
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
from preprocessing import HyperspectralPreprocessor


def load_wavelengths_from_header(data_dir):
    """
    Load wavelength information from header.json.

    Args:
        data_dir: Directory containing header.json

    Returns:
        List of wavelengths in nm
    """
    header_path = os.path.join(data_dir, 'header.json')

    if not os.path.exists(header_path):
        raise FileNotFoundError(f"header.json not found in {data_dir}")

    with open(header_path, 'r') as f:
        header_data = json.load(f)

    wavelengths = header_data.get('wavelength (nm)', [])

    if not wavelengths:
        raise ValueError(f"No wavelength data found in {header_path}")

    print(f"\nWavelength information from header.json:")
    print(f"  Number of bands: {len(wavelengths)}")
    print(f"  Range: {min(wavelengths):.2f} - {max(wavelengths):.2f} nm")
    print(f"  Step size: {(max(wavelengths) - min(wavelengths)) / (len(wavelengths) - 1):.2f} nm")

    return wavelengths


def calculate_snr(band):
    """Calculate Signal-to-Noise Ratio."""
    signal = np.mean(band)
    noise = np.std(band)

    if noise < 1e-6:
        return 0.0

    snr = signal / noise
    return snr


def detect_noisy_bands(spectral_cube, snr_threshold=5.0, variance_threshold=10.0):
    """Detect noisy bands using quality criteria."""
    n_bands, height, width = spectral_cube.shape
    quality_metrics = []

    print("\nAnalyzing band quality...")
    for i in tqdm(range(n_bands), desc="Analyzing"):
        band = spectral_cube[i]

        mean_val = np.mean(band)
        std_val = np.std(band)
        variance = np.var(band)
        snr = calculate_snr(band)

        # Calculate saturation
        max_possible = 65535 if band.max() > 4095 else 4095
        saturated_pixels = np.sum(band >= max_possible * 0.99)
        saturation_pct = 100 * saturated_pixels / (height * width)

        # Determine quality
        is_clean = True
        issues = []

        if snr < snr_threshold:
            is_clean = False
            issues.append(f"Low SNR ({snr:.2f})")

        if variance < variance_threshold:
            is_clean = False
            issues.append(f"Low variance ({variance:.2f})")

        if saturation_pct > 10:
            is_clean = False
            issues.append(f"High saturation ({saturation_pct:.1f}%)")

        if mean_val < 10:
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


def export_clean_bands_with_header(
    input_dir='training_dataset',
    output_dir='clean_normalized_bands',
    method='percentile',
    brightness_boost=True,
    band_wise=True,
    snr_threshold=5.0,
    variance_threshold=10.0
):
    """
    Export clean normalized bands using actual wavelengths from header.json.

    Args:
        input_dir: Directory with spectral bands and header.json
        output_dir: Output directory
        method: Normalization method
        brightness_boost: Apply brightness boost
        band_wise: Band-wise normalization
        snr_threshold: Minimum SNR
        variance_threshold: Minimum variance
    """

    print("\n" + "="*90)
    print("EXPORTING CLEAN BANDS WITH WAVELENGTH FROM HEADER.JSON")
    print("="*90)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load wavelengths from header.json
    print(f"\n1. Reading wavelength information from {input_dir}/header.json")
    wavelengths = load_wavelengths_from_header(input_dir)

    # Load spectral bands
    print(f"\n2. Loading spectral bands from: {input_dir}")
    image_files = sorted(glob.glob(os.path.join(input_dir, 'ImagesStack*.png')))

    if len(image_files) == 0:
        raise FileNotFoundError(f"No ImagesStack*.png files found in {input_dir}")

    n_bands = len(image_files)
    print(f"   Found {n_bands} spectral band images")

    # Verify wavelength count matches band count
    if len(wavelengths) != n_bands:
        print(f"\n   ⚠ Warning: Wavelength count ({len(wavelengths)}) doesn't match band count ({n_bands})")
        print(f"   Using first {min(len(wavelengths), n_bands)} bands")
        n_bands = min(len(wavelengths), n_bands)
        wavelengths = wavelengths[:n_bands]
        image_files = image_files[:n_bands]

    # Load dimensions
    first_img = np.array(Image.open(image_files[0]))
    height, width = first_img.shape
    print(f"   Image dimensions: {height} x {width}")

    # Pre-allocate array
    spectral_cube = np.zeros((n_bands, height, width), dtype=np.float32)

    # Load all bands
    print(f"\n3. Loading all {n_bands} bands...")
    for i, img_path in enumerate(tqdm(image_files, desc="   Loading")):
        img = np.array(Image.open(img_path))
        spectral_cube[i] = img.astype(np.float32)

    # Detect noisy bands
    print(f"\n4. Detecting noisy bands (SNR≥{snr_threshold}, Var≥{variance_threshold})...")
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
            wl = wavelengths[m['band_index']]
            issues_str = ", ".join(m['issues'])
            print(f"     Band {m['band_index']+1:3d} ({wl:.1f}nm): {issues_str}")

    # Apply normalization
    print(f"\n5. Applying band-wise normalization...")
    preprocessor = HyperspectralPreprocessor(
        method=method,
        brightness_boost=brightness_boost,
        band_wise=band_wise,
        percentile_low=1,
        percentile_high=99
    )
    normalized_cube = preprocessor.preprocess(spectral_cube, fit_pca=False)

    # Export only CLEAN bands
    print(f"\n6. Exporting {len(clean_bands)} clean bands to: {output_dir}")

    exported_files = []
    clean_export_count = 0

    for metric in tqdm(quality_metrics, desc="   Exporting"):
        i = metric['band_index']
        wavelength = wavelengths[i]

        # Create filename with actual wavelength
        # Format: Band_001_450.5nm_normalized.png
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
            # Mark as skipped
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
    print(f"\n7. Generating quality reports...")
    generate_quality_report(quality_metrics, exported_files, wavelengths, output_dir, n_bands)
    generate_band_list(exported_files, wavelengths, output_dir)

    # Generate visualization
    print(f"\n8. Generating visualizations...")
    visualize_clean_vs_noisy(spectral_cube, normalized_cube, quality_metrics,
                            wavelengths, output_dir)

    print(f"\n{'='*90}")
    print("EXPORT COMPLETE!")
    print(f"{'='*90}")
    print(f"\nClean bands exported to: {output_dir}/")
    print(f"Quality report: {output_dir}/quality_report.txt")
    print(f"Clean band list: {output_dir}/clean_bands_list.txt")
    print(f"Comparison visualization: {output_dir}/clean_vs_noisy_comparison.png")
    print(f"\nTotal clean bands: {clean_export_count}/{n_bands}")
    print(f"Wavelength range (clean): {min([w for i, w in enumerate(wavelengths) if quality_metrics[i]['is_clean']]):.1f} - {max([w for i, w in enumerate(wavelengths) if quality_metrics[i]['is_clean']]):.1f} nm")


def generate_quality_report(quality_metrics, exported_files, wavelengths, output_dir, n_bands):
    """Generate detailed quality report."""

    report_path = os.path.join(output_dir, 'quality_report.txt')

    with open(report_path, 'w') as f:
        f.write("="*90 + "\n")
        f.write("SPECTRAL BAND QUALITY REPORT\n")
        f.write("="*90 + "\n\n")

        f.write("WAVELENGTH INFORMATION (from header.json):\n")
        f.write(f"  Total bands: {n_bands}\n")
        f.write(f"  Wavelength range: {min(wavelengths):.2f} - {max(wavelengths):.2f} nm\n")
        f.write(f"  Average step: {(max(wavelengths)-min(wavelengths))/(len(wavelengths)-1):.2f} nm\n\n")

        clean_count = sum(1 for m in quality_metrics if m['is_clean'])
        f.write(f"SUMMARY:\n")
        f.write(f"  Clean bands: {clean_count}/{n_bands} ({100*clean_count/n_bands:.1f}%)\n")
        f.write(f"  Noisy bands: {n_bands-clean_count}/{n_bands} ({100*(n_bands-clean_count)/n_bands:.1f}%)\n\n")

        f.write("="*90 + "\n")
        f.write("CLEAN BANDS (EXPORTED)\n")
        f.write("="*90 + "\n\n")

        f.write(f"{'Band':<6} {'Wavelength':<15} {'SNR':<10} {'Variance':<12} {'Filename':<40}\n")
        f.write("-"*90 + "\n")

        for item in exported_files:
            if item['is_clean']:
                f.write(f"{item['band_index']:<6} {item['wavelength']:.2f} nm{'':<6} "
                       f"{item['snr']:<10.2f} {item['variance']:<12.2f} {item['filename']:<40}\n")

        f.write("\n" + "="*90 + "\n")
        f.write("NOISY BANDS (SKIPPED)\n")
        f.write("="*90 + "\n\n")

        f.write(f"{'Band':<6} {'Wavelength':<15} {'SNR':<10} {'Variance':<12} {'Issues':<40}\n")
        f.write("-"*90 + "\n")

        for item in exported_files:
            if not item['is_clean']:
                issues_str = ", ".join(item['issues'][:2])
                f.write(f"{item['band_index']:<6} {item['wavelength']:.2f} nm{'':<6} "
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

        f.write("WAVELENGTH RANGE OF CLEAN BANDS:\n")
        clean_wavelengths = [wavelengths[m['band_index']] for m in quality_metrics if m['is_clean']]
        if clean_wavelengths:
            f.write(f"  Min wavelength: {min(clean_wavelengths):.2f} nm\n")
            f.write(f"  Max wavelength: {max(clean_wavelengths):.2f} nm\n")
            f.write(f"  Coverage: {max(clean_wavelengths) - min(clean_wavelengths):.2f} nm\n\n")

        f.write("RECOMMENDATION:\n")
        f.write(f"  Use only the {clean_count} clean bands for training.\n")
        f.write(f"  Update model to accept {clean_count} spectral bands.\n")
        f.write(f"  See clean_bands_list.txt for indices and wavelengths.\n")

    print(f"   ✓ Quality report saved: {report_path}")


def generate_band_list(exported_files, wavelengths, output_dir):
    """Generate clean band list."""

    list_path = os.path.join(output_dir, 'clean_bands_list.txt')

    clean_items = [item for item in exported_files if item['is_clean']]

    with open(list_path, 'w') as f:
        f.write("# Clean Band Indices and Wavelengths\n")
        f.write("# Format: Band_Index, Wavelength (nm)\n")
        f.write(f"# Data source: header.json\n")
        f.write(f"# Total clean bands: {len(clean_items)}\n\n")

        for item in clean_items:
            f.write(f"{item['band_index']}, {item['wavelength']:.2f}\n")

        f.write(f"\n# Python list (0-based indices):\n")
        f.write("clean_band_indices = [\n")

        python_indices = [item['band_index'] - 1 for item in clean_items]
        for i in range(0, len(python_indices), 10):
            chunk = python_indices[i:i+10]
            f.write("    " + ", ".join(map(str, chunk)) + ",\n")
        f.write("]\n\n")

        f.write(f"# Wavelengths (nm):\n")
        f.write("clean_wavelengths = [\n")
        for i in range(0, len(clean_items), 5):
            chunk = [f"{item['wavelength']:.2f}" for item in clean_items[i:i+5]]
            f.write("    " + ", ".join(chunk) + ",\n")
        f.write("]\n")

    print(f"   ✓ Band list saved: {list_path}")


def visualize_clean_vs_noisy(original_cube, normalized_cube, quality_metrics,
                             wavelengths, output_dir):
    """Visualize clean vs noisy bands."""

    import matplotlib.pyplot as plt

    clean_bands = [m for m in quality_metrics if m['is_clean']]
    noisy_bands = [m for m in quality_metrics if not m['is_clean']]

    if len(clean_bands) >= 2 and len(noisy_bands) >= 2:
        clean_samples = [
            clean_bands[len(clean_bands)//4],
            clean_bands[3*len(clean_bands)//4]
        ]
        noisy_samples = [
            noisy_bands[0],
            noisy_bands[len(noisy_bands)//2] if len(noisy_bands) > 1 else noisy_bands[0]
        ]

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        # Clean bands
        for i, metric in enumerate(clean_samples):
            idx = metric['band_index']
            wl = wavelengths[idx]

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
            wl = wavelengths[idx]

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

        fig.suptitle('Clean vs Noisy Bands (Wavelengths from header.json)', fontsize=14, fontweight='bold')
        plt.tight_layout()

        vis_path = os.path.join(output_dir, 'clean_vs_noisy_comparison.png')
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   ✓ Visualization saved: {vis_path}")


if __name__ == '__main__':
    """Main execution."""

    # Export clean bands using actual wavelengths from header.json
    export_clean_bands_with_header(
        input_dir='training_dataset',
        output_dir='clean_normalized_bands',
        method='percentile',
        brightness_boost=True,
        band_wise=True,
        snr_threshold=5.0,
        variance_threshold=10.0
    )

    print("\n" + "="*90)
    print("USAGE INSTRUCTIONS")
    print("="*90)
    print("""
✓ Clean bands exported with actual wavelengths from header.json
✓ Filename format: Band_XXX_WWW.Wnm_normalized.png
  Example: Band_001_450.5nm_normalized.png
  Example: Band_230_653.4nm_normalized.png

Wavelength range: 450.5 - 853.6 nm (visible to near-infrared)

To adjust filtering:
  - Increase snr_threshold for stricter filtering
  - Decrease snr_threshold for more lenient filtering

Check quality_report.txt for detailed analysis.
Use clean_bands_list.txt to update your model.
""")
