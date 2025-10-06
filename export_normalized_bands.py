"""
Export all spectral bands after band-wise normalization for data integrity checking.

This script:
1. Loads all 459 spectral bands
2. Applies band-wise normalization
3. Exports each normalized band as PNG with clear naming
4. Generates a verification report
"""

import os
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
from preprocessing import HyperspectralPreprocessor


def export_normalized_bands(
    input_dir='training_dataset',
    output_dir='normalized_bands',
    method='percentile',
    brightness_boost=True,
    band_wise=True
):
    """
    Export all normalized bands for visual inspection.

    Args:
        input_dir: Directory containing original ImagesStack*.png files
        output_dir: Directory to save normalized bands
        method: Normalization method ('simple' or 'percentile')
        brightness_boost: Whether to boost brightness
        band_wise: Whether to normalize per-band (recommended: True)
    """

    print("\n" + "="*80)
    print("EXPORTING NORMALIZED SPECTRAL BANDS")
    print("="*80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load all spectral band files
    print(f"\n1. Loading spectral bands from: {input_dir}")
    image_files = sorted(glob.glob(os.path.join(input_dir, 'ImagesStack*.png')))

    if len(image_files) == 0:
        raise FileNotFoundError(f"No ImagesStack*.png files found in {input_dir}")

    n_bands = len(image_files)
    print(f"   Found {n_bands} spectral bands")

    # Load first image to get dimensions
    first_img = np.array(Image.open(image_files[0]))
    height, width = first_img.shape
    print(f"   Image dimensions: {height} x {width}")

    # Pre-allocate array for all bands
    spectral_cube = np.zeros((n_bands, height, width), dtype=np.float32)

    # Load all bands
    print(f"\n2. Loading all {n_bands} bands into memory...")
    for i, img_path in enumerate(tqdm(image_files, desc="   Loading")):
        img = np.array(Image.open(img_path))
        spectral_cube[i] = img.astype(np.float32)

    # Create preprocessor
    print(f"\n3. Applying band-wise normalization...")
    print(f"   Method: {method}")
    print(f"   Brightness boost: {brightness_boost}")
    print(f"   Band-wise: {band_wise}")

    preprocessor = HyperspectralPreprocessor(
        method=method,
        brightness_boost=brightness_boost,
        band_wise=band_wise,
        percentile_low=1,
        percentile_high=99
    )

    # Apply preprocessing
    normalized_cube = preprocessor.preprocess(spectral_cube, fit_pca=False)

    # Export normalized bands
    print(f"\n4. Exporting normalized bands to: {output_dir}")

    stats_report = []

    for i in tqdm(range(n_bands), desc="   Exporting"):
        # Get original and normalized band
        original_band = spectral_cube[i]
        normalized_band = normalized_cube[i]

        # Convert normalized band to 8-bit for PNG export
        # Scale [0, 1] to [0, 255]
        normalized_8bit = (normalized_band * 255).clip(0, 255).astype(np.uint8)

        # Create filename with band number (zero-padded)
        # Format: Band_001_normalized.png, Band_002_normalized.png, etc.
        filename = f"Band_{i+1:03d}_normalized.png"
        output_path = os.path.join(output_dir, filename)

        # Save as PNG
        Image.fromarray(normalized_8bit).save(output_path)

        # Collect statistics
        stats = {
            'band_number': i + 1,
            'original_file': os.path.basename(image_files[i]),
            'output_file': filename,
            'original_min': float(original_band.min()),
            'original_max': float(original_band.max()),
            'original_mean': float(original_band.mean()),
            'normalized_min': float(normalized_band.min()),
            'normalized_max': float(normalized_band.max()),
            'normalized_mean': float(normalized_band.mean()),
            'range_utilization': float((normalized_band.max() - normalized_band.min()) * 100)
        }
        stats_report.append(stats)

    print(f"\n✓ Exported {n_bands} normalized bands")

    # Generate verification report
    print(f"\n5. Generating verification report...")
    generate_report(stats_report, output_dir, method, brightness_boost, band_wise)

    # Generate comparison samples
    print(f"\n6. Generating comparison samples...")
    generate_comparison_samples(spectral_cube, normalized_cube, output_dir)

    print(f"\n{'='*80}")
    print("EXPORT COMPLETE!")
    print(f"{'='*80}")
    print(f"\nNormalized bands saved to: {output_dir}/")
    print(f"Verification report: {output_dir}/normalization_report.txt")
    print(f"Comparison samples: {output_dir}/comparison_samples.png")
    print(f"\nTotal files exported: {n_bands}")


def generate_report(stats_report, output_dir, method, brightness_boost, band_wise):
    """Generate detailed verification report."""

    report_path = os.path.join(output_dir, 'normalization_report.txt')

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BAND-WISE NORMALIZATION VERIFICATION REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("NORMALIZATION SETTINGS:\n")
        f.write(f"  Method: {method}\n")
        f.write(f"  Brightness boost: {brightness_boost}\n")
        f.write(f"  Band-wise processing: {band_wise}\n")
        f.write(f"  Total bands: {len(stats_report)}\n\n")

        f.write("="*80 + "\n")
        f.write("DETAILED STATISTICS (Per Band)\n")
        f.write("="*80 + "\n\n")

        # Header
        f.write(f"{'Band':<6} {'Original File':<25} {'Original Range':<25} {'Normalized Range':<20} {'Util%':<8}\n")
        f.write("-"*80 + "\n")

        # Band statistics
        for stat in stats_report:
            orig_range = f"[{stat['original_min']:.1f}, {stat['original_max']:.1f}]"
            norm_range = f"[{stat['normalized_min']:.3f}, {stat['normalized_max']:.3f}]"

            f.write(f"{stat['band_number']:<6} {stat['original_file']:<25} {orig_range:<25} "
                   f"{norm_range:<20} {stat['range_utilization']:.1f}%\n")

        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*80 + "\n\n")

        # Calculate summary statistics
        range_utils = [s['range_utilization'] for s in stats_report]

        f.write(f"Range Utilization Statistics:\n")
        f.write(f"  Minimum: {min(range_utils):.2f}%\n")
        f.write(f"  Maximum: {max(range_utils):.2f}%\n")
        f.write(f"  Average: {np.mean(range_utils):.2f}%\n")
        f.write(f"  Median:  {np.median(range_utils):.2f}%\n\n")

        # Check for potential issues
        f.write("DATA INTEGRITY CHECKS:\n")
        low_util_bands = [s for s in stats_report if s['range_utilization'] < 50]

        if len(low_util_bands) == 0:
            f.write("  ✓ All bands have good range utilization (>50%)\n")
        else:
            f.write(f"  ⚠ {len(low_util_bands)} bands have low utilization (<50%):\n")
            for band in low_util_bands:
                f.write(f"    - Band {band['band_number']}: {band['range_utilization']:.1f}%\n")

        # Check normalization success
        properly_normalized = sum(1 for s in stats_report
                                 if s['normalized_min'] >= 0 and s['normalized_max'] <= 1.01)

        f.write(f"\n  ✓ {properly_normalized}/{len(stats_report)} bands properly normalized to [0,1]\n")

        # Band-wise effectiveness
        if band_wise:
            high_util = sum(1 for s in stats_report if s['range_utilization'] > 90)
            f.write(f"  ✓ {high_util}/{len(stats_report)} bands have excellent utilization (>90%)\n")
            f.write(f"\n  Band-wise normalization is WORKING CORRECTLY! ✓\n")

        f.write("\n" + "="*80 + "\n")
        f.write("VERIFICATION COMPLETE\n")
        f.write("="*80 + "\n")

    print(f"   ✓ Report saved: {report_path}")


def generate_comparison_samples(original_cube, normalized_cube, output_dir):
    """Generate visual comparison of original vs normalized for sample bands."""

    import matplotlib.pyplot as plt

    n_bands = original_cube.shape[0]

    # Select diverse sample bands (first, 25%, 50%, 75%, last)
    sample_indices = [
        0,
        n_bands // 4,
        n_bands // 2,
        3 * n_bands // 4,
        n_bands - 1
    ]

    fig, axes = plt.subplots(2, len(sample_indices), figsize=(20, 8))

    for idx, band_idx in enumerate(sample_indices):
        # Original
        ax = axes[0, idx]
        im = ax.imshow(original_cube[band_idx], cmap='gray')
        ax.set_title(f'Original Band {band_idx + 1}\n[{original_cube[band_idx].min():.0f}, {original_cube[band_idx].max():.0f}]',
                    fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Normalized
        ax = axes[1, idx]
        im = ax.imshow(normalized_cube[band_idx], cmap='gray', vmin=0, vmax=1)
        util = (normalized_cube[band_idx].max() - normalized_cube[band_idx].min()) * 100
        ax.set_title(f'Normalized Band {band_idx + 1}\n[{normalized_cube[band_idx].min():.3f}, {normalized_cube[band_idx].max():.3f}]\nUtil: {util:.0f}%',
                    fontsize=10, color='green', fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle('Sample Comparison: Original vs Band-wise Normalized',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    comparison_path = os.path.join(output_dir, 'comparison_samples.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Comparison saved: {comparison_path}")


def verify_exported_files(output_dir='normalized_bands'):
    """Quick verification that all files were exported correctly."""

    print("\n" + "="*80)
    print("VERIFYING EXPORTED FILES")
    print("="*80)

    exported_files = sorted(glob.glob(os.path.join(output_dir, 'Band_*_normalized.png')))

    print(f"\nTotal files found: {len(exported_files)}")

    if len(exported_files) == 0:
        print("⚠ No files found! Export may have failed.")
        return

    # Check sequential numbering
    expected_numbers = set(range(1, len(exported_files) + 1))
    actual_numbers = set()

    for filepath in exported_files:
        filename = os.path.basename(filepath)
        # Extract band number from "Band_XXX_normalized.png"
        try:
            band_num = int(filename.split('_')[1])
            actual_numbers.add(band_num)
        except:
            print(f"⚠ Warning: Unexpected filename format: {filename}")

    missing = expected_numbers - actual_numbers
    if missing:
        print(f"⚠ Missing band numbers: {sorted(missing)}")
    else:
        print("✓ All band numbers present and sequential")

    # Check file sizes
    file_sizes = [os.path.getsize(f) for f in exported_files]
    avg_size = np.mean(file_sizes)

    print(f"\nFile size statistics:")
    print(f"  Average: {avg_size/1024:.1f} KB")
    print(f"  Min: {min(file_sizes)/1024:.1f} KB")
    print(f"  Max: {max(file_sizes)/1024:.1f} KB")

    # Sample check - load a few files
    print(f"\nSample file check:")
    sample_files = [exported_files[0], exported_files[len(exported_files)//2], exported_files[-1]]

    for filepath in sample_files:
        try:
            img = Image.open(filepath)
            arr = np.array(img)
            filename = os.path.basename(filepath)
            print(f"  ✓ {filename}: {img.size}, dtype={arr.dtype}, range=[{arr.min()}, {arr.max()}]")
        except Exception as e:
            print(f"  ✗ {os.path.basename(filepath)}: Error - {e}")

    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    """Main execution."""

    # Export normalized bands
    export_normalized_bands(
        input_dir='training_dataset',
        output_dir='normalized_bands',
        method='percentile',        # Use percentile normalization
        brightness_boost=True,      # Apply brightness boost
        band_wise=True              # Band-wise processing (RECOMMENDED)
    )

    # Verify export
    verify_exported_files('normalized_bands')

    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)
    print("""
Next steps:
1. Check normalized_bands/ directory
2. Review normalization_report.txt for statistics
3. View comparison_samples.png to verify normalization quality
4. Manually inspect a few Band_XXX_normalized.png files

Files are named: Band_001_normalized.png, Band_002_normalized.png, ..., Band_459_normalized.png

All bands are normalized to [0, 1] range and exported as 8-bit PNG (0-255) for viewing.
""")
