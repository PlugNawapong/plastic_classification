"""
Export clean bands with wavelengths from header.json (Auto-adjusted thresholds)

This version automatically adjusts quality thresholds based on your data.
"""

import os
import json
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
from preprocessing import HyperspectralPreprocessor


def load_wavelengths_from_header(data_dir):
    """Load wavelengths from header.json."""
    header_path = os.path.join(data_dir, 'header.json')

    if not os.path.exists(header_path):
        raise FileNotFoundError(f"header.json not found in {data_dir}")

    with open(header_path, 'r') as f:
        header_data = json.load(f)

    wavelengths = header_data.get('wavelength (nm)', [])

    if not wavelengths:
        raise ValueError(f"No wavelength data in {header_path}")

    print(f"\nWavelength info from header.json:")
    print(f"  Bands: {len(wavelengths)}")
    print(f"  Range: {min(wavelengths):.1f} - {max(wavelengths):.1f} nm")
    print(f"  Step: {(max(wavelengths)-min(wavelengths))/(len(wavelengths)-1):.2f} nm")

    return wavelengths


def calculate_quality_metrics(spectral_cube):
    """Calculate quality metrics for all bands."""
    n_bands, height, width = spectral_cube.shape
    metrics = []

    print("\nAnalyzing band quality...")
    for i in tqdm(range(n_bands), desc="Analyzing"):
        band = spectral_cube[i]

        mean_val = float(np.mean(band))
        std_val = float(np.std(band))
        variance = float(np.var(band))

        # SNR
        snr = mean_val / (std_val + 1e-10)

        # Saturation check
        max_possible = 255 if band.max() <= 255 else (4095 if band.max() <= 4095 else 65535)
        saturated = np.sum(band >= max_possible * 0.99)
        saturation_pct = 100.0 * saturated / (height * width)

        metrics.append({
            'band_index': i,
            'mean': mean_val,
            'std': std_val,
            'variance': variance,
            'snr': snr,
            'saturation_pct': saturation_pct
        })

    return metrics


def auto_determine_thresholds(metrics, percentile=25):
    """
    Automatically determine quality thresholds.

    Uses percentile-based approach: keep top 75% of bands.
    """
    snr_values = [m['snr'] for m in metrics]
    var_values = [m['variance'] for m in metrics]

    # Use percentile to determine thresholds
    # Keep bands above 25th percentile (i.e., top 75%)
    snr_threshold = np.percentile(snr_values, percentile)
    var_threshold = np.percentile(var_values, percentile)

    print(f"\nAuto-determined thresholds (keeping top {100-percentile}%):")
    print(f"  SNR threshold: {snr_threshold:.2f}")
    print(f"  Variance threshold: {var_threshold:.2f}")

    return snr_threshold, var_threshold


def filter_noisy_bands(metrics, snr_threshold, var_threshold):
    """Filter bands based on thresholds."""
    for metric in metrics:
        is_clean = True
        issues = []

        if metric['snr'] < snr_threshold:
            is_clean = False
            issues.append(f"Low SNR ({metric['snr']:.2f})")

        if metric['variance'] < var_threshold:
            is_clean = False
            issues.append(f"Low variance ({metric['variance']:.2f})")

        if metric['saturation_pct'] > 10:
            is_clean = False
            issues.append(f"High saturation ({metric['saturation_pct']:.1f}%)")

        if metric['mean'] < 1:
            is_clean = False
            issues.append("Near-zero signal")

        metric['is_clean'] = is_clean
        metric['issues'] = issues

    clean_count = sum(1 for m in metrics if m['is_clean'])
    noisy_count = len(metrics) - clean_count

    print(f"\nFiltering results:")
    print(f"  ✓ Clean: {clean_count}/{len(metrics)} ({100*clean_count/len(metrics):.1f}%)")
    print(f"  ✗ Noisy: {noisy_count}/{len(metrics)} ({100*noisy_count/len(metrics):.1f}%)")

    return metrics


def export_clean_bands(
    input_dir='training_dataset',
    output_dir='clean_normalized_bands',
    auto_threshold=True,
    keep_percentage=75,
    manual_snr=None,
    manual_var=None
):
    """
    Export clean bands with automatic or manual thresholds.

    Args:
        input_dir: Input directory
        output_dir: Output directory
        auto_threshold: Use automatic threshold detection
        keep_percentage: Percentage of bands to keep (if auto_threshold=True)
        manual_snr: Manual SNR threshold (if auto_threshold=False)
        manual_var: Manual variance threshold (if auto_threshold=False)
    """

    print("="*90)
    print("EXPORT CLEAN BANDS WITH WAVELENGTH FROM HEADER.JSON")
    print("="*90)

    os.makedirs(output_dir, exist_ok=True)

    # Load wavelengths
    print(f"\n1. Loading wavelengths from {input_dir}/header.json")
    wavelengths = load_wavelengths_from_header(input_dir)

    # Load images
    print(f"\n2. Loading spectral bands")
    image_files = sorted(glob.glob(os.path.join(input_dir, 'ImagesStack*.png')))

    if not image_files:
        raise FileNotFoundError(f"No images in {input_dir}")

    n_bands = min(len(image_files), len(wavelengths))
    print(f"   Found {len(image_files)} images, {len(wavelengths)} wavelengths")
    print(f"   Using {n_bands} bands")

    # Load first to get dims
    first_img = np.array(Image.open(image_files[0]))
    height, width = first_img.shape

    # Load all
    spectral_cube = np.zeros((n_bands, height, width), dtype=np.float32)
    for i in tqdm(range(n_bands), desc="   Loading"):
        img = np.array(Image.open(image_files[i]))
        spectral_cube[i] = img.astype(np.float32)

    # Calculate metrics
    print(f"\n3. Calculating quality metrics")
    metrics = calculate_quality_metrics(spectral_cube)

    # Determine thresholds
    print(f"\n4. Determining quality thresholds")
    if auto_threshold:
        snr_thresh, var_thresh = auto_determine_thresholds(metrics, percentile=100-keep_percentage)
    else:
        snr_thresh = manual_snr or 3.0
        var_thresh = manual_var or 1.0
        print(f"\nUsing manual thresholds:")
        print(f"  SNR: {snr_thresh}")
        print(f"  Variance: {var_thresh}")

    # Filter
    print(f"\n5. Filtering noisy bands")
    metrics = filter_noisy_bands(metrics, snr_thresh, var_thresh)

    # Normalize
    print(f"\n6. Applying band-wise normalization")
    preprocessor = HyperspectralPreprocessor(
        method='percentile',
        brightness_boost=True,
        band_wise=True,
        percentile_low=1,
        percentile_high=99
    )
    normalized_cube = preprocessor.preprocess(spectral_cube, fit_pca=False)

    # Export
    print(f"\n7. Exporting clean bands")
    exported = []
    clean_count = 0

    for metric in tqdm(metrics, desc="   Exporting"):
        i = metric['band_index']
        wl = wavelengths[i]
        filename = f"Band_{i+1:03d}_{wl:.1f}nm_normalized.png"

        if metric['is_clean']:
            output_path = os.path.join(output_dir, filename)
            band_8bit = (normalized_cube[i] * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(band_8bit).save(output_path)
            clean_count += 1

        exported.append({
            'band': i+1,
            'wavelength': wl,
            'filename': filename,
            'is_clean': metric['is_clean'],
            'snr': metric['snr'],
            'variance': metric['variance'],
            'issues': metric.get('issues', [])
        })

    # Save reports
    print(f"\n8. Generating reports")

    # Quality report
    with open(os.path.join(output_dir, 'quality_report.txt'), 'w') as f:
        f.write("QUALITY REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Wavelength range: {min(wavelengths):.1f} - {max(wavelengths):.1f} nm\n")
        f.write(f"Total bands: {n_bands}\n")
        f.write(f"Clean bands: {clean_count}/{n_bands} ({100*clean_count/n_bands:.1f}%)\n")
        f.write(f"SNR threshold: {snr_thresh:.2f}\n")
        f.write(f"Variance threshold: {var_thresh:.2f}\n\n")

        f.write("CLEAN BANDS:\n")
        f.write("-"*80 + "\n")
        for item in exported:
            if item['is_clean']:
                f.write(f"Band {item['band']:3d}  {item['wavelength']:7.1f}nm  "
                       f"SNR:{item['snr']:6.2f}  Var:{item['variance']:8.1f}  {item['filename']}\n")

        f.write("\nNOISY BANDS (skipped):\n")
        f.write("-"*80 + "\n")
        for item in exported:
            if not item['is_clean']:
                issues = ', '.join(item['issues'][:2])
                f.write(f"Band {item['band']:3d}  {item['wavelength']:7.1f}nm  {issues}\n")

    # Band list
    with open(os.path.join(output_dir, 'clean_bands_list.txt'), 'w') as f:
        f.write("# Clean band indices and wavelengths\n")
        f.write(f"# Total: {clean_count}\n\n")

        for item in exported:
            if item['is_clean']:
                f.write(f"{item['band']}, {item['wavelength']:.2f}\n")

        f.write("\n# Python (0-based):\n")
        f.write("clean_indices = [\n")
        indices = [str(item['band']-1) for item in exported if item['is_clean']]
        for i in range(0, len(indices), 10):
            f.write("    " + ", ".join(indices[i:i+10]) + ",\n")
        f.write("]\n")

    print(f"\n{'='*90}")
    print(f"EXPORT COMPLETE!")
    print(f"{'='*90}")
    print(f"\nOutput directory: {output_dir}/")
    print(f"Clean bands: {clean_count}/{n_bands}")
    print(f"Files: quality_report.txt, clean_bands_list.txt")
    print(f"\nFilename format: Band_XXX_WWW.Wnm_normalized.png")
    print(f"Example: Band_001_{wavelengths[0]:.1f}nm_normalized.png")


if __name__ == '__main__':
    # Run with auto thresholds (keeps top 75% of bands)
    export_clean_bands(
        input_dir='training_dataset',
        output_dir='clean_normalized_bands',
        auto_threshold=True,
        keep_percentage=75  # Keep top 75% of bands
    )

    # Or use manual thresholds:
    # export_clean_bands(
    #     input_dir='training_dataset',
    #     output_dir='clean_normalized_bands',
    #     auto_threshold=False,
    #     manual_snr=2.0,
    #     manual_var=1.0
    # )
