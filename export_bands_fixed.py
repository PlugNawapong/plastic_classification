"""
FIXED: Export clean bands with proper percentage control.

This version correctly respects the keep_percentage parameter.
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


def filter_by_percentile(metrics, keep_percentage):
    """
    Filter bands by percentile - FIXED VERSION.

    Keeps top X% of bands based on SNR score only.
    No hard-coded overrides.
    """
    # Calculate percentile threshold
    percentile = 100 - keep_percentage
    snr_values = [m['snr'] for m in metrics]

    # Determine threshold
    snr_threshold = np.percentile(snr_values, percentile)

    print(f"\nFiltering by percentile (keeping top {keep_percentage}%):")
    print(f"  SNR threshold: {snr_threshold:.2f}")

    # Mark bands as clean/noisy based ONLY on SNR percentile
    clean_count = 0
    for metric in metrics:
        if metric['snr'] >= snr_threshold:
            metric['is_clean'] = True
            metric['issues'] = []
            clean_count += 1
        else:
            metric['is_clean'] = False
            metric['issues'] = [f"SNR {metric['snr']:.2f} below threshold {snr_threshold:.2f}"]

    noisy_count = len(metrics) - clean_count

    print(f"\nFiltering results:")
    print(f"  ✓ Clean: {clean_count}/{len(metrics)} ({100*clean_count/len(metrics):.1f}%)")
    print(f"  ✗ Noisy: {noisy_count}/{len(metrics)} ({100*noisy_count/len(metrics):.1f}%)")

    return metrics, snr_threshold


def filter_by_manual_thresholds(metrics, snr_threshold, var_threshold, check_saturation=True):
    """
    Filter bands using manual thresholds.

    Args:
        metrics: Quality metrics
        snr_threshold: Minimum SNR
        var_threshold: Minimum variance
        check_saturation: Whether to check saturation (optional)
    """
    print(f"\nFiltering with manual thresholds:")
    print(f"  SNR threshold: {snr_threshold:.2f}")
    print(f"  Variance threshold: {var_threshold:.2f}")
    print(f"  Check saturation: {check_saturation}")

    clean_count = 0
    for metric in metrics:
        is_clean = True
        issues = []

        if metric['snr'] < snr_threshold:
            is_clean = False
            issues.append(f"Low SNR ({metric['snr']:.2f})")

        if metric['variance'] < var_threshold:
            is_clean = False
            issues.append(f"Low variance ({metric['variance']:.2f})")

        if check_saturation and metric['saturation_pct'] > 50:  # Very lenient
            is_clean = False
            issues.append(f"High saturation ({metric['saturation_pct']:.1f}%)")

        if metric['mean'] < 0.1:  # Nearly all zeros
            is_clean = False
            issues.append("Near-zero signal")

        metric['is_clean'] = is_clean
        metric['issues'] = issues

        if is_clean:
            clean_count += 1

    noisy_count = len(metrics) - clean_count

    print(f"\nFiltering results:")
    print(f"  ✓ Clean: {clean_count}/{len(metrics)} ({100*clean_count/len(metrics):.1f}%)")
    print(f"  ✗ Noisy: {noisy_count}/{len(metrics)} ({100*noisy_count/len(metrics):.1f}%)")

    return metrics, snr_threshold


def export_clean_bands(
    input_dir='training_dataset',
    output_dir='clean_normalized_bands',
    keep_percentage=75,
    use_manual=False,
    manual_snr=2.0,
    manual_var=50.0
):
    """
    Export clean bands - FIXED to respect keep_percentage.

    Args:
        input_dir: Input directory
        output_dir: Output directory
        keep_percentage: Percentage to keep (auto mode)
        use_manual: Use manual thresholds instead
        manual_snr: Manual SNR threshold
        manual_var: Manual variance threshold
    """

    print("="*90)
    print("EXPORT CLEAN BANDS (FIXED VERSION)")
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

    # Filter
    print(f"\n4. Filtering bands")
    if use_manual:
        metrics, threshold = filter_by_manual_thresholds(metrics, manual_snr, manual_var)
        threshold_type = f"SNR≥{manual_snr:.1f}, Var≥{manual_var:.1f}"
    else:
        metrics, threshold = filter_by_percentile(metrics, keep_percentage)
        threshold_type = f"Top {keep_percentage}% (SNR≥{threshold:.2f})"

    # Normalize
    print(f"\n5. Applying band-wise normalization")
    preprocessor = HyperspectralPreprocessor(
        method='percentile',
        brightness_boost=True,
        band_wise=True,
        percentile_low=1,
        percentile_high=99
    )
    normalized_cube = preprocessor.preprocess(spectral_cube, fit_pca=False)

    # Export
    print(f"\n6. Exporting clean bands")
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
    print(f"\n7. Generating reports")

    # Quality report
    with open(os.path.join(output_dir, 'quality_report.txt'), 'w') as f:
        f.write("QUALITY REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Wavelength range: {min(wavelengths):.1f} - {max(wavelengths):.1f} nm\n")
        f.write(f"Total bands: {n_bands}\n")
        f.write(f"Clean bands: {clean_count}/{n_bands} ({100*clean_count/n_bands:.1f}%)\n")
        f.write(f"Threshold: {threshold_type}\n\n")

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
        f.write(f"# Total: {clean_count}\n")
        f.write(f"# Filter: {threshold_type}\n\n")

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
    print(f"\nOutput: {output_dir}/")
    print(f"Clean bands: {clean_count}/{n_bands} ({100*clean_count/n_bands:.1f}%)")
    print(f"Threshold: {threshold_type}")
    print(f"\nFilename format: Band_XXX_WWW.Wnm_normalized.png")


if __name__ == '__main__':
    # PERCENTILE MODE (keeps exact percentage)
    export_clean_bands(
        input_dir='training_dataset',
        output_dir='clean_normalized_bands',
        keep_percentage=75,     # Change this: 50, 60, 75, 80, 90
        use_manual=False
    )

    # Or MANUAL MODE
    # export_clean_bands(
    #     input_dir='training_dataset',
    #     output_dir='clean_normalized_bands',
    #     use_manual=True,
    #     manual_snr=2.0,
    #     manual_var=50.0
    # )
