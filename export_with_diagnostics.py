"""
Export bands with built-in diagnostics.
Shows you the data first, then lets you choose thresholds.
"""

import os
import json
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
from preprocessing import HyperspectralPreprocessor


def export_with_diagnostics(
    input_dir='training_dataset',
    output_dir='clean_normalized_bands',
    mode='auto',           # 'auto' or 'manual'
    keep_percentage=75,    # For auto mode
    manual_snr=None,       # For manual mode
    manual_var=None        # For manual mode
):
    """
    Export bands with diagnostics to help choose thresholds.
    """

    print("="*90)
    print("EXPORT BANDS WITH DIAGNOSTICS")
    print("="*90)

    # Load wavelengths
    print(f"\n1. Loading wavelengths from {input_dir}/header.json")
    header_path = os.path.join(input_dir, 'header.json')
    with open(header_path) as f:
        wavelengths = json.load(f)['wavelength (nm)']

    print(f"   Wavelength range: {min(wavelengths):.1f} - {max(wavelengths):.1f} nm")
    print(f"   Number of bands: {len(wavelengths)}")

    # Load images
    print(f"\n2. Loading spectral bands from {input_dir}")
    image_files = sorted(glob.glob(os.path.join(input_dir, 'ImagesStack*.png')))
    n_bands = min(len(image_files), len(wavelengths))

    first_img = np.array(Image.open(image_files[0]))
    height, width = first_img.shape
    print(f"   Image dimensions: {height} x {width}")
    print(f"   Total bands: {n_bands}")

    spectral_cube = np.zeros((n_bands, height, width), dtype=np.float32)
    print(f"\n   Loading all bands...")
    for i in tqdm(range(n_bands)):
        img = np.array(Image.open(image_files[i]))
        spectral_cube[i] = img.astype(np.float32)

    # Analyze quality
    print(f"\n3. Analyzing band quality...")
    snr_values = []
    var_values = []
    mean_values = []

    for i in tqdm(range(n_bands), desc="   Analyzing"):
        band = spectral_cube[i]
        mean = float(np.mean(band))
        std = float(np.std(band))
        var = float(np.var(band))
        snr = mean / (std + 1e-10)

        snr_values.append(snr)
        var_values.append(var)
        mean_values.append(mean)

    # DIAGNOSTICS - Show the data!
    print(f"\n{'='*90}")
    print("DIAGNOSTICS: YOUR DATA STATISTICS")
    print(f"{'='*90}")

    print(f"\nSNR Statistics:")
    print(f"  Minimum:    {min(snr_values):.2f}")
    print(f"  Maximum:    {max(snr_values):.2f}")
    print(f"  Mean:       {np.mean(snr_values):.2f}")
    print(f"  Median:     {np.median(snr_values):.2f}")
    print(f"  25th %ile:  {np.percentile(snr_values, 25):.2f}")
    print(f"  75th %ile:  {np.percentile(snr_values, 75):.2f}")

    print(f"\nVariance Statistics:")
    print(f"  Minimum:    {min(var_values):.2f}")
    print(f"  Maximum:    {max(var_values):.2f}")
    print(f"  Mean:       {np.mean(var_values):.2f}")
    print(f"  Median:     {np.median(var_values):.2f}")
    print(f"  25th %ile:  {np.percentile(var_values, 25):.2f}")
    print(f"  75th %ile:  {np.percentile(var_values, 75):.2f}")

    # Determine thresholds
    print(f"\n{'='*90}")
    print("THRESHOLD SELECTION")
    print(f"{'='*90}")

    if mode == 'auto':
        # Percentile mode
        percentile = 100 - keep_percentage
        snr_threshold = np.percentile(snr_values, percentile)
        var_threshold = np.percentile(var_values, percentile)

        print(f"\nAUTO MODE: Keeping top {keep_percentage}%")
        print(f"  Using {percentile}th percentile as threshold")
        print(f"  SNR threshold:      {snr_threshold:.2f}")
        print(f"  Variance threshold: {var_threshold:.2f}")

    else:
        # Manual mode
        if manual_snr is None or manual_var is None:
            # Auto-suggest based on data
            manual_snr = manual_snr or min(snr_values) * 0.8
            manual_var = manual_var or min(var_values) * 0.8
            print(f"\nMANUAL MODE: Auto-suggested thresholds (lenient)")
        else:
            print(f"\nMANUAL MODE: User-specified thresholds")

        snr_threshold = manual_snr
        var_threshold = manual_var

        print(f"  SNR threshold:      {snr_threshold:.2f}")
        print(f"  Variance threshold: {var_threshold:.2f}")

        # Predict outcome
        snr_pass = sum(1 for s in snr_values if s >= snr_threshold)
        var_pass = sum(1 for v in var_values if v >= var_threshold)
        both_pass = sum(1 for s, v in zip(snr_values, var_values)
                       if s >= snr_threshold and v >= var_threshold)

        print(f"\n  Prediction:")
        print(f"    Bands passing SNR only:      {snr_pass}/{n_bands} ({100*snr_pass/n_bands:.1f}%)")
        print(f"    Bands passing Variance only: {var_pass}/{n_bands} ({100*var_pass/n_bands:.1f}%)")
        print(f"    Bands passing BOTH:          {both_pass}/{n_bands} ({100*both_pass/n_bands:.1f}%) ✓")

        if both_pass == 0:
            print(f"\n  ⚠ WARNING: With these thresholds, 0 bands will be exported!")
            print(f"  ⚠ Try lower values:")
            print(f"     SNR:      {min(snr_values)*0.5:.2f} (below minimum)")
            print(f"     Variance: {min(var_values)*0.5:.2f} (below minimum)")

    # Filter bands
    print(f"\n4. Filtering bands...")
    clean_count = 0
    metrics = []

    for i in range(n_bands):
        is_clean = (snr_values[i] >= snr_threshold and
                   var_values[i] >= var_threshold)

        if is_clean:
            clean_count += 1

        metrics.append({
            'band_index': i,
            'wavelength': wavelengths[i],
            'snr': snr_values[i],
            'variance': var_values[i],
            'mean': mean_values[i],
            'is_clean': is_clean
        })

    print(f"   Clean bands: {clean_count}/{n_bands} ({100*clean_count/n_bands:.1f}%)")

    if clean_count == 0:
        print(f"\n   ✗ ERROR: No bands passed filtering!")
        print(f"\n   Your thresholds are too strict. Recommended fix:")
        print(f"   export_with_diagnostics(")
        print(f"       mode='manual',")
        print(f"       manual_snr={min(snr_values)*0.5:.2f},    # Below minimum SNR")
        print(f"       manual_var={min(var_values)*0.5:.2f}     # Below minimum variance")
        print(f"   )")
        return

    # Normalize
    print(f"\n5. Applying band-wise normalization...")
    preprocessor = HyperspectralPreprocessor(
        method='percentile',
        brightness_boost=True,
        band_wise=True,
        percentile_low=1,
        percentile_high=99
    )
    normalized_cube = preprocessor.preprocess(spectral_cube, fit_pca=False)

    # Export
    print(f"\n6. Exporting {clean_count} clean bands...")
    os.makedirs(output_dir, exist_ok=True)

    for metric in tqdm(metrics, desc="   Exporting"):
        if metric['is_clean']:
            i = metric['band_index']
            wl = metric['wavelength']
            filename = f"Band_{i+1:03d}_{wl:.1f}nm_normalized.png"
            output_path = os.path.join(output_dir, filename)

            band_8bit = (normalized_cube[i] * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(band_8bit).save(output_path)

    # Save report
    print(f"\n7. Saving report...")
    with open(os.path.join(output_dir, 'quality_report.txt'), 'w') as f:
        f.write("QUALITY REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Mode: {mode.upper()}\n")
        f.write(f"Total bands: {n_bands}\n")
        f.write(f"Clean bands: {clean_count}/{n_bands} ({100*clean_count/n_bands:.1f}%)\n")
        f.write(f"SNR threshold: {snr_threshold:.2f}\n")
        f.write(f"Variance threshold: {var_threshold:.2f}\n\n")

        f.write("DATA STATISTICS:\n")
        f.write(f"  SNR range: {min(snr_values):.2f} - {max(snr_values):.2f}\n")
        f.write(f"  Variance range: {min(var_values):.2f} - {max(var_values):.2f}\n\n")

        f.write("CLEAN BANDS:\n")
        f.write("-"*80 + "\n")
        for m in metrics:
            if m['is_clean']:
                f.write(f"Band {m['band_index']+1:3d}  {m['wavelength']:7.1f}nm  "
                       f"SNR:{m['snr']:6.2f}  Var:{m['variance']:8.1f}\n")

    # Save band list
    with open(os.path.join(output_dir, 'clean_bands_list.txt'), 'w') as f:
        f.write(f"# Clean bands: {clean_count}\n")
        f.write(f"# Threshold: SNR≥{snr_threshold:.2f}, Var≥{var_threshold:.2f}\n\n")
        for m in metrics:
            if m['is_clean']:
                f.write(f"{m['band_index']+1}, {m['wavelength']:.2f}\n")

        f.write("\n# Python (0-based):\nclean_indices = [\n")
        indices = [str(m['band_index']) for m in metrics if m['is_clean']]
        for i in range(0, len(indices), 10):
            f.write("    " + ", ".join(indices[i:i+10]) + ",\n")
        f.write("]\n")

    print(f"\n{'='*90}")
    print("EXPORT COMPLETE!")
    print(f"{'='*90}")
    print(f"\nOutput: {output_dir}/")
    print(f"Clean bands: {clean_count}/{n_bands}")
    print(f"Thresholds: SNR≥{snr_threshold:.2f}, Var≥{var_threshold:.2f}")


if __name__ == '__main__':
    # AUTO MODE (recommended)
    export_with_diagnostics(
        mode='auto',
        keep_percentage=75
    )

    # Or MANUAL MODE with diagnostics
    # export_with_diagnostics(
    #     mode='manual',
    #     manual_snr=1.0,
    #     manual_var=50.0
    # )

    # Or let it auto-suggest lenient thresholds
    # export_with_diagnostics(
    #     mode='manual'  # Will auto-suggest based on data
    # )
