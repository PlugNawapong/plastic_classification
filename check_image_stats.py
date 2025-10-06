"""
Quick diagnostic: Check actual SNR and variance values in your images.
This will help determine appropriate thresholds.
"""

import numpy as np
from PIL import Image
import glob
import json

# Load wavelengths
with open('training_dataset/header.json') as f:
    wavelengths = json.load(f)['wavelength (nm)']

# Load images
image_files = sorted(glob.glob('training_dataset/ImagesStack*.png'))[:20]  # First 20

print("="*90)
print("IMAGE STATISTICS ANALYSIS")
print("="*90)
print(f"\nAnalyzing first 20 bands to determine appropriate thresholds...\n")

snr_values = []
var_values = []
mean_values = []

print(f"{'Band':<6} {'Wavelength':<12} {'Min':<8} {'Max':<8} {'Mean':<10} {'Std':<10} {'Variance':<12} {'SNR':<8}")
print("-"*90)

for i, img_path in enumerate(image_files):
    img = np.array(Image.open(img_path))

    min_val = float(img.min())
    max_val = float(img.max())
    mean = float(np.mean(img))
    std = float(np.std(img))
    var = float(np.var(img))
    snr = mean / (std + 1e-10)

    snr_values.append(snr)
    var_values.append(var)
    mean_values.append(mean)

    wl = wavelengths[i]
    print(f"{i+1:<6} {wl:<12.1f} {min_val:<8.0f} {max_val:<8.0f} {mean:<10.2f} {std:<10.2f} {var:<12.2f} {snr:<8.2f}")

print("\n" + "="*90)
print("SUMMARY STATISTICS (first 20 bands)")
print("="*90)

print(f"\nSNR Statistics:")
print(f"  Min:    {min(snr_values):.2f}")
print(f"  Max:    {max(snr_values):.2f}")
print(f"  Mean:   {np.mean(snr_values):.2f}")
print(f"  Median: {np.median(snr_values):.2f}")

print(f"\nVariance Statistics:")
print(f"  Min:    {min(var_values):.2f}")
print(f"  Max:    {max(var_values):.2f}")
print(f"  Mean:   {np.mean(var_values):.2f}")
print(f"  Median: {np.median(var_values):.2f}")

print(f"\nMean Intensity Statistics:")
print(f"  Min:    {min(mean_values):.2f}")
print(f"  Max:    {max(mean_values):.2f}")
print(f"  Mean:   {np.mean(mean_values):.2f}")

print("\n" + "="*90)
print("RECOMMENDED THRESHOLDS")
print("="*90)

# Recommendations based on actual data
min_snr = min(snr_values)
min_var = min(var_values)
mean_snr = np.mean(snr_values)
mean_var = np.mean(var_values)

print(f"\nBased on your data, here are recommended thresholds:\n")

print("VERY LENIENT (keep ~95% of bands):")
print(f"  SNR:      {min_snr * 0.5:.2f}")
print(f"  Variance: {min_var * 0.5:.2f}")

print("\nLENIENT (keep ~85-90% of bands):")
print(f"  SNR:      {min_snr * 0.8:.2f}")
print(f"  Variance: {min_var * 0.8:.2f}")

print("\nMODERATE (keep ~70-80% of bands):")
print(f"  SNR:      {mean_snr * 0.6:.2f}")
print(f"  Variance: {mean_var * 0.6:.2f}")

print("\nSTRICT (keep ~50-60% of bands):")
print(f"  SNR:      {mean_snr * 0.9:.2f}")
print(f"  Variance: {mean_var * 0.9:.2f}")

print("\n" + "="*90)
print("EXAMPLE COMMANDS")
print("="*90)

print(f"""
# Very lenient (keep most bands):
python -c "
from export_bands_fixed import export_clean_bands
export_clean_bands(
    use_manual=True,
    manual_snr={min_snr * 0.5:.2f},
    manual_var={min_var * 0.5:.2f},
    output_dir='clean_very_lenient'
)
"

# Lenient:
python -c "
from export_bands_fixed import export_clean_bands
export_clean_bands(
    use_manual=True,
    manual_snr={min_snr * 0.8:.2f},
    manual_var={min_var * 0.8:.2f},
    output_dir='clean_lenient'
)
"

# Moderate:
python -c "
from export_bands_fixed import export_clean_bands
export_clean_bands(
    use_manual=True,
    manual_snr={mean_snr * 0.6:.2f},
    manual_var={mean_var * 0.6:.2f},
    output_dir='clean_moderate'
)
"
""")

print("\n" + "="*90)
print("KEY INSIGHT")
print("="*90)
print("""
If you're getting 0 exported bands, it's because:
1. Your SNR threshold is too high, OR
2. Your Variance threshold is too high, OR
3. BOTH conditions must be met (SNR AND Variance)

To fix: Lower BOTH thresholds based on the recommendations above.
""")
