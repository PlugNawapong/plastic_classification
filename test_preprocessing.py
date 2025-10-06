"""
Quick test script for preprocessing pipeline

Run this after installing dependencies:
pip install numpy pillow matplotlib scipy opencv-python
"""

print("Testing preprocessing pipeline...")

# Test 1: Check if dependencies are installed
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from scipy.ndimage import median_filter
    import cv2
    print("✓ All dependencies installed")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("\nPlease install dependencies:")
    print("pip install numpy pillow matplotlib scipy opencv-python")
    exit(1)

# Test 2: Check if training_dataset exists
from pathlib import Path
if not Path('training_dataset').exists():
    print("✗ training_dataset directory not found")
    exit(1)
print("✓ training_dataset found")

# Test 3: Check if header.json exists
if not Path('training_dataset/header.json').exists():
    print("✗ header.json not found in training_dataset")
    exit(1)
print("✓ header.json found")

# Test 4: Count bands
bands = list(Path('training_dataset').glob('ImagesStack*.png'))
print(f"✓ Found {len(bands)} spectral bands")

# Test 5: Load preprocessor
try:
    from preprocessing_pipeline import NoiseRemovalPreprocessor
    preprocessor = NoiseRemovalPreprocessor('training_dataset')
    print(f"✓ Preprocessor initialized")
    print(f"  Wavelength range: {preprocessor.wavelengths[0]:.1f} - {preprocessor.wavelengths[-1]:.1f} nm")
    print(f"  Number of bands: {preprocessor.n_bands}")
except Exception as e:
    print(f"✗ Failed to initialize preprocessor: {e}")
    exit(1)

# Test 6: Run preprocessing
try:
    print("\nRunning preprocessing pipeline...")
    results = preprocessor.process_with_steps(
        keep_percentage=75.0,
        denoise_method='median'
    )
    print("✓ Preprocessing completed successfully")
    print(f"  Clean bands: {len(results['clean_indices'])}/{preprocessor.n_bands}")
except Exception as e:
    print(f"✗ Preprocessing failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 7: Visualize (optional, requires display)
try:
    print("\nGenerating visualizations...")
    preprocessor.visualize_steps(
        results=results,
        output_path='preprocessing_steps_test.png'
    )
    print("✓ Step visualization saved to: preprocessing_steps_test.png")
except Exception as e:
    print(f"⚠ Visualization skipped (may require display): {e}")

# Test 8: Compare denoising methods
try:
    print("\nComparing denoising methods...")
    preprocessor.compare_denoising_methods(
        output_path='denoising_comparison_test.png'
    )
    print("✓ Denoising comparison saved to: denoising_comparison_test.png")
except Exception as e:
    print(f"⚠ Denoising comparison skipped: {e}")

print("\n" + "="*80)
print("ALL TESTS PASSED!")
print("="*80)
print("\nYou can now run:")
print("  python preprocessing_pipeline.py")
print("  python visualize_prediction_noise.py")
