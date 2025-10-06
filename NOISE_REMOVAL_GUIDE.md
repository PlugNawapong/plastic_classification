# Noise Removal and Preprocessing Visualization Guide

## Overview

This guide helps you identify and remove noise from predictions. The noise you're seeing (spreading on background and between materials) can come from:

1. **Noisy spectral bands** - Some wavelengths have poor signal quality
2. **Lack of spatial denoising** - Salt-and-pepper noise in individual bands
3. **Poor normalization** - Artifacts from normalization process

## Quick Start

### Step 1: Visualize Preprocessing Steps

See exactly what happens at each preprocessing step:

```bash
python preprocessing_pipeline.py
```

**This will:**
- ✓ Load a sample from training_dataset
- ✓ Filter out noisy bands (keep top 75%)
- ✓ Apply spatial denoising (median filter)
- ✓ Normalize band-wise
- ✓ Save visualization showing all steps side-by-side

**Output:** `preprocessing_steps_ImageStack000.png`

**What to look for:**
- **Step 1 (Raw):** Original noisy data
- **Step 3 (Denoised):** Should be smoother, less speckled
- **Difference map:** Shows exactly what noise was removed (hot spots = noise)

### Step 2: Compare Denoising Methods

Find the best denoising method for your data:

```bash
python -c "
from preprocessing_pipeline import NoiseRemovalPreprocessor
proc = NoiseRemovalPreprocessor('training_dataset')
proc.compare_denoising_methods(
    sample_name='ImageStack000',
    output_path='denoising_comparison.png'
)
"
```

**Methods compared:**
1. **Median filter** - Best for salt-and-pepper noise (random speckles)
2. **Gaussian filter** - Best for Gaussian noise (smooth noise)
3. **Bilateral filter** - Edge-preserving (keeps material boundaries sharp)
4. **Non-local means (NLM)** - Highest quality (slowest)

**What to choose:**
- If you see random dots → Use **median**
- If you see smooth grain → Use **gaussian**
- If edges are blurry → Use **bilateral**
- Best quality → Use **nlm** (but slower)

### Step 3: Analyze Prediction Noise

Visualize where noise appears in predictions:

```bash
python visualize_prediction_noise.py
```

**This will:**
- ✓ Run inference on Inference_dataset1
- ✓ Identify noise pixels (isolated, inconsistent predictions)
- ✓ Show where noise appears: background, edges, or materials
- ✓ Show confidence map (low confidence = likely noise)
- ✓ Compare different preprocessing methods

**Output:**
- `noise_analysis_ImageStack000.png` - Detailed noise breakdown
- `preprocessing_comparison_ImageStack000.png` - Method comparison

**What to look for:**
- **Red pixels** in noise map = noise
- **Noise location:** Most noise should be at edges/background, NOT in materials
- **Confidence map:** Green = confident, Red = uncertain (likely noise)

## Detailed Workflow

### Workflow 1: Identify Noise Source

```python
from preprocessing_pipeline import NoiseRemovalPreprocessor

proc = NoiseRemovalPreprocessor('training_dataset')

# Process with full pipeline
results = proc.process_with_steps(
    sample_name='ImageStack000',
    keep_percentage=75.0,      # Keep top 75% cleanest bands
    denoise_method='median'     # Use median filter
)

# Visualize
proc.visualize_steps(results)
```

**Check the difference map:**
- **Hot spots (bright areas)** = High noise removed
- **Dark areas** = Low noise (clean data)
- If difference map is mostly dark → Denoising not helping much
- If difference map has bright speckles → Denoising is removing salt-and-pepper noise

### Workflow 2: Compare Preprocessing Impact on Predictions

```python
from visualize_prediction_noise import PredictionNoiseAnalyzer

analyzer = PredictionNoiseAnalyzer(
    model_path='best_model.pth',
    dataset_path='Inference_dataset1'
)

# Compare preprocessing methods
analyzer.compare_preprocessing_methods(
    sample_name='ImageStack000',
    output_path='comparison.png'
)
```

**Read the summary:**
```
NOISE REDUCTION SUMMARY
==================================================================================

No denoising:
  Noise pixels:    15,432 (2.45%)
  Background:      8,234
  Edges:           5,123
  Materials:       2,075     ← BAD! Noise in materials

Median filter:
  Noise pixels:    8,921 (1.42%)
  Background:      5,432
  Edges:           3,012
  Materials:       477       ← GOOD! Less noise in materials

Bilateral filter:
  Noise pixels:    7,234 (1.15%)
  Background:      4,123
  Edges:           2,876
  Materials:       235       ← BEST! Minimal noise in materials
```

**Goal:** Minimize "Materials" noise (noise between/inside materials)

### Workflow 3: Tune Band Filtering

Adjust how many bands to keep:

```python
from preprocessing_pipeline import NoiseRemovalPreprocessor

proc = NoiseRemovalPreprocessor('training_dataset')

# Try different percentages
for keep_pct in [50, 60, 75, 85, 95]:
    results = proc.process_with_steps(
        sample_name='ImageStack000',
        keep_percentage=keep_pct,
        denoise_method='median'
    )

    print(f"\nKeeping {keep_pct}%:")
    print(f"  Bands used: {len(results['clean_indices'])}/{proc.n_bands}")
    print(f"  Wavelength range: {proc.wavelengths[results['clean_indices'][0]]:.1f} - "
          f"{proc.wavelengths[results['clean_indices'][-1]]:.1f} nm")
```

**What to choose:**
- **50-60%** - Very clean, but might lose useful information
- **75%** - Good balance (recommended)
- **85-95%** - More information, but more noise

**Rule of thumb:** Start with 75%, decrease if noise is still high

## Understanding the Visualizations

### Preprocessing Steps Visualization

**6 panels showing:**

1. **Step 1: Raw Band** (top-left)
   - Original 8-bit data from camera
   - Shows natural noise and artifacts
   - Range: 0-255

2. **Step 2: After Band Filtering** (top-middle)
   - Same as raw (filtering happens across bands, not spatially)
   - This step removes entire noisy bands from the stack

3. **Step 3: Spatial Denoising** (top-right)
   - After applying median/gaussian/bilateral/NLM filter
   - Should look smoother than raw
   - Speckles should be reduced

4. **Step 4: Normalized** (bottom-left)
   - After band-wise normalization
   - Range: 0-1
   - Contrast should be enhanced

5. **Noise Removed by Denoising** (bottom-middle)
   - Absolute difference: |Raw - Denoised|
   - **Hot colors (red/yellow)** = High noise removed
   - **Cool colors (blue/black)** = Low noise
   - This shows WHERE the denoising is working

6. **Quality Metrics** (bottom-right)
   - SNR: Signal-to-noise ratio (higher = better)
   - Variance: Information content (too low = no info, too high = noise)
   - Improvement: How much SNR increased after processing

### Prediction Noise Visualization

**6 panels showing:**

1. **Prediction Mask** (top-left)
   - Final classification result
   - Each color = different plastic class
   - Look for: scattered pixels, noisy edges

2. **Confidence Map** (top-middle)
   - How confident the model is for each pixel
   - **Green** = High confidence (>0.8)
   - **Yellow** = Medium confidence (0.5-0.8)
   - **Red** = Low confidence (<0.5)
   - Low confidence areas are likely noise

3. **Noise Pixels (Red)** (top-right)
   - Detected noise based on morphological analysis
   - **Red** = Isolated pixels (noise)
   - **Gray** = Consistent regions (clean)
   - Shows total noise count and percentage

4. **Noise Location** (bottom-left)
   - **Light Red** = Noise in background
   - **Yellow** = Noise at edges between materials
   - **Dark Red** = Noise inside materials (most problematic!)
   - Goal: Minimize dark red pixels

5. **Low Confidence Regions** (bottom-middle)
   - Prediction overlaid with red tint on low-confidence areas
   - These regions are uncertain → likely to be noise
   - Should correlate with noise map

6. **Statistics or Ground Truth** (bottom-right)
   - If ground truth available: shows errors in magenta
   - Otherwise: shows detailed statistics table

## Tuning Guide

### Problem: Too much noise in background

**Solution:** Increase band filtering, use stronger denoising

```python
results = proc.process_with_steps(
    sample_name='ImageStack000',
    keep_percentage=60,         # ← Stricter (was 75)
    denoise_method='bilateral'  # ← Stronger (was median)
)
```

### Problem: Noise spreading between materials

**Solution:** Use edge-preserving denoising

```python
results = proc.process_with_steps(
    sample_name='ImageStack000',
    keep_percentage=75,
    denoise_method='bilateral'  # ← Edge-preserving
)
```

### Problem: Materials look blurry/merged

**Solution:** Reduce denoising strength or use NLM

```python
results = proc.process_with_steps(
    sample_name='ImageStack000',
    keep_percentage=75,
    denoise_method='nlm'  # ← Preserves details better
)
```

### Problem: Edges are jagged/pixelated

**Solution:** Add post-processing morphological operations

```python
from scipy.ndimage import binary_opening, binary_closing
import numpy as np

# After getting prediction mask
prediction_mask = results['prediction_mask']

# For each class, smooth edges
smoothed_mask = np.zeros_like(prediction_mask)
for class_id in np.unique(prediction_mask):
    class_mask = (prediction_mask == class_id)

    # Remove small holes
    class_mask = binary_closing(class_mask, structure=np.ones((5, 5)))

    # Remove small objects
    class_mask = binary_opening(class_mask, structure=np.ones((3, 3)))

    smoothed_mask[class_mask] = class_id
```

## Best Practices

### For Training

Use **aggressive** preprocessing to ensure clean training data:

```python
results = proc.process_with_steps(
    sample_name='ImageStack000',
    keep_percentage=60,         # Keep only 60% cleanest bands
    denoise_method='bilateral'  # Edge-preserving
)
```

### For Inference

Use **same** preprocessing as training:

```python
# If you trained with 75% bands and median filter
results = proc.process_with_steps(
    sample_name='ImageStack000',
    keep_percentage=75,      # SAME as training
    denoise_method='median'  # SAME as training
)
```

**⚠️ IMPORTANT:** Preprocessing must match between training and inference!

### Recommended Pipeline

Based on the paper and your improvements:

```python
results = proc.process_with_steps(
    sample_name='ImageStack000',
    keep_percentage=75,         # Keep top 75%
    denoise_method='median'     # Fast and effective
)

# The normalize_bandwise already does:
# ✓ Brightness boost (scale to full range)
# ✓ Percentile clipping (1st-99th percentile)
# ✓ Band-wise normalization (per-band)
```

## Files Reference

### Main Tools

1. **[preprocessing_pipeline.py](preprocessing_pipeline.py)** - Core preprocessing with visualization
   - `NoiseRemovalPreprocessor` class
   - Methods: `process_with_steps()`, `visualize_steps()`, `compare_denoising_methods()`

2. **[visualize_prediction_noise.py](visualize_prediction_noise.py)** - Prediction noise analysis
   - `PredictionNoiseAnalyzer` class
   - Methods: `predict_with_preprocessing()`, `visualize_noise()`, `compare_preprocessing_methods()`

### Configuration Files

- **[labels.json](labels.json)** - Class definitions and colors
- **training_dataset/header.json** - Wavelength calibration data

### Output Files

- `preprocessing_steps_*.png` - Step-by-step preprocessing visualization
- `denoising_comparison_*.png` - Denoising methods comparison
- `noise_analysis_*.png` - Prediction noise breakdown
- `preprocessing_comparison_*.png` - Impact of preprocessing on predictions

## Troubleshooting

### ModuleNotFoundError

```bash
pip install numpy pillow matplotlib scipy opencv-python torch torchvision
```

### Model not found error

Make sure you have trained the model first:

```bash
python train.py
```

This creates `best_model.pth`

### All bands filtered out

Lower the `keep_percentage`:

```python
results = proc.process_with_steps(
    sample_name='ImageStack000',
    keep_percentage=90,  # ← More permissive
    denoise_method='median'
)
```

### Denoising too slow

Use faster method:

```python
results = proc.process_with_steps(
    sample_name='ImageStack000',
    keep_percentage=75,
    denoise_method='median'  # ← Fastest (not 'nlm')
)
```

## Next Steps

1. **Run preprocessing pipeline** to see steps
   ```bash
   python preprocessing_pipeline.py
   ```

2. **Compare denoising methods** to find best one
   ```bash
   python -c "from preprocessing_pipeline import NoiseRemovalPreprocessor; p = NoiseRemovalPreprocessor('training_dataset'); p.compare_denoising_methods('ImageStack000')"
   ```

3. **Analyze prediction noise** to verify improvement
   ```bash
   python visualize_prediction_noise.py
   ```

4. **Update training** with chosen preprocessing
   - Edit [train.py](train.py) to use your chosen settings
   - Retrain model with clean data

5. **Update inference** with same preprocessing
   - Edit [inference.py](inference.py) to match training settings
   - Run inference on test datasets

---

**Summary:** The noise in your predictions likely comes from noisy bands and lack of spatial denoising. Use the visualization tools to identify the noise source, then apply appropriate filtering and denoising. The key is to use **consistent preprocessing** between training and inference.
