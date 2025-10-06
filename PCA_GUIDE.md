# PCA Band Selection Guide

## Overview

This guide explains how to use PCA (Principal Component Analysis) for hyperspectral band selection and noise reduction in your plastic classification project.

## What Does PCA Do?

### 1. **Dimensionality Reduction**
- Reduces 459 spectral bands to fewer essential components (e.g., 50-200)
- Retains 95-99% of variance with much fewer dimensions
- Speeds up training and inference significantly

### 2. **Noise Reduction**
PCA reduces noise in **multiple ways**:

#### **Variance-Based Filtering**
- PCA components are ordered by variance: PC1 > PC2 > ... > PCn
- High-variance components = **signal** (meaningful patterns)
- Low-variance components = **noise** (random fluctuations)
- By keeping only top components, we automatically filter out noisy bands

#### **Spectral Smoothing**
- PCA reconstruction averages correlated bands
- Acts as spectral filtering
- Removes high-frequency noise while preserving structure

#### **Reduced Overfitting**
- Fewer dimensions → less overfitting
- Better generalization → smoother predictions
- Less sensitivity to noisy training examples

### 3. **When Does PCA Help Reduce Noise?**

✅ **PCA HELPS when:**
- Spectral bands are highly correlated (redundant information)
- Dataset has noise in high-frequency components
- Model tends to overfit (too many features vs samples)
- You see noisy/spotty predictions after classification

❌ **PCA MAY NOT HELP when:**
- All bands contain unique, independent information (rare in hyperspectral)
- Critical features are in low-variance components (rare)
- Preprocessing already removed most noise
- Data is already very clean

## Quick Start

### Step 1: Analyze Your Data

First, check if PCA will benefit your dataset:

```bash
python analyze_pca_benefits.py
```

This will:
- Calculate band correlation matrix
- Estimate noise levels
- Determine optimal number of components
- Provide clear recommendations
- Generate `pca_analysis.png` with visualizations

**Example output:**
```
Band Correlation Statistics:
  Mean correlation: 0.847
  High correlation (>0.9): 67.3%
  → HIGH correlation: PCA will be very effective! ✓

Intrinsic Dimensionality:
  Components for 99% variance: 142 (30.9% of original)
  → PCA can reduce dimensions by 69% with minimal loss! ✓
  → Strongly recommended to use PCA ✓✓✓
```

### Step 2: Train with PCA Comparison

Compare different PCA configurations:

```bash
# Compare baseline vs PCA with different component counts
python train_with_pca.py --compare --compare-configs 50 100 150 200
```

This will:
1. Train model WITHOUT PCA (baseline)
2. Train models WITH PCA (50, 100, 150, 200 components)
3. Compare validation accuracy
4. Save best configuration

**Example output:**
```
TRAINING COMPARISON SUMMARY
================================================================================
Config               Bands      Best Val Acc    Best Epoch
--------------------------------------------------------------------------------
No PCA               459            94.23%            42
PCA-50               50             93.87%            38
PCA-100              100            95.12%            35
PCA-150              150            95.45%            33
PCA-200              200            95.18%            34

BEST CONFIGURATION: PCA-150
  Validation Accuracy: 95.45%
  Number of Bands: 150
  Reduction: 67.3%
```

### Step 3: Use Best Configuration for Inference

After finding the best PCA configuration, use it for inference:

```python
from pca_band_selection import PCABandSelector
from pca_band_selection import load_normalized_hypercube

# Load PCA model (trained during training)
pca_selector = PCABandSelector()
pca_selector.load_model('checkpoints_pca_150/pca_model.pkl')

# Load test hypercube (normalized)
hypercube, wavelengths = load_normalized_hypercube('test_dataset')

# Apply PCA transformation
reduced_cube = pca_selector.transform(hypercube)

# Now use reduced_cube for inference with your model
# Shape: (150, height, width) instead of (459, height, width)
```

## Detailed Usage

### Manual PCA Training (Single Configuration)

Train with specific number of components:

```bash
# No PCA (baseline)
python train_with_pca.py

# With 150 components
python train_with_pca.py --pca-components 150
```

### Advanced PCA Analysis

For detailed PCA analysis and visualizations:

```python
from pca_band_selection import PCABandSelector, load_normalized_hypercube

# Load data
hypercube, wavelengths = load_normalized_hypercube('training_dataset')

# Fit PCA (auto-select components for 99% variance)
pca = PCABandSelector(n_components=None, variance_threshold=0.99)
pca.fit(hypercube, wavelengths)

# Visualizations
pca.visualize_variance_explained('variance.png')
pca.visualize_principal_components('components.png')
pca.visualize_noise_reduction(hypercube, reduced_cube, 'noise.png')

# Save model
pca.save_model('pca_model.pkl')
```

### Understanding the Visualizations

**1. `pca_analysis.png`** - Complete analysis with:
   - Band correlation heatmap (shows redundancy)
   - Correlation distribution (high = good for PCA)
   - Variance per band (which bands are informative)
   - Individual PC variance (how much each component explains)
   - Cumulative variance (how many components needed)
   - Recommendation summary

**2. `pca_variance_explained.png`**:
   - Left: Variance explained by each component
   - Right: Cumulative variance curve
   - Shows: How many components capture 95%, 99% variance

**3. `pca_components.png`**:
   - Shows which original bands contribute to each PC
   - Helps interpret what each component represents
   - Wavelength-based visualization

**4. `pca_noise_reduction.png`**:
   - Original band vs PCA reconstructed
   - Noise removed by PCA (difference map)
   - Intensity histogram comparison

## Does PCA Reduce Noise in Predictions?

### YES! Here's how:

#### 1. **Removes Low-Variance Noise**
```
Original 459 bands = Signal + Noise
PCA transformation:
- PC 1-150: High variance (mostly signal)
- PC 151-459: Low variance (mostly noise) ← DISCARDED
Result: Cleaner features for classifier
```

#### 2. **Spectral Denoising Effect**
```
Each PCA component = weighted combination of all bands
High correlation → smooth averaging
Result: High-frequency noise is smoothed out
```

#### 3. **Reduces Prediction Noise**
```
Fewer features → Less overfitting → Smoother predictions
Before PCA: Spotty, noisy classification map
After PCA: Smooth, clean boundaries
```

### Example Scenario

**Without PCA (459 bands):**
- Training accuracy: 96.5%
- Validation accuracy: 94.2%
- Prediction map: Noisy with salt-and-pepper artifacts
- Inference time: 450ms

**With PCA (150 components, 99% variance):**
- Training accuracy: 95.8%
- Validation accuracy: 95.4% ← Better generalization!
- Prediction map: Smoother with cleaner boundaries
- Inference time: 180ms ← 60% faster!

## Best Practices

### 1. **Always Compare**
Don't assume PCA will help - compare:
```bash
python train_with_pca.py --compare
```

### 2. **Choose Components Based on Variance**
- **95% variance**: Aggressive reduction, some info loss
- **99% variance**: Balanced (recommended)
- **99.9% variance**: Conservative, minimal loss

### 3. **Monitor Reconstruction Quality**
```python
# Check what information is lost
error = pca.calculate_reconstruction_error(original_cube, reduced_cube)
print(f"Reconstruction quality: {error['reconstruction_quality']*100:.2f}%")
# Aim for >95% reconstruction quality
```

### 4. **Consider Your Goals**

**Prioritize Accuracy:**
- Use more components (99.9% variance)
- Compare carefully with baseline
- Accept longer training time

**Prioritize Speed:**
- Use fewer components (95% variance)
- Accept small accuracy drop
- Much faster inference

**Prioritize Noise Reduction:**
- Use moderate components (99% variance)
- Check prediction smoothness
- May improve generalization

## Integration with Existing Pipeline

### Modify Your Workflow:

**Before:**
```
Load bands → Normalize → Train model → Predict
```

**After (with PCA):**
```
Load bands → Normalize → Fit PCA → Transform → Train model → Predict
                          ↓
                   Save PCA model
                          ↓
                   Use same PCA for inference
```

### Update Inference Code:

```python
# Load PCA model (once)
pca = PCABandSelector()
pca.load_model('checkpoints_pca_150/pca_model.pkl')

# For each test image:
hypercube = load_test_hypercube('test_dataset')
reduced_cube = pca.transform(hypercube)

# Use reduced_cube with your trained model
predictions = model.predict(reduced_cube)
```

## Troubleshooting

### Issue: PCA doesn't improve accuracy
**Solution:** Your bands may have unique information. Use baseline model.

### Issue: Too much information loss
**Solution:** Increase components or use higher variance threshold (99.9%).

### Issue: Slower training
**Solution:** This shouldn't happen - check implementation. PCA should speed up training.

### Issue: Different PCA results each time
**Solution:** Set random seed before fitting PCA:
```python
import numpy as np
np.random.seed(42)
```

## Summary

1. **Run analysis first**: `python analyze_pca_benefits.py`
2. **Compare configurations**: `python train_with_pca.py --compare`
3. **Choose best config** based on accuracy/speed tradeoff
4. **Use in production** with saved PCA model

PCA is particularly effective for hyperspectral data because:
- ✓ Spectral bands are highly correlated (redundant)
- ✓ Noise is often in low-variance components
- ✓ Dimensionality reduction improves generalization
- ✓ Faster inference with minimal accuracy loss

**Expected results for your 459-band dataset:**
- Reduction to ~100-200 bands (60-80% reduction)
- 95-99% variance retained
- Comparable or better validation accuracy
- Smoother, less noisy predictions
- 60-80% faster inference
