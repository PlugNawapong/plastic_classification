# Installation and Quick Start Guide

## Step 1: Install Dependencies

Run this command once:

```bash
pip install numpy pillow matplotlib scipy opencv-python torch torchvision
```

**Or** if you prefer to install from requirements.txt:

```bash
pip install -r requirements.txt
```

## Step 2: Test Installation

Run the test script to verify everything is installed correctly:

```bash
python test_preprocessing.py
```

**Expected output:**
```
✓ All dependencies installed
✓ training_dataset found
✓ header.json found
✓ Found 458 spectral bands
✓ Preprocessor initialized
  Wavelength range: 450.5 - 853.6 nm
  Number of bands: 458

[Processing steps...]

✓ Preprocessing completed successfully
✓ Step visualization saved to: preprocessing_steps_test.png
✓ Denoising comparison saved to: denoising_comparison_test.png

ALL TESTS PASSED!
```

## Step 3: Run Preprocessing Visualization

### Option A: Training Dataset

```bash
python preprocessing_pipeline.py
```

**This will:**
- Load training_dataset
- Filter noisy bands (keep top 75%)
- Apply median denoising
- Show step-by-step visualization
- Compare 4 denoising methods
- Save: `preprocessing_steps_training.png` and `denoising_comparison_training.png`

### Option B: Inference Dataset

```python
python -c "
from preprocessing_pipeline import NoiseRemovalPreprocessor

# Use Inference_dataset1
proc = NoiseRemovalPreprocessor('Inference_dataset1')

# Run pipeline
results = proc.process_with_steps(
    keep_percentage=75.0,
    denoise_method='median'
)

# Visualize
proc.visualize_steps(results, output_path='preprocessing_steps_inference1.png')
proc.compare_denoising_methods(output_path='denoising_comparison_inference1.png')
"
```

### Option C: Custom Settings

```python
from preprocessing_pipeline import NoiseRemovalPreprocessor

proc = NoiseRemovalPreprocessor('training_dataset')

# Try different settings
results = proc.process_with_steps(
    keep_percentage=60.0,        # More aggressive filtering
    denoise_method='bilateral'   # Edge-preserving denoising
)

proc.visualize_steps(results, output_path='custom_preprocessing.png')
```

## Step 4: Analyze Prediction Noise (Optional)

**Requirements:**
- Trained model (best_model.pth)
- Inference dataset

```bash
python visualize_prediction_noise.py
```

**This will:**
- Run inference on Inference_dataset1
- Identify noise in predictions
- Show where noise appears (background, edges, materials)
- Compare preprocessing methods
- Save: `noise_analysis_*.png` and `preprocessing_comparison_*.png`

## Troubleshooting

### Error: ModuleNotFoundError

**Solution:** Install dependencies

```bash
pip install numpy pillow matplotlib scipy opencv-python torch torchvision
```

### Error: training_dataset not found

**Solution:** Make sure you're in the correct directory

```bash
# Check current directory
pwd

# Should show: /Users/nawapong/Projects/plastic_classification

# List directories
ls
# Should show: training_dataset, Inference_dataset1, etc.
```

### Error: No module named 'cv2'

**Solution:** Install OpenCV

```bash
pip install opencv-python
```

### Visualization windows don't show

If you're running on a server without display, add `plt.ioff()` at the top of the script to save files without displaying:

```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()
```

### Warning: Band X not found

**Solution:** Some bands might be missing. This is OK - the script will skip them.

## Output Files

After running the scripts, you'll have:

### From test_preprocessing.py
- `preprocessing_steps_test.png` - Test visualization
- `denoising_comparison_test.png` - Test denoising comparison

### From preprocessing_pipeline.py
- `preprocessing_steps_training.png` - Full pipeline visualization
- `denoising_comparison_training.png` - 4 denoising methods comparison

### From visualize_prediction_noise.py
- `noise_analysis_ImageStack*.png` - Detailed noise breakdown
- `preprocessing_comparison_ImageStack*.png` - Method comparison

## Understanding the Outputs

### preprocessing_steps_*.png

Shows 6 panels:
1. **Raw Band** - Original data
2. **After Band Filtering** - Noisy bands removed
3. **Spatial Denoising** - Smoothed
4. **Normalized** - Final preprocessed
5. **Noise Removed** - Difference map (hot colors = noise)
6. **Quality Metrics** - SNR improvement

**What to look for:**
- Hot spots in difference map = high noise removed
- SNR should increase after processing
- Normalized image should be clearer than raw

### denoising_comparison_*.png

Shows 5 panels comparing:
1. Original (filtered)
2. Median filter
3. Gaussian filter
4. Bilateral filter
5. Non-local means

**What to choose:**
- **Median** - Best for salt-and-pepper noise (random dots)
- **Bilateral** - Best for edge preservation
- **NLM** - Best quality (slowest)

### noise_analysis_*.png

Shows 6 panels:
1. **Prediction Mask** - Classification result
2. **Confidence Map** - Model certainty
3. **Noise Pixels** - Detected noise (red)
4. **Noise Location** - Where noise appears
5. **Low Confidence** - Uncertain regions
6. **Statistics** - Noise counts

**What to look for:**
- Dark red in "Noise Location" = noise in materials (BAD!)
- Yellow = noise at edges (minor issue)
- Light red = noise in background (OK)

## Next Steps

1. **Install dependencies** (Step 1)
2. **Run test** (Step 2)
3. **Visualize preprocessing** (Step 3)
4. **Choose best denoising method** based on visualizations
5. **Update training/inference** to use chosen method
6. **Retrain model** with clean preprocessing
7. **Analyze prediction noise** to verify improvement

## Quick Reference

```bash
# Install (once)
pip install numpy pillow matplotlib scipy opencv-python torch torchvision

# Test (verify installation)
python test_preprocessing.py

# Visualize preprocessing (training data)
python preprocessing_pipeline.py

# Visualize preprocessing (inference data)
python -c "from preprocessing_pipeline import NoiseRemovalPreprocessor; p = NoiseRemovalPreprocessor('Inference_dataset1'); r = p.process_with_steps(); p.visualize_steps(r)"

# Analyze prediction noise (requires trained model)
python visualize_prediction_noise.py
```

---

**Need help?** Check [NOISE_REMOVAL_GUIDE.md](NOISE_REMOVAL_GUIDE.md) for detailed documentation.
