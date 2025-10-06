# Preprocessing Verification - Simple Guide

## ✅ Updated: Focus on Last (Noisiest) Band

The preprocessing tools now show the **last band** after filtering, which is typically the noisiest band that still passed quality checks. This gives you a better view of how well the preprocessing handles challenging data.

## 🎯 Simple Tool: check_preprocessing.py

**Use this tool to verify preprocessing quality before training.**

### Quick Start

```bash
# 1. Install dependencies (if not already installed)
pip install numpy pillow matplotlib scipy

# 2. Run preprocessing check
python check_preprocessing.py

# Output: preprocessing_check_training_dataset.png
```

### What You'll See

**6-panel visualization showing:**

1. **Step 1: Raw Band** - Original noisy data (8-bit, 0-255)
2. **Step 2: After Band Filtering** - Shows which bands were kept
3. **Step 3: Median Denoising** - After noise removal
4. **Step 4: Band-wise Normalized** - Final preprocessed (0-1)
5. **Noise Removed** - Difference map (HOT COLORS = HIGH NOISE) ⭐ **Key panel!**
6. **Summary Statistics** - SNR improvement, noise metrics

### What to Look For

#### Panel 5: "Noise Removed" (Most Important!)

- **Hot colors (red/yellow/white)** = High noise removed ✓ Good!
- **Cool colors (blue/black)** = Low noise removed
- Check the "Mean" value: Higher = more noise was present

**Interpretation:**
- Mean noise > 5: Significant noise, denoising is helping
- Mean noise 2-5: Moderate noise, denoising still useful
- Mean noise < 2: Low noise, denoising minimal impact

#### SNR (Signal-to-Noise Ratio)

- **Raw SNR**: Quality before preprocessing
- **Denoised SNR**: After median filter
- **Final SNR**: After normalization

**Good signs:**
- SNR stays similar or increases
- SNR doesn't drop significantly

#### Compare Raw vs Normalized

- **Raw (Panel 1)**: May look dim or low contrast
- **Normalized (Panel 4)**: Should look clearer, better contrast
- Edges should be preserved, not blurry

## 🔧 Advanced: preprocessing_pipeline.py

For more detailed analysis and comparing different denoising methods.

```bash
python preprocessing_pipeline.py
```

**Output:** `preprocessing_steps_training.png`

**To compare denoising methods (optional):**

```bash
python -c "from preprocessing_pipeline import NoiseRemovalPreprocessor; p = NoiseRemovalPreprocessor('training_dataset'); p.compare_denoising_methods()"
```

**Output:** `denoising_comparison_training.png` showing median, gaussian, bilateral, and NLM filters.

## 📊 Check Different Datasets

### Training Dataset

```bash
python check_preprocessing.py training_dataset
```

### Inference Dataset

```bash
python check_preprocessing.py Inference_dataset1
python check_preprocessing.py Inference_dataset2
python check_preprocessing.py Inference_dataset3
```

**Output files:**
- `preprocessing_check_training_dataset.png`
- `preprocessing_check_Inference_dataset1.png`
- `preprocessing_check_Inference_dataset2.png`
- `preprocessing_check_Inference_dataset3.png`

## 🎓 Understanding the Preprocessing Steps

### Step 1: Load Raw Bands
- Loads all 458 spectral bands
- Each band is a grayscale image (8-bit, 0-255)
- Wavelength range: 450.5 - 853.6 nm

### Step 2: Filter Noisy Bands
- Calculates SNR for each band
- Keeps top 75% cleanest bands (~343 bands)
- Removes bands with poor signal quality
- **Result:** ~343 bands, wavelength range: 450.5 - 752.2 nm

### Step 3: Median Denoising
- Applies 3×3 median filter to each band
- Removes salt-and-pepper noise (random speckles)
- Preserves edges better than Gaussian blur
- **Why median?** Fast, effective, standard method

### Step 4: Band-wise Normalization
1. **Brightness boost:** Scale each band to full range (0-255)
2. **Percentile clipping:** Remove outliers (1st-99th percentile)
3. **Normalize:** Scale to [0, 1]

**Why band-wise?** Each wavelength has different brightness. Normalizing each band independently enhances contrast across all wavelengths.

## ✅ Quality Checklist

Before proceeding to training, verify:

- [ ] **Noise removed panel shows hot spots** - Denoising is working
- [ ] **SNR doesn't drop significantly** - Quality maintained
- [ ] **Normalized image looks clearer than raw** - Preprocessing improves quality
- [ ] **Edges are preserved** - Not too blurry
- [ ] **No artifacts or distortions** - Preprocessing not creating new problems

## 🚫 What NOT to Do

**Don't skip preprocessing verification!** This is critical:
- If preprocessing adds artifacts → Model learns wrong patterns
- If preprocessing removes too much → Model loses information
- If preprocessing is inconsistent → Poor generalization

## 📝 Next Steps

1. **Install dependencies**
   ```bash
   pip install numpy pillow matplotlib scipy
   ```

2. **Run preprocessing check**
   ```bash
   python check_preprocessing.py
   ```

3. **Open the output PNG file**
   - Check "Noise Removed" panel
   - Verify SNR improvement
   - Compare raw vs normalized

4. **If preprocessing looks good:**
   - Update your training script to use same preprocessing
   - Use: 75% band filtering, median denoising, band-wise normalization

5. **Train your model**
   ```bash
   python train.py
   ```

6. **Apply same preprocessing to inference**
   - Must match training preprocessing exactly!

## 🔍 Troubleshooting

### "Not enough noise removed"

Try stronger filtering:
```python
from preprocessing_pipeline import NoiseRemovalPreprocessor
proc = NoiseRemovalPreprocessor('training_dataset')
results = proc.process_with_steps(
    keep_percentage=60,  # More aggressive (was 75)
    denoise_method='median'
)
proc.visualize_steps(results)
```

### "Edges look blurry"

Try bilateral filter (edge-preserving):
```python
results = proc.process_with_steps(
    keep_percentage=75,
    denoise_method='bilateral'  # Edge-preserving
)
```

### "Too much information lost"

Keep more bands:
```python
results = proc.process_with_steps(
    keep_percentage=85,  # Less aggressive (was 75)
    denoise_method='median'
)
```

## 📁 File Reference

- **check_preprocessing.py** - Simple, focused preprocessing verification ⭐ **Use this first**
- **preprocessing_pipeline.py** - Detailed analysis with method comparison
- **visualize_prediction_noise.py** - Prediction noise analysis (after training)

## 💡 Key Takeaway

**The "Noise Removed" panel (hot colors) tells you if preprocessing is working.**

If you see hot spots → Denoising is removing noise ✓
If it's mostly dark → Either low noise or denoising not effective

---

**Ready?** Run `python check_preprocessing.py` and check the output!
