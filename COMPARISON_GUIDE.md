# Preprocessing Comparison Guide

## Updated: Compare Denoising Impact After Normalization

The preprocessing checker now shows **both paths** side-by-side:
- **Path 1 (RED):** Band Filtering → Normalization (NO denoising)
- **Path 2 (GREEN):** Band Filtering → Denoising → Normalization

This answers the question: **"Does denoising still matter after normalization?"**

## Quick Start

```bash
python check_preprocessing.py
```

**Output:** `preprocessing_check_training_dataset.png` (8 panels)

## Understanding the 8-Panel Visualization

### Top Row: Preprocessing Steps

1. **Panel 1: Raw Band**
   - Original 8-bit data (0-255)
   - Shows natural noise and artifacts
   - This is the last band after filtering (noisiest)

2. **Panel 2: After Band Filtering**
   - Same as raw for this specific band
   - Shows that this band passed quality checks
   - (115 noisy bands were removed from the stack)

3. **Panel 3: Median Denoising**
   - After applying 3×3 median filter
   - Should look smoother than Panel 1
   - Salt-and-pepper noise reduced

4. **Panel 4: Noise Removed**
   - Difference: |Panel 2 - Panel 3|
   - **HOT COLORS = HIGH NOISE REMOVED**
   - This shows noise BEFORE normalization

### Bottom Row: Comparison After Normalization

5. **Panel 5: Normalized WITHOUT Denoise** (RED title)
   - Raw → Filter → Normalize (skip denoising)
   - Shows what you get if you don't denoise
   - Range: [0, 1]

6. **Panel 6: Normalized WITH Denoise** (GREEN title)
   - Raw → Filter → Denoise → Normalize
   - Shows what you get with denoising
   - Range: [0, 1]

7. **Panel 7: Difference After Normalization** ⭐ **KEY PANEL**
   - Difference: |Panel 5 - Panel 6|
   - **HOT COLORS = DENOISING STILL MATTERS**
   - This shows if denoising helps AFTER normalization
   - Range: [0, 0.2]

8. **Panel 8: Summary Statistics**
   - SNR comparisons
   - Noise impact metrics
   - Recommendation

## How to Interpret Results

### Scenario 1: Denoising Matters

**Panel 7 shows HOT COLORS (red/yellow/white):**
- Mean difference > 0.01
- Visual difference between Panel 5 and Panel 6
- Recommendation: **USE denoising in your pipeline**

**Why?** Even after normalization, the noise is still affecting the final preprocessed data.

### Scenario 2: Denoising Minimal Impact

**Panel 7 mostly DARK (blue/black):**
- Mean difference < 0.01
- Panel 5 and Panel 6 look very similar
- Recommendation: **Normalization alone may be sufficient**

**Why?** Normalization (especially percentile clipping and brightness boost) already removes most of the noise impact.

### Scenario 3: Mixed Results

**Panel 7 has some hot spots, some dark areas:**
- Mean difference ~0.01
- Noise visible in certain regions
- Recommendation: **Use denoising for best quality**

**Why?** Denoising helps in noisy regions, doesn't hurt in clean regions.

## Reading the Metrics

### Console Output Example

```
Quality Metrics:
  Raw SNR:              1.23
  Denoised SNR:         1.45

After Normalization:
  WITHOUT denoise SNR:  2.15
  WITH denoise SNR:     2.34

Noise Impact:
  Before normalize:     3.45 (mean diff)
  After normalize:      0.0234 (mean diff)
  Reduction:            72.3%
```

**Interpretation:**

1. **Raw vs Denoised SNR**
   - Denoised SNR should be ≥ Raw SNR
   - Increase of ~20% is good (1.23 → 1.45)

2. **After Normalization SNR**
   - Both increase due to normalization
   - Compare WITH vs WITHOUT denoise
   - Larger gap = denoising still helping

3. **Noise Impact**
   - **Before normalize:** 3.45 (raw pixel values, 0-255 scale)
   - **After normalize:** 0.0234 (normalized values, 0-1 scale)
   - **Reduction:** 72.3% means normalization reduces noise visibility

### Decision Tree

```
Check Panel 7 (Difference After Normalization)
│
├─ Mean > 0.02 → SIGNIFICANT impact
│  └─ ✓ Use denoising (strong recommendation)
│
├─ Mean 0.01-0.02 → MODERATE impact
│  └─ ✓ Use denoising (recommended)
│
├─ Mean 0.005-0.01 → SMALL impact
│  └─ • Optional, depends on computational budget
│
└─ Mean < 0.005 → MINIMAL impact
   └─ • Normalization alone likely sufficient
```

## Visual Comparison Checklist

Compare Panel 5 (NO denoise) vs Panel 6 (WITH denoise):

- [ ] **Edges:** Are edges sharper in Panel 6?
- [ ] **Texture:** Is texture smoother in Panel 6?
- [ ] **Speckles:** Are random dots reduced in Panel 6?
- [ ] **Contrast:** Is contrast similar in both?
- [ ] **Panel 7:** Are there hot spots showing difference?

**If you check 3+ boxes → Denoising is helping**

## Example Scenarios

### Example 1: High Noise Data

```
Panel 4 (Noise Removed): Mean = 8.5 (HOT colors everywhere)
Panel 7 (Diff After Norm): Mean = 0.045 (HOT colors visible)

Conclusion: High noise before normalization, and denoising
still makes a difference after normalization.
→ Definitely use denoising!
```

### Example 2: Low Noise Data

```
Panel 4 (Noise Removed): Mean = 2.1 (Mostly dark, some spots)
Panel 7 (Diff After Norm): Mean = 0.003 (Mostly dark)

Conclusion: Low noise data, normalization handles it well.
→ Denoising optional, normalization sufficient.
```

### Example 3: Normalization Removes Most Noise

```
Panel 4 (Noise Removed): Mean = 5.5 (Moderate hot colors)
Panel 7 (Diff After Norm): Mean = 0.008 (Mostly dark, few spots)

Conclusion: Denoising removes noise, but normalization
(percentile clipping + brightness boost) also removes most of it.
→ Denoising provides small additional benefit.
```

## Recommendations by Application

### For Training (Most Important)

**If Panel 7 mean > 0.01:**
```python
# Use full preprocessing pipeline
1. Filter bands (75%)
2. Median denoise
3. Band-wise normalize
```

**If Panel 7 mean < 0.01:**
```python
# Simplified pipeline may work
1. Filter bands (75%)
2. Band-wise normalize (skip denoising)
```

### For Inference (Must Match Training)

**CRITICAL:** Use the SAME preprocessing as training!

If you trained WITH denoising → Inference MUST use denoising
If you trained WITHOUT denoising → Inference MUST NOT use denoising

**Mismatch = Poor performance!**

## Technical Details

### Why Normalization Reduces Noise

Band-wise normalization has built-in noise reduction:

1. **Brightness Boost**
   - Scales each band to full range [0, 255]
   - Enhances low-contrast bands
   - Makes signal more prominent

2. **Percentile Clipping**
   - Clips at 1st and 99th percentile
   - Removes extreme outliers (noise spikes)
   - **This is a form of noise removal!**

3. **Final Normalization**
   - Scales to [0, 1]
   - Standardizes dynamic range

**Combined effect:** Percentile clipping + brightness boost can remove significant noise, reducing the additional benefit of median filtering.

### When Denoising Still Helps

Even with percentile clipping, denoising helps when:

1. **Spatial noise patterns**
   - Salt-and-pepper noise (random dots)
   - Gaussian noise (smooth grain)
   - These affect neighboring pixels

2. **Percentile clipping limitations**
   - Only removes extreme values
   - Doesn't smooth spatial patterns
   - Median filter better for spatial coherence

3. **Edge preservation**
   - Denoising can preserve edges while removing noise
   - Percentile clipping works globally, not spatially

## Next Steps

1. **Run the checker**
   ```bash
   python check_preprocessing.py
   ```

2. **Open the output PNG**
   - Focus on Panel 7 (bottom row, third panel)
   - Check the "Difference" hot spots

3. **Read the conclusion**
   - Console output will recommend using denoising or not
   - Based on mean difference threshold

4. **Update your pipeline**
   - If denoising recommended → Use it in training and inference
   - If minimal impact → Can skip denoising (faster)

5. **Test on multiple datasets**
   ```bash
   python check_preprocessing.py training_dataset
   python check_preprocessing.py Inference_dataset1
   python check_preprocessing.py Inference_dataset2
   ```

6. **Make decision**
   - If all datasets show denoising helps → Use it
   - If inconsistent → Use denoising (safer)
   - If all show minimal impact → Can skip

## FAQ

### Q: Why does Panel 5 look similar to Panel 6?

**A:** Normalization (especially percentile clipping) already removes much of the noise. This is actually good - your preprocessing is robust!

### Q: Panel 7 shows some hot spots but mostly dark. Should I use denoising?

**A:** Check the mean value. If > 0.01, yes. If < 0.01, optional. Hot spots in specific regions suggest localized noise that denoising can help with.

### Q: The recommendation says "minimal impact" but I see visual difference. Which to trust?

**A:** Visual inspection is important too! If you see clear improvement in Panel 6 vs Panel 5, use denoising regardless of metrics.

### Q: Should I use a stronger denoising method than median?

**A:** Check Panel 7 first. If median shows minimal impact after normalization, stronger methods won't help much either. If median shows significant impact, you could try bilateral or NLM for even better results.

### Q: What if training and inference datasets show different results?

**A:** Use the more conservative approach (use denoising) to ensure consistent quality across all datasets.

---

**Key Takeaway:** Panel 7 (Difference After Normalization) tells you if denoising is worth the computational cost. Hot colors = yes, dark = optional.
