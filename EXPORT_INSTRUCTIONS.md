# How to Export Normalized Bands

## Quick Start

### 1. Install Dependencies (if not already installed)

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install numpy pillow matplotlib tqdm scikit-learn scipy
```

### 2. Run the Export Script

```bash
python export_normalized_bands.py
```

This will:
- ✓ Load all 459 spectral bands from `training_dataset/`
- ✓ Apply band-wise normalization (brightness boost + percentile)
- ✓ Export each normalized band as PNG with clear naming
- ✓ Generate verification report
- ✓ Create comparison samples

## Output Files

### Normalized Band Images
Location: `normalized_bands/`

Files are named:
```
Band_001_normalized.png  (Band 1)
Band_002_normalized.png  (Band 2)
Band_003_normalized.png  (Band 3)
...
Band_459_normalized.png  (Band 459)
```

**Format**: 8-bit PNG (0-255), grayscale
**Original values**: [0, 1] scaled to [0, 255] for viewing

### Verification Report
Location: `normalized_bands/normalization_report.txt`

Contains:
- ✓ Original vs normalized range for each band
- ✓ Range utilization percentage (should be >90% for band-wise)
- ✓ Data integrity checks
- ✓ Summary statistics

### Comparison Samples
Location: `normalized_bands/comparison_samples.png`

Visual comparison showing:
- Sample bands: 1st, 25%, 50%, 75%, last
- Original vs normalized side-by-side
- Range utilization for each

## What to Check

### 1. File Integrity

Check that all 459 files were created:
```bash
ls normalized_bands/Band_*_normalized.png | wc -l
# Should output: 459
```

### 2. Sequential Naming

Files should be numbered 001 to 459:
```bash
ls normalized_bands/ | head -10
# Should show: Band_001_normalized.png, Band_002_normalized.png, etc.
```

### 3. Visual Inspection

Open a few random files to verify:
```bash
# View first band
open normalized_bands/Band_001_normalized.png

# View middle band
open normalized_bands/Band_230_normalized.png

# View last band
open normalized_bands/Band_459_normalized.png
```

**What to look for:**
- Images should have good contrast (not all black or all white)
- Different bands should show different patterns
- No corruption or artifacts

### 4. Review Report

```bash
cat normalized_bands/normalization_report.txt
```

**Key metrics to check:**
- Range Utilization Average: Should be >90% for band-wise
- All bands normalized to [0,1]: Should be 459/459
- No warning messages

### 5. View Comparison

```bash
open normalized_bands/comparison_samples.png
```

**What to expect:**
- Normalized images have full contrast
- All samples utilize full [0,1] range
- Utilization ~95-100% for each band

## Verification Checklist

- [ ] All 459 files exported
- [ ] Files named sequentially (001-459)
- [ ] File sizes reasonable (100-600 KB each)
- [ ] Sample images load correctly
- [ ] Report shows >90% average utilization
- [ ] Comparison samples look good
- [ ] No error messages in report

## Troubleshooting

### Issue: Not all bands exported

**Solution:** Check available disk space:
```bash
df -h .
```

Each band ~200-500 KB, total ~200 MB needed.

### Issue: Low range utilization (<90%)

**Cause:** Band-wise normalization may not be enabled.

**Fix:** Edit `export_normalized_bands.py` line 245:
```python
export_normalized_bands(
    band_wise=True  # Make sure this is True!
)
```

### Issue: Images look corrupted

**Cause:** File may have been interrupted during write.

**Fix:** Re-run the export script:
```bash
rm -rf normalized_bands/
python export_normalized_bands.py
```

### Issue: Module not found errors

**Solution:** Install dependencies:
```bash
pip install numpy pillow matplotlib tqdm scikit-learn scipy
```

## Advanced: Export Different Normalization Methods

To export using different preprocessing:

### Paper's Method (for comparison)
```python
export_normalized_bands(
    output_dir='normalized_bands_paper',
    method='simple',
    brightness_boost=False,
    band_wise=False
)
```

### Global Method
```python
export_normalized_bands(
    output_dir='normalized_bands_global',
    method='percentile',
    brightness_boost=True,
    band_wise=False
)
```

### Band-wise Method (RECOMMENDED - default)
```python
export_normalized_bands(
    output_dir='normalized_bands_bandwise',
    method='percentile',
    brightness_boost=True,
    band_wise=True  # ⭐ BEST
)
```

## File Naming Convention

```
Band_XXX_normalized.png
     │││
     │││
     │││└─ Sequential number (001-459)
     ││└── Zero-padded 3 digits
     │└─── Band prefix
     └──── Output identifier
```

Examples:
- `Band_001_normalized.png` = First spectral band (900nm)
- `Band_230_normalized.png` = Middle band (~1300nm)
- `Band_459_normalized.png` = Last band (1700nm)

## Data Integrity Verification

The script automatically verifies:

1. **Completeness**: All bands exported
2. **Range**: Values properly normalized to [0,1]
3. **Utilization**: Each band uses >90% of available range
4. **File format**: Valid PNG files
5. **Sequential**: No missing band numbers

All results in: `normalized_bands/normalization_report.txt`

## Expected Output Structure

```
normalized_bands/
├── Band_001_normalized.png
├── Band_002_normalized.png
├── Band_003_normalized.png
├── ...
├── Band_459_normalized.png
├── normalization_report.txt        # Verification report
└── comparison_samples.png          # Visual comparison
```

Total: 461 files (459 bands + 2 reports)
Size: ~200-250 MB

## Next Steps

After verifying data integrity:

1. ✓ Confirm all 459 files present
2. ✓ Review normalization report
3. ✓ Spot-check random samples visually
4. Proceed with training using normalized data
5. Compare accuracy vs paper's method

Ready to train!
