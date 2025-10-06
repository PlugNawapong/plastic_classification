"""
Dataset loader that uses only clean (non-noisy) spectral bands.

This version filters out noisy bands during loading for better model performance.
"""

import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List
import glob
from preprocessing import HyperspectralPreprocessor


def load_clean_band_indices(band_list_path='clean_normalized_bands/clean_bands_list.txt'):
    """
    Load clean band indices from the quality report.

    Args:
        band_list_path: Path to clean_bands_list.txt

    Returns:
        List of clean band indices (0-based)
    """
    if not os.path.exists(band_list_path):
        print(f"Warning: {band_list_path} not found. Using all bands.")
        return None

    clean_indices = []

    with open(band_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Parse "band_index, wavelength" format
                parts = line.split(',')
                if len(parts) >= 1:
                    try:
                        idx = int(parts[0])
                        clean_indices.append(idx - 1)  # Convert to 0-based
                    except ValueError:
                        continue

    print(f"Loaded {len(clean_indices)} clean band indices from {band_list_path}")
    return sorted(clean_indices)


class HyperspectralPlasticDatasetClean(Dataset):
    """
    Dataset for hyperspectral plastic classification using only clean bands.

    Automatically filters out noisy bands based on quality analysis.
    """

    def __init__(
        self,
        data_dir: str,
        label_path: str,
        clean_bands_list: Optional[str] = 'clean_normalized_bands/clean_bands_list.txt',
        transform=None,
        normalize: bool = True,
        max_samples: Optional[int] = None,
        sample_background: bool = True,
        background_ratio: float = 0.2,
        preprocessing_method: str = 'percentile',
        brightness_boost: bool = True,
        band_wise_norm: bool = True
    ):
        """
        Args:
            data_dir: Directory containing spectral band images
            label_path: Path to labels.json
            clean_bands_list: Path to clean_bands_list.txt (None = use all bands)
            transform: Optional transform
            normalize: Whether to normalize
            max_samples: Maximum pixels to load
            sample_background: Whether to subsample background
            background_ratio: Ratio of background pixels
            preprocessing_method: 'simple' or 'percentile'
            brightness_boost: Whether to boost brightness
            band_wise_norm: Whether to normalize per-band
        """
        self.data_dir = data_dir
        self.label_path = label_path
        self.transform = transform
        self.normalize = normalize
        self.max_samples = max_samples
        self.sample_background = sample_background
        self.background_ratio = background_ratio

        # Load clean band indices
        self.clean_band_indices = load_clean_band_indices(clean_bands_list) if clean_bands_list else None

        # Initialize preprocessor
        self.preprocessor = HyperspectralPreprocessor(
            method=preprocessing_method,
            brightness_boost=brightness_boost,
            band_wise=band_wise_norm
        ) if normalize else None

        # Load class mapping
        self.class_mapping = self._load_class_mapping()
        self.n_classes = len(self.class_mapping)

        # Load spectral images (only clean bands)
        self.spectral_bands, self.n_bands = self._load_spectral_bands()

        # Load labels
        self.labels = self._load_labels()

        # Extract pixel-level data
        self.spectral_data, self.pixel_labels = self._extract_pixels()

        print(f"Dataset loaded: {len(self.spectral_data)} pixels, {self.n_bands} bands, {self.n_classes} classes")
        self._print_class_distribution()

    def _load_class_mapping(self) -> dict:
        """Load class mapping from labels.json."""
        label_dir = os.path.dirname(self.label_path)
        json_path = os.path.join(label_dir, 'labels.json')

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                content = f.read()
                import ast
                start_idx = content.find('CLASS_MAPPING = {')
                end_idx = content.find('}', start_idx) + 1
                class_map_str = content[start_idx:end_idx].replace('CLASS_MAPPING = ', '')
                class_mapping = ast.literal_eval(class_map_str)
                return class_mapping
        else:
            raise FileNotFoundError(f"labels.json not found at {json_path}")

    def _load_spectral_bands(self) -> Tuple[np.ndarray, int]:
        """Load only clean spectral band images."""
        image_files = sorted(glob.glob(os.path.join(self.data_dir, 'ImagesStack*.png')))

        if len(image_files) == 0:
            raise FileNotFoundError(f"No ImagesStack*.png files found in {self.data_dir}")

        # Determine which bands to load
        if self.clean_band_indices is not None:
            bands_to_load = [image_files[i] for i in self.clean_band_indices if i < len(image_files)]
            print(f"\nUsing {len(bands_to_load)} clean bands out of {len(image_files)} total")
        else:
            bands_to_load = image_files
            print(f"\nUsing all {len(image_files)} bands (no filtering)")

        # Load dimensions
        first_img = np.array(Image.open(bands_to_load[0]))
        height, width = first_img.shape
        n_bands = len(bands_to_load)

        # Pre-allocate array
        spectral_bands = np.zeros((n_bands, height, width), dtype=np.float32)

        # Load bands
        print(f"Loading {n_bands} spectral bands...")
        for i, img_path in enumerate(bands_to_load):
            img = np.array(Image.open(img_path))
            spectral_bands[i] = img.astype(np.float32)

            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1}/{n_bands} bands")

        # Apply preprocessing
        if self.normalize and self.preprocessor is not None:
            print("\nApplying preprocessing...")
            spectral_bands = self.preprocessor.preprocess(spectral_bands, fit_pca=False)

        return spectral_bands, n_bands

    def _load_labels(self) -> np.ndarray:
        """Load label image."""
        label_img_path = self.label_path.replace('.json', '.png')

        if not os.path.exists(label_img_path):
            raise FileNotFoundError(f"Label image not found at {label_img_path}")

        label_img = np.array(Image.open(label_img_path))
        height, width = label_img.shape[:2]
        labels = np.zeros((height, width), dtype=np.int64)

        for rgb_color, class_idx in self.class_mapping.items():
            mask = np.all(label_img == np.array(rgb_color), axis=-1)
            labels[mask] = class_idx

        return labels

    def _extract_pixels(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pixel-level spectral signatures."""
        _, height, width = self.spectral_bands.shape

        spectral_pixels = self.spectral_bands.reshape(self.n_bands, -1).T
        label_pixels = self.labels.flatten()

        # Subsample background
        if self.sample_background:
            bg_mask = label_pixels == 0
            fg_mask = ~bg_mask

            n_bg = bg_mask.sum()
            n_fg = fg_mask.sum()

            n_bg_keep = int(n_fg * self.background_ratio)

            if n_bg > n_bg_keep:
                bg_indices = np.where(bg_mask)[0]
                np.random.shuffle(bg_indices)
                bg_keep_indices = bg_indices[:n_bg_keep]

                keep_indices = np.concatenate([np.where(fg_mask)[0], bg_keep_indices])
                np.random.shuffle(keep_indices)

                spectral_pixels = spectral_pixels[keep_indices]
                label_pixels = label_pixels[keep_indices]

        # Limit samples
        if self.max_samples and len(spectral_pixels) > self.max_samples:
            indices = np.random.choice(len(spectral_pixels), self.max_samples, replace=False)
            spectral_pixels = spectral_pixels[indices]
            label_pixels = label_pixels[indices]

        return spectral_pixels, label_pixels

    def _print_class_distribution(self):
        """Print class distribution."""
        unique, counts = np.unique(self.pixel_labels, return_counts=True)
        print("\nClass distribution:")
        for class_idx, count in zip(unique, counts):
            percentage = 100 * count / len(self.pixel_labels)
            print(f"  Class {class_idx}: {count:,} pixels ({percentage:.2f}%)")

    def __len__(self) -> int:
        return len(self.spectral_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        spectral = self.spectral_data[idx]
        label = self.pixel_labels[idx]

        spectral = torch.from_numpy(spectral).float()
        label = torch.tensor(label, dtype=torch.long)

        spectral = spectral.unsqueeze(0)

        if self.transform:
            spectral = self.transform(spectral)

        return spectral, label


def create_clean_dataloaders(
    train_dir: str,
    label_path: str,
    clean_bands_list: str = 'clean_normalized_bands/clean_bands_list.txt',
    batch_size: int = 640,
    train_split: float = 0.9,
    num_workers: int = 4,
    max_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders using only clean bands.

    Args:
        train_dir: Directory with training spectral images
        label_path: Path to labels
        clean_bands_list: Path to clean bands list
        batch_size: Batch size
        train_split: Train/val split
        num_workers: Number of workers
        max_samples: Max samples to load

    Returns:
        train_loader, val_loader
    """
    # Load dataset with clean bands only
    full_dataset = HyperspectralPlasticDatasetClean(
        train_dir,
        label_path,
        clean_bands_list=clean_bands_list,
        normalize=True,
        max_samples=max_samples,
        sample_background=True,
        background_ratio=0.2,
        preprocessing_method='percentile',
        brightness_boost=True,
        band_wise_norm=True
    )

    # Split
    n_train = int(len(full_dataset) * train_split)
    n_val = len(full_dataset) - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\nDataloaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Spectral bands: {full_dataset.n_bands} (clean bands only)")

    return train_loader, val_loader


if __name__ == "__main__":
    # Test clean bands dataset
    train_loader, val_loader = create_clean_dataloaders(
        train_dir="training_dataset",
        label_path="Ground_Truth/labels.json",
        clean_bands_list='clean_normalized_bands/clean_bands_list.txt',
        batch_size=640,
        max_samples=10000
    )

    # Test batch
    for spectral, labels in train_loader:
        print(f"\nBatch shapes:")
        print(f"  Spectral: {spectral.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Label range: {labels.min()} - {labels.max()}")
        break
