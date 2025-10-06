"""
Dataset loader for hyperspectral plastic classification.
Handles loading of stacked spectral band images and pixel-level labels.
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


class HyperspectralPlasticDataset(Dataset):
    """
    Dataset for hyperspectral plastic classification.

    Loads spectral bands from PNG images and converts them into 1D spectral signatures
    for pixel-level classification.
    """

    def __init__(
        self,
        data_dir: str,
        label_path: str,
        transform=None,
        normalize: bool = True,
        max_samples: Optional[int] = None,
        sample_background: bool = True,
        background_ratio: float = 0.2,
        preprocessing_method: str = 'percentile',  # 'simple' or 'percentile'
        brightness_boost: bool = True,
        band_wise_norm: bool = True
    ):
        """
        Args:
            data_dir: Directory containing spectral band images (ImagesStack*.png)
            label_path: Path to labels.json or labels.png
            transform: Optional transform to apply to spectral data
            normalize: Whether to normalize spectral values to [0, 1]
            max_samples: Maximum number of pixels to load (for memory management)
            sample_background: Whether to subsample background pixels
            background_ratio: Ratio of background pixels to keep if sampling
            preprocessing_method: 'simple' (paper) or 'percentile' (improved)
            brightness_boost: Whether to boost brightness before normalization
            band_wise_norm: Whether to normalize each band independently
        """
        self.data_dir = data_dir
        self.label_path = label_path
        self.transform = transform
        self.normalize = normalize
        self.max_samples = max_samples
        self.sample_background = sample_background
        self.background_ratio = background_ratio

        # Initialize preprocessor
        self.preprocessor = HyperspectralPreprocessor(
            method=preprocessing_method,
            brightness_boost=brightness_boost,
            band_wise=band_wise_norm
        ) if normalize else None

        # Load class mapping from labels.json
        self.class_mapping = self._load_class_mapping()
        self.n_classes = len(self.class_mapping)

        # Load spectral images
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
                # Extract CLASS_MAPPING dictionary
                import ast
                start_idx = content.find('CLASS_MAPPING = {')
                end_idx = content.find('}', start_idx) + 1
                class_map_str = content[start_idx:end_idx].replace('CLASS_MAPPING = ', '')
                class_mapping = ast.literal_eval(class_map_str)
                return class_mapping
        else:
            raise FileNotFoundError(f"labels.json not found at {json_path}")

    def _load_spectral_bands(self) -> Tuple[np.ndarray, int]:
        """Load all spectral band images."""
        # Find all ImagesStack*.png files
        image_files = sorted(glob.glob(os.path.join(self.data_dir, 'ImagesStack*.png')))

        if len(image_files) == 0:
            raise FileNotFoundError(f"No ImagesStack*.png files found in {self.data_dir}")

        # Load first image to get dimensions
        first_img = np.array(Image.open(image_files[0]))
        height, width = first_img.shape
        n_bands = len(image_files)

        # Pre-allocate array for all bands
        spectral_bands = np.zeros((n_bands, height, width), dtype=np.float32)

        # Load all bands
        print(f"Loading {n_bands} spectral bands...")
        for i, img_path in enumerate(image_files):
            img = np.array(Image.open(img_path))
            spectral_bands[i] = img.astype(np.float32)

            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1}/{n_bands} bands")

        # Apply preprocessing if requested
        if self.normalize and self.preprocessor is not None:
            print("\nApplying preprocessing...")
            spectral_bands = self.preprocessor.preprocess(spectral_bands, fit_pca=False)

        return spectral_bands, n_bands

    def _load_labels(self) -> np.ndarray:
        """Load label image and convert to class indices."""
        label_img_path = self.label_path.replace('.json', '.png')

        if not os.path.exists(label_img_path):
            raise FileNotFoundError(f"Label image not found at {label_img_path}")

        # Load RGB label image
        label_img = np.array(Image.open(label_img_path))

        # Convert RGB to class indices
        height, width = label_img.shape[:2]
        labels = np.zeros((height, width), dtype=np.int64)

        for rgb_color, class_idx in self.class_mapping.items():
            # Create mask for this color
            mask = np.all(label_img == np.array(rgb_color), axis=-1)
            labels[mask] = class_idx

        return labels

    def _extract_pixels(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract pixel-level spectral signatures and labels."""
        _, height, width = self.spectral_bands.shape

        # Reshape spectral bands: (n_bands, H, W) -> (H*W, n_bands)
        spectral_pixels = self.spectral_bands.reshape(self.n_bands, -1).T

        # Flatten labels: (H, W) -> (H*W,)
        label_pixels = self.labels.flatten()

        # Optionally subsample background (class 0)
        if self.sample_background:
            bg_mask = label_pixels == 0
            fg_mask = ~bg_mask

            n_bg = bg_mask.sum()
            n_fg = fg_mask.sum()

            # Keep only a fraction of background pixels
            n_bg_keep = int(n_fg * self.background_ratio)

            if n_bg > n_bg_keep:
                bg_indices = np.where(bg_mask)[0]
                np.random.shuffle(bg_indices)
                bg_keep_indices = bg_indices[:n_bg_keep]

                # Combine foreground and sampled background
                keep_indices = np.concatenate([np.where(fg_mask)[0], bg_keep_indices])
                np.random.shuffle(keep_indices)

                spectral_pixels = spectral_pixels[keep_indices]
                label_pixels = label_pixels[keep_indices]

        # Optionally limit total samples
        if self.max_samples and len(spectral_pixels) > self.max_samples:
            indices = np.random.choice(len(spectral_pixels), self.max_samples, replace=False)
            spectral_pixels = spectral_pixels[indices]
            label_pixels = label_pixels[indices]

        return spectral_pixels, label_pixels

    def _print_class_distribution(self):
        """Print distribution of classes in the dataset."""
        unique, counts = np.unique(self.pixel_labels, return_counts=True)
        print("\nClass distribution:")
        for class_idx, count in zip(unique, counts):
            percentage = 100 * count / len(self.pixel_labels)
            print(f"  Class {class_idx}: {count:,} pixels ({percentage:.2f}%)")

    def __len__(self) -> int:
        return len(self.spectral_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single pixel's spectral signature and label.

        Returns:
            spectral: Tensor of shape (1, n_bands)
            label: Tensor of shape ()
        """
        spectral = self.spectral_data[idx]
        label = self.pixel_labels[idx]

        # Convert to tensor
        spectral = torch.from_numpy(spectral).float()
        label = torch.tensor(label, dtype=torch.long)

        # Add channel dimension for 1D conv: (n_bands,) -> (1, n_bands)
        spectral = spectral.unsqueeze(0)

        if self.transform:
            spectral = self.transform(spectral)

        return spectral, label


def create_dataloaders(
    train_dir: str,
    label_path: str,
    batch_size: int = 640,
    train_split: float = 0.9,
    num_workers: int = 4,
    max_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        train_dir: Directory with training spectral images
        label_path: Path to labels file
        batch_size: Batch size for training
        train_split: Fraction of data for training (rest for validation)
        num_workers: Number of worker processes for data loading
        max_samples: Maximum number of samples to load

    Returns:
        train_loader, val_loader
    """
    # Load full dataset
    full_dataset = HyperspectralPlasticDataset(
        train_dir,
        label_path,
        normalize=True,
        max_samples=max_samples,
        sample_background=True,
        background_ratio=0.2
    )

    # Split into train and validation
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

    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    train_dir = "training_dataset"
    label_path = "Ground_Truth/labels.json"

    train_loader, val_loader = create_dataloaders(
        train_dir,
        label_path,
        batch_size=640,
        max_samples=10000  # Limit for testing
    )

    # Test batch loading
    for spectral, labels in train_loader:
        print(f"\nBatch shapes:")
        print(f"  Spectral: {spectral.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Label range: {labels.min()} - {labels.max()}")
        break
