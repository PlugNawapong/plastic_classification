"""
Training Pipeline with PCA Band Selection

This script trains the classifier with PCA-reduced bands:
1. Load normalized hypercube
2. Apply PCA dimensionality reduction
3. Train model on reduced features
4. Compare with baseline (no PCA)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from pca_band_selection import PCABandSelector
from model import create_model
from train import Trainer


class PCAPreparedDataset(Dataset):
    """Dataset that loads PCA-reduced hyperspectral data."""

    def __init__(self, dataset_path: str, label_path: str,
                 pca_selector: PCABandSelector = None,
                 max_samples: int = None):
        """
        Args:
            dataset_path: Path to dataset folder
            label_path: Path to labels.json
            pca_selector: Fitted PCA selector (None = no PCA)
            max_samples: Maximum number of samples to load
        """
        self.dataset_path = Path(dataset_path)
        self.pca_selector = pca_selector

        # Load wavelengths
        with open(self.dataset_path / 'header.json', 'r') as f:
            header = json.load(f)
        self.wavelengths = header['wavelength (nm)']
        self.n_bands = len(self.wavelengths)

        # Load labels
        with open(label_path, 'r') as f:
            labels_data = json.load(f)

        # Extract pixel coordinates and labels
        self.samples = []
        for label_info in labels_data:
            class_id = label_info['label']
            for coord in label_info['coordinates']:
                x, y = coord
                self.samples.append({'x': x, 'y': y, 'label': class_id})

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"✓ Dataset loaded: {len(self.samples)} samples")
        if pca_selector:
            print(f"✓ PCA components: {pca_selector.n_components_selected}")

    def load_pixel_spectrum(self, x: int, y: int) -> np.ndarray:
        """Load spectral signature for a single pixel."""
        spectrum = np.zeros(self.n_bands, dtype=np.float32)

        for band_idx in range(1, self.n_bands + 1):
            img_path = self.dataset_path / f'ImagesStack{band_idx:03d}.png'
            if img_path.exists():
                img = Image.open(img_path).convert('L')
                spectrum[band_idx - 1] = np.array(img)[y, x] / 255.0

        return spectrum

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x, y, label = sample['x'], sample['y'], sample['label']

        # Load spectrum
        spectrum = self.load_pixel_spectrum(x, y)

        # Apply PCA if available
        if self.pca_selector:
            # Reshape for PCA: (n_bands,) → (1, n_bands)
            spectrum_2d = spectrum.reshape(1, -1)

            # Standardize
            if self.pca_selector.standardize and self.pca_selector.scaler:
                spectrum_2d = self.pca_selector.scaler.transform(spectrum_2d)

            # Transform
            spectrum_reduced = self.pca_selector.pca.transform(spectrum_2d)
            spectrum = spectrum_reduced.flatten()

        # Convert to tensor
        spectrum_tensor = torch.from_numpy(spectrum).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return spectrum_tensor, label_tensor


def create_pca_dataloaders(
    dataset_path: str,
    label_path: str,
    pca_selector: PCABandSelector = None,
    batch_size: int = 640,
    train_split: float = 0.9,
    num_workers: int = 4,
    max_samples: int = None
):
    """
    Create dataloaders with optional PCA transformation.

    Args:
        dataset_path: Path to dataset
        label_path: Path to labels.json
        pca_selector: Fitted PCA selector (None = no PCA)
        batch_size: Batch size
        train_split: Train/val split ratio
        num_workers: Number of data loading workers
        max_samples: Maximum samples to load

    Returns:
        train_loader, val_loader
    """
    # Load full dataset
    full_dataset = PCAPreparedDataset(
        dataset_path=dataset_path,
        label_path=label_path,
        pca_selector=pca_selector,
        max_samples=max_samples
    )

    # Split train/val
    n_samples = len(full_dataset)
    n_train = int(n_samples * train_split)
    n_val = n_samples - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val]
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

    print(f"✓ Train samples: {n_train}")
    print(f"✓ Val samples: {n_val}")
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")

    return train_loader, val_loader


def train_with_pca_comparison(n_components_list=[None, 50, 100, 150, 200]):
    """
    Train models with different PCA configurations and compare results.

    Args:
        n_components_list: List of component counts to test (None = no PCA)
    """
    config = {
        'data_dir': 'training_dataset',
        'label_path': 'Ground_Truth/labels.json',
        'n_classes': 11,
        'batch_size': 640,
        'train_split': 0.9,
        'num_workers': 4,
        'learning_rate': 0.001,
        'lr_min': 0.0001,
        'warmup_epochs': 10,
        'total_epochs': 50,
        'dropout_rate': 0.5,
        'max_samples': None,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load hypercube for PCA fitting (only once)
    print("Loading hypercube for PCA fitting...")
    from pca_band_selection import load_normalized_hypercube
    hypercube, wavelengths = load_normalized_hypercube(config['data_dir'])

    results = []

    for n_comp in n_components_list:
        print("\n" + "="*80)
        if n_comp is None:
            print(f"TRAINING WITHOUT PCA (BASELINE)")
            exp_name = "no_pca"
            n_bands = len(wavelengths)
        else:
            print(f"TRAINING WITH PCA: {n_comp} components")
            exp_name = f"pca_{n_comp}"
            n_bands = n_comp
        print("="*80)

        # Fit or skip PCA
        if n_comp is None:
            pca_selector = None
        else:
            print(f"\nFitting PCA with {n_comp} components...")
            pca_selector = PCABandSelector(n_components=n_comp, standardize=True)
            pca_selector.fit(hypercube, wavelengths=wavelengths)

        # Create dataloaders
        print("\nCreating dataloaders...")
        train_loader, val_loader = create_pca_dataloaders(
            dataset_path=config['data_dir'],
            label_path=config['label_path'],
            pca_selector=pca_selector,
            batch_size=config['batch_size'],
            train_split=config['train_split'],
            num_workers=config['num_workers'],
            max_samples=config['max_samples']
        )

        # Create model
        print("\nCreating model...")
        model = create_model(
            n_spectral_bands=n_bands,
            n_classes=config['n_classes'],
            dropout_rate=config['dropout_rate']
        )

        # Create trainer
        checkpoint_dir = f"checkpoints_{exp_name}"
        log_dir = f"logs_{exp_name}"

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=config['learning_rate'],
            lr_min=config['lr_min'],
            warmup_epochs=config['warmup_epochs'],
            total_epochs=config['total_epochs'],
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir
        )

        # Train
        trainer.train()

        # Record results
        results.append({
            'n_components': n_comp,
            'n_bands': n_bands,
            'best_val_acc': trainer.best_val_acc,
            'best_epoch': trainer.best_epoch,
            'checkpoint_dir': checkpoint_dir
        })

        # Save PCA model if used
        if pca_selector:
            pca_model_path = os.path.join(checkpoint_dir, 'pca_model.pkl')
            pca_selector.save_model(pca_model_path)

    # Summary comparison
    print("\n" + "="*80)
    print("TRAINING COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Config':<20} {'Bands':<10} {'Best Val Acc':<15} {'Best Epoch':<12}")
    print("-" * 80)

    for result in results:
        config_name = f"No PCA" if result['n_components'] is None else f"PCA-{result['n_components']}"
        print(f"{config_name:<20} {result['n_bands']:<10} "
              f"{result['best_val_acc']:>13.2f}% {result['best_epoch']:>12}")

    # Find best configuration
    best_result = max(results, key=lambda x: x['best_val_acc'])
    best_config = "No PCA" if best_result['n_components'] is None else f"PCA-{best_result['n_components']}"

    print("\n" + "="*80)
    print(f"BEST CONFIGURATION: {best_config}")
    print(f"  Validation Accuracy: {best_result['best_val_acc']:.2f}%")
    print(f"  Number of Bands: {best_result['n_bands']}")
    print(f"  Best Epoch: {best_result['best_epoch']}")
    print(f"  Checkpoint: {best_result['checkpoint_dir']}/best_model.pth")
    print("="*80)

    # Save comparison results
    comparison_file = 'pca_training_comparison.json'
    with open(comparison_file, 'w') as f:
        json.dump({
            'results': results,
            'best_config': best_result
        }, f, indent=2)

    print(f"\n✓ Comparison results saved to: {comparison_file}")


def main():
    """Main training function with PCA options."""
    import argparse

    parser = argparse.ArgumentParser(description='Train classifier with PCA band selection')
    parser.add_argument('--pca-components', type=int, default=None,
                       help='Number of PCA components (None = no PCA)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple PCA configurations')
    parser.add_argument('--compare-configs', nargs='+', type=int,
                       default=[50, 100, 150, 200],
                       help='PCA component counts to compare (when --compare is used)')

    args = parser.parse_args()

    if args.compare:
        # Compare multiple configurations (including no PCA)
        configs = [None] + args.compare_configs
        print(f"\nComparing configurations: {configs}")
        train_with_pca_comparison(n_components_list=configs)
    else:
        # Single training run
        configs = [args.pca_components]
        train_with_pca_comparison(n_components_list=configs)


if __name__ == '__main__':
    main()
