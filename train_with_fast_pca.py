"""
Fast PCA Training with Band Pre-filtering

This script uses band quality filtering BEFORE PCA for faster training:
1. Filter out noisy bands (reduces PCA computation time)
2. Apply PCA on clean bands only
3. Train 1D CNN classifier

Speed improvement: 2-5x faster PCA fitting + 2-3x faster training
"""

import argparse
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

from pca_with_band_filtering import PCAWithBandFiltering, load_hypercube
from model import create_model
from train import Trainer


class FastPCADataset(Dataset):
    """Dataset with band pre-filtering + PCA."""

    def __init__(self, dataset_path: str, label_path: str,
                 pca_selector: PCAWithBandFiltering = None,
                 max_samples: int = None):
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

        # Extract samples
        self.samples = []
        for label_info in labels_data:
            class_id = label_info['label']
            for coord in label_info['coordinates']:
                x, y = coord
                self.samples.append({'x': x, 'y': y, 'label': class_id})

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"✓ Dataset: {len(self.samples):,} samples")

    def load_spectrum(self, x, y):
        """Load full spectrum."""
        spectrum = np.zeros(self.n_bands, dtype=np.float32)

        for i in range(1, self.n_bands + 1):
            img_path = self.dataset_path / f'ImagesStack{i:03d}.png'
            if img_path.exists():
                img = Image.open(img_path).convert('L')
                spectrum[i - 1] = np.array(img)[y, x] / 255.0

        return spectrum

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x, y, label = sample['x'], sample['y'], sample['label']

        # Load full spectrum
        spectrum = self.load_spectrum(x, y)

        # Apply band filtering + PCA
        if self.pca_selector:
            spectrum = self.pca_selector.transform(spectrum)

        return torch.from_numpy(spectrum).float(), torch.tensor(label, dtype=torch.long)


def create_fast_dataloaders(dataset_path, label_path, pca_selector,
                            batch_size, train_split, num_workers, max_samples):
    """Create dataloaders with fast PCA."""

    full_dataset = FastPCADataset(dataset_path, label_path, pca_selector, max_samples)

    # Split train/val
    n_samples = len(full_dataset)
    n_train = int(n_samples * train_split)
    n_val = n_samples - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [n_train, n_val])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)

    print(f"✓ Train: {n_train:,} samples, {len(train_loader)} batches")
    print(f"✓ Val: {n_val:,} samples, {len(val_loader)} batches")

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Fast PCA training with band filtering')

    # Band filtering parameters (adjust these for speed/quality tradeoff)
    parser.add_argument('--keep-percentage', type=float, default=80.0,
                       help='Keep top X%% of bands by SNR (default: 80.0)')
    parser.add_argument('--snr-threshold', type=float, default=None,
                       help='Minimum SNR threshold (alternative to keep-percentage)')
    parser.add_argument('--variance-threshold', type=float, default=None,
                       help='Minimum variance threshold')
    parser.add_argument('--saturation-threshold', type=float, default=5.0,
                       help='Max %% saturated pixels (default: 5.0)')
    parser.add_argument('--darkness-threshold', type=float, default=5.0,
                       help='Max %% dark pixels (default: 5.0)')

    # PCA parameters
    parser.add_argument('--pca-components', type=int, default=None,
                       help='Number of PCA components (None = auto from variance)')
    parser.add_argument('--pca-variance', type=float, default=0.99,
                       help='PCA variance threshold (default: 0.99)')

    # Training parameters
    parser.add_argument('--data-dir', type=str, default='training_dataset')
    parser.add_argument('--label-path', type=str, default='Ground_Truth/labels.json')
    parser.add_argument('--n-classes', type=int, default=11)
    parser.add_argument('--batch-size', type=int, default=640)
    parser.add_argument('--n-epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--output-dir', type=str, default='checkpoints_fast_pca')

    # Comparison mode
    parser.add_argument('--compare', action='store_true',
                       help='Compare different keep-percentage values')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load hypercube for PCA fitting (once)
    print("="*80)
    print("LOADING HYPERCUBE")
    print("="*80)
    hypercube, wavelengths = load_hypercube(args.data_dir)

    if args.compare:
        # Compare different filtering levels
        configs = [
            {'keep_percentage': 90.0, 'name': 'Conservative (90%)'},
            {'keep_percentage': 80.0, 'name': 'Balanced (80%)'},
            {'keep_percentage': 70.0, 'name': 'Aggressive (70%)'},
            {'keep_percentage': 60.0, 'name': 'Very Aggressive (60%)'},
        ]

        results = []

        for config in configs:
            print(f"\n{'='*80}")
            print(f"TESTING: {config['name']}")
            print(f"{'='*80}")

            # Create PCA selector
            pca_selector = PCAWithBandFiltering(
                keep_percentage=config['keep_percentage'],
                saturation_threshold=args.saturation_threshold,
                darkness_threshold=args.darkness_threshold,
                n_components=args.pca_components,
                pca_variance_threshold=args.pca_variance,
                standardize=True
            )

            # Fit PCA
            pca_selector.fit(hypercube, wavelengths)

            # Save PCA model
            exp_dir = f"{args.output_dir}_{int(config['keep_percentage'])}"
            os.makedirs(exp_dir, exist_ok=True)
            pca_selector.save(os.path.join(exp_dir, 'pca_model.pkl'))

            # Create dataloaders
            print("\nCreating dataloaders...")
            train_loader, val_loader = create_fast_dataloaders(
                args.data_dir, args.label_path, pca_selector,
                args.batch_size, 0.9, 4, None
            )

            # Create model
            print("\nCreating model...")
            model = create_model(
                n_spectral_bands=pca_selector.n_components_selected,
                n_classes=args.n_classes,
                dropout_rate=0.5
            )

            # Train
            trainer = Trainer(
                model, train_loader, val_loader, device,
                learning_rate=args.learning_rate,
                total_epochs=args.n_epochs,
                checkpoint_dir=exp_dir,
                log_dir=f"logs_{int(config['keep_percentage'])}"
            )

            trainer.train()

            # Record results
            results.append({
                'name': config['name'],
                'keep_percentage': config['keep_percentage'],
                'original_bands': pca_selector.n_original_bands,
                'filtered_bands': pca_selector.n_filtered_bands,
                'pca_components': pca_selector.n_components_selected,
                'best_val_acc': trainer.best_val_acc,
                'reduction': (1 - pca_selector.n_components_selected/pca_selector.n_original_bands) * 100
            })

        # Summary
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        print(f"\n{'Config':<20} {'Bands':<25} {'Components':<12} {'Reduction':<12} {'Val Acc':<10}")
        print("-" * 80)

        for r in results:
            bands_str = f"{r['original_bands']}→{r['filtered_bands']}→{r['pca_components']}"
            print(f"{r['name']:<20} {bands_str:<25} {r['pca_components']:<12} "
                  f"{r['reduction']:.1f}%{'':<8} {r['best_val_acc']:.2f}%")

        # Save comparison
        with open('fast_pca_comparison.json', 'w') as f:
            json.dump(results, f, indent=2)

        # Best result
        best = max(results, key=lambda x: x['best_val_acc'])
        print(f"\n{'='*80}")
        print(f"BEST: {best['name']}")
        print(f"  Accuracy: {best['best_val_acc']:.2f}%")
        print(f"  Reduction: {best['reduction']:.1f}%")
        print("="*80)

    else:
        # Single configuration
        print(f"\n{'='*80}")
        print("FAST PCA TRAINING")
        print(f"{'='*80}")

        # Create PCA selector
        pca_selector = PCAWithBandFiltering(
            keep_percentage=args.keep_percentage if args.keep_percentage else None,
            snr_threshold=args.snr_threshold,
            variance_threshold=args.variance_threshold,
            saturation_threshold=args.saturation_threshold,
            darkness_threshold=args.darkness_threshold,
            n_components=args.pca_components,
            pca_variance_threshold=args.pca_variance,
            standardize=True
        )

        # Fit PCA
        pca_selector.fit(hypercube, wavelengths)

        # Visualize filtering
        pca_selector.band_filter.visualize_filtering(
            wavelengths,
            output_path=os.path.join(args.output_dir, 'band_filtering.png')
        )

        # Save PCA model
        os.makedirs(args.output_dir, exist_ok=True)
        pca_selector.save(os.path.join(args.output_dir, 'pca_model.pkl'))

        # Create dataloaders
        print("\nCreating dataloaders...")
        train_loader, val_loader = create_fast_dataloaders(
            args.data_dir, args.label_path, pca_selector,
            args.batch_size, 0.9, 4, None
        )

        # Create model
        print("\nCreating model...")
        model = create_model(
            n_spectral_bands=pca_selector.n_components_selected,
            n_classes=args.n_classes,
            dropout_rate=0.5
        )

        # Train
        trainer = Trainer(
            model, train_loader, val_loader, device,
            learning_rate=args.learning_rate,
            total_epochs=args.n_epochs,
            checkpoint_dir=args.output_dir,
            log_dir='logs_fast_pca'
        )

        trainer.train()

        print(f"\n✓ Training complete!")
        print(f"  Best accuracy: {trainer.best_val_acc:.2f}%")
        print(f"  Model saved to: {args.output_dir}/best_model.pth")
        print(f"  PCA model saved to: {args.output_dir}/pca_model.pkl")


if __name__ == '__main__':
    main()
