"""
Training pipeline for P1CH plastic classifier with cosine annealing scheduler.
Implements industrial-standard training with comprehensive logging and checkpointing.
"""

import os
import time
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import create_model
from dataset import create_dataloaders


class CosineAnnealingWithWarmup:
    """
    Cosine annealing scheduler with warmup phase.

    Learning rate schedule:
    - Warmup phase: Linear increase from eta_min to eta_max
    - Cosine phase: Cosine annealing from eta_max to eta_min
    """

    def __init__(self, optimizer, eta_max=0.001, eta_min=0.0001, T_warmup=10, T_total=50):
        """
        Args:
            optimizer: Optimizer instance
            eta_max: Maximum learning rate
            eta_min: Minimum learning rate
            T_warmup: Number of warmup epochs
            T_total: Total number of epochs
        """
        self.optimizer = optimizer
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.T_warmup = T_warmup
        self.T_total = T_total
        self.current_epoch = 0

    def step(self, epoch=None):
        """Update learning rate for the given epoch."""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        if self.current_epoch < self.T_warmup:
            # Warmup phase: linear increase
            lr = self.eta_min + (self.eta_max - self.eta_min) * (self.current_epoch / self.T_warmup)
        else:
            # Cosine annealing phase
            progress = (self.current_epoch - self.T_warmup) / (self.T_total - self.T_warmup)
            lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def get_last_lr(self):
        """Return current learning rate."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class Trainer:
    """Training manager for P1CH classifier."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device='cuda',
        learning_rate=0.001,
        lr_min=0.0001,
        warmup_epochs=10,
        total_epochs=50,
        checkpoint_dir='checkpoints',
        log_dir='logs'
    ):
        """
        Args:
            model: P1CH model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Initial/maximum learning rate
            lr_min: Minimum learning rate for cosine annealing
            warmup_epochs: Number of warmup epochs
            total_epochs: Total training epochs
            checkpoint_dir: Directory to save model checkpoints
            log_dir: Directory for tensorboard logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.total_epochs = total_epochs
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Loss function: Cross-Entropy
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer: Adam
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Learning rate scheduler: Cosine annealing with warmup
        self.scheduler = CosineAnnealingWithWarmup(
            self.optimizer,
            eta_max=learning_rate,
            eta_min=lr_min,
            T_warmup=warmup_epochs,
            T_total=total_epochs
        )

        # Tensorboard writer
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(os.path.join(log_dir, f'run_{timestamp}'))

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.train_history = []
        self.val_history = []

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}/{self.total_epochs} [Train]')

        for batch_idx, (spectral, labels) in enumerate(pbar):
            spectral, labels = spectral.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(spectral)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            acc = 100.0 * correct / total
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{acc:.2f}%'})

        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch}/{self.total_epochs} [Val]')

            for batch_idx, (spectral, labels) in enumerate(pbar):
                spectral, labels = spectral.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(spectral)
                loss = self.criterion(outputs, labels)

                # Statistics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update progress bar
                avg_loss = total_loss / (batch_idx + 1)
                acc = 100.0 * correct / total
                pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{acc:.2f}%'})

        epoch_loss = total_loss / len(self.val_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_history': self.train_history,
            'val_history': self.val_history,
        }

        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f'✓ Best model saved (Val Acc: {self.best_val_acc:.2f}%)')

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        print(f'✓ Checkpoint loaded from epoch {self.current_epoch}')

    def train(self, resume_from=None):
        """
        Main training loop.

        Args:
            resume_from: Path to checkpoint to resume from (optional)
        """
        if resume_from and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)
            start_epoch = self.current_epoch + 1
        else:
            start_epoch = 1

        print(f"\n{'='*70}")
        print(f"Training P1CH Classifier")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Total epochs: {self.total_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Model parameters: {self.model.get_num_params():,}")
        print(f"{'='*70}\n")

        for epoch in range(start_epoch, self.total_epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Update learning rate
            lr = self.scheduler.step(epoch)
            print(f'\nEpoch {epoch}/{self.total_epochs} | LR: {lr:.6f}')

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            epoch_time = time.time() - epoch_start

            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning_rate', lr, epoch)

            # Save history
            self.train_history.append({'epoch': epoch, 'loss': train_loss, 'acc': train_acc})
            self.val_history.append({'epoch': epoch, 'loss': val_loss, 'acc': val_acc})

            # Print summary
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            print(f'Time: {epoch_time:.2f}s')

            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch

            self.save_checkpoint(is_best)

        print(f"\n{'='*70}")
        print(f"Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        print(f"{'='*70}\n")

        # Save training history
        history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump({
                'train': self.train_history,
                'val': self.val_history,
                'best_epoch': self.best_epoch,
                'best_val_acc': self.best_val_acc
            }, f, indent=2)

        self.writer.close()


def main():
    """Main training function."""
    # Configuration
    config = {
        'data_dir': 'training_dataset',
        'label_path': 'Ground_Truth/labels.json',
        'n_spectral_bands': 459,  # Number of spectral bands in dataset
        'n_classes': 11,  # Number of classes from labels.json
        'batch_size': 640,
        'train_split': 0.9,
        'num_workers': 4,
        'learning_rate': 0.001,
        'lr_min': 0.0001,
        'warmup_epochs': 10,
        'total_epochs': 50,
        'dropout_rate': 0.5,
        'max_samples': None,  # Set to limit samples for testing
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs'
    }

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataloaders
    print("\nLoading dataset...")
    train_loader, val_loader = create_dataloaders(
        config['data_dir'],
        config['label_path'],
        batch_size=config['batch_size'],
        train_split=config['train_split'],
        num_workers=config['num_workers'],
        max_samples=config['max_samples']
    )

    # Create model
    print("\nCreating model...")
    model = create_model(
        n_spectral_bands=config['n_spectral_bands'],
        n_classes=config['n_classes'],
        dropout_rate=config['dropout_rate']
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config['learning_rate'],
        lr_min=config['lr_min'],
        warmup_epochs=config['warmup_epochs'],
        total_epochs=config['total_epochs'],
        checkpoint_dir=config['checkpoint_dir'],
        log_dir=config['log_dir']
    )

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
