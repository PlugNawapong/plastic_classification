"""
Evaluation metrics and visualization for P1CH plastic classifier.
Includes confusion matrix, per-class metrics, and prediction visualization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
import torch
from tqdm import tqdm


class Evaluator:
    """Evaluation manager for P1CH classifier."""

    def __init__(self, model, device='cuda', class_names=None):
        """
        Args:
            model: Trained P1CH model
            device: Device to run evaluation on
            class_names: List of class names for visualization
        """
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names or [f'Class_{i}' for i in range(11)]

    def evaluate(self, data_loader, return_predictions=False):
        """
        Evaluate model on a dataset.

        Args:
            data_loader: DataLoader for evaluation
            return_predictions: Whether to return predictions and labels

        Returns:
            metrics: Dictionary of evaluation metrics
            predictions (optional): Numpy array of predictions
            labels (optional): Numpy array of ground truth labels
        """
        self.model.eval()

        all_predictions = []
        all_labels = []
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for spectral, labels in tqdm(data_loader, desc='Evaluating'):
                spectral, labels = spectral.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(spectral)
                loss = criterion(outputs, labels)

                # Get predictions
                _, predicted = outputs.max(1)

                # Collect predictions and labels
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_loss += loss.item()

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_predictions, total_loss / len(data_loader))

        if return_predictions:
            return metrics, all_predictions, all_labels
        else:
            return metrics

    def _calculate_metrics(self, y_true, y_pred, avg_loss):
        """Calculate comprehensive evaluation metrics."""
        # Overall accuracy
        accuracy = 100.0 * (y_true == y_pred).sum() / len(y_true)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Per-class metrics
        per_class_acc = 100.0 * cm.diagonal() / cm.sum(axis=1)
        per_class_precision = 100.0 * cm.diagonal() / (cm.sum(axis=0) + 1e-10)

        # Recall (same as per-class accuracy for multi-class)
        mean_recall = per_class_acc.mean()

        # Cohen's Kappa
        kappa = cohen_kappa_score(y_true, y_pred)

        # Compile metrics
        metrics = {
            'loss': avg_loss,
            'overall_accuracy': accuracy,
            'mean_recall': mean_recall,
            'kappa_score': kappa,
            'confusion_matrix': cm,
            'per_class_accuracy': per_class_acc,
            'per_class_precision': per_class_precision,
        }

        return metrics

    def print_metrics(self, metrics):
        """Print evaluation metrics in a formatted table."""
        print(f"\n{'='*70}")
        print("EVALUATION METRICS")
        print(f"{'='*70}")
        print(f"Overall Accuracy:  {metrics['overall_accuracy']:.2f}%")
        print(f"Mean Recall:       {metrics['mean_recall']:.2f}%")
        print(f"Kappa Score:       {metrics['kappa_score']:.4f}")
        print(f"Loss:              {metrics['loss']:.4f}")
        print(f"{'='*70}\n")

        print("PER-CLASS METRICS:")
        print(f"{'Class':<20} {'Accuracy':<12} {'Precision':<12}")
        print(f"{'-'*44}")

        for i, class_name in enumerate(self.class_names):
            if i < len(metrics['per_class_accuracy']):
                acc = metrics['per_class_accuracy'][i]
                prec = metrics['per_class_precision'][i]
                print(f"{class_name:<20} {acc:>6.2f}%      {prec:>6.2f}%")

        print(f"{'='*70}\n")

    def plot_confusion_matrix(self, metrics, save_path=None, normalize=True):
        """
        Plot confusion matrix.

        Args:
            metrics: Dictionary of metrics containing confusion matrix
            save_path: Path to save the figure (optional)
            normalize: Whether to normalize the confusion matrix
        """
        cm = metrics['confusion_matrix']

        if normalize:
            cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
            cm_plot = cm_normalized
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            cm_plot = cm
            fmt = 'd'
            title = 'Confusion Matrix'

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_plot,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names[:len(cm)],
            yticklabels=self.class_names[:len(cm)],
            cbar_kws={'label': 'Percentage' if normalize else 'Count'}
        )

        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {save_path}")

        plt.show()

    def plot_per_class_metrics(self, metrics, save_path=None):
        """
        Plot per-class accuracy and precision.

        Args:
            metrics: Dictionary of metrics
            save_path: Path to save the figure (optional)
        """
        n_classes = len(metrics['per_class_accuracy'])
        x = np.arange(n_classes)
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 6))

        bars1 = ax.bar(x - width/2, metrics['per_class_accuracy'],
                      width, label='Accuracy', color='steelblue')
        bars2 = ax.bar(x + width/2, metrics['per_class_precision'],
                      width, label='Precision', color='coral')

        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('Per-Class Accuracy and Precision', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names[:n_classes], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 105])

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Per-class metrics saved to {save_path}")

        plt.show()

    def plot_training_history(self, history_dict, save_path=None):
        """
        Plot training and validation curves.

        Args:
            history_dict: Dictionary with 'train' and 'val' history
            save_path: Path to save the figure (optional)
        """
        train_history = history_dict['train']
        val_history = history_dict['val']

        epochs = [h['epoch'] for h in train_history]
        train_loss = [h['loss'] for h in train_history]
        train_acc = [h['acc'] for h in train_history]
        val_loss = [h['loss'] for h in val_history]
        val_acc = [h['acc'] for h in val_history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Accuracy plot
        ax2.plot(epochs, train_acc, 'b-', label='Train Acc', linewidth=2)
        ax2.plot(epochs, val_acc, 'r-', label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)

        # Mark best epoch
        if 'best_epoch' in history_dict:
            best_epoch = history_dict['best_epoch']
            best_val_acc = history_dict['best_val_acc']
            ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
            ax2.plot(best_epoch, best_val_acc, 'g*', markersize=15,
                    label=f'Best (Epoch {best_epoch}: {best_val_acc:.2f}%)')
            ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Training history saved to {save_path}")

        plt.show()


def load_model_for_evaluation(checkpoint_path, n_spectral_bands=459, n_classes=11, device='cuda'):
    """
    Load a trained model from checkpoint for evaluation.

    Args:
        checkpoint_path: Path to model checkpoint
        n_spectral_bands: Number of spectral bands
        n_classes: Number of classes
        device: Device to load model on

    Returns:
        Loaded model
    """
    from model import create_model

    model = create_model(n_spectral_bands, n_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print(f"✓ Model loaded from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Best Val Acc: {checkpoint.get('best_val_acc', 'N/A')}")

    return model


if __name__ == '__main__':
    """Example evaluation script."""
    import json
    from dataset import create_dataloaders

    # Configuration
    checkpoint_path = 'checkpoints/best_model.pth'
    data_dir = 'training_dataset'
    label_path = 'Ground_Truth/labels.json'

    # Class names based on labels.json
    class_names = [
        'Background', '95PU', 'HIPS', 'HVDF-HFP', 'GPSS',
        'PU', '75PU', '85PU', 'PETE', 'PET', 'PMMA'
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    print("Loading dataset...")
    train_loader, val_loader = create_dataloaders(
        data_dir, label_path, batch_size=640, max_samples=10000
    )

    # Load model
    print("\nLoading model...")
    model = load_model_for_evaluation(checkpoint_path, n_spectral_bands=459, n_classes=11, device=device)

    # Create evaluator
    evaluator = Evaluator(model, device=device, class_names=class_names)

    # Evaluate
    print("\nEvaluating on validation set...")
    metrics = evaluator.evaluate(val_loader)

    # Print metrics
    evaluator.print_metrics(metrics)

    # Plot confusion matrix
    evaluator.plot_confusion_matrix(metrics, save_path='results/confusion_matrix.png')

    # Plot per-class metrics
    evaluator.plot_per_class_metrics(metrics, save_path='results/per_class_metrics.png')

    # Plot training history
    history_path = 'checkpoints/training_history.json'
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        evaluator.plot_training_history(history, save_path='results/training_history.png')
