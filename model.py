"""
P1CH (Pixel-wise 1D Convolutional Hyperspectral) Classifier
Industrial-standard 1D CNN for plastic classification based on hyperspectral imaging.

Reference: "A Deep Learning Approach for Pixel-level Material Classification
via Hyperspectral Imaging"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """1D Residual Block with batch normalization and skip connections."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection: adjust channels if needed
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out)

        return out


class P1CH_Classifier(nn.Module):
    """
    P1CH: Pixel-wise 1D Convolutional Hyperspectral Classifier

    Architecture:
    - Input: (batch_size, 1, n_spectral_bands)
    - 2 Conv1D layers with pooling
    - 2 Residual blocks
    - Fully connected layers with dropout
    - Output: class probabilities
    """

    def __init__(self, n_spectral_bands=224, n_classes=11, dropout_rate=0.5):
        """
        Args:
            n_spectral_bands: Number of spectral bands (wavelengths)
            n_classes: Number of plastic types + background
            dropout_rate: Dropout probability for regularization
        """
        super(P1CH_Classifier, self).__init__()

        self.n_spectral_bands = n_spectral_bands
        self.n_classes = n_classes

        # Initial convolutional layers
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Residual blocks
        self.res_block1 = ResidualBlock1D(32, 64, kernel_size=3)
        self.res_block2 = ResidualBlock1D(64, 128, kernel_size=3)

        # Calculate flattened size after convolutions and pooling
        # After 2 pooling layers with kernel=2, stride=2: n_bands / 4
        self.flattened_size = 128 * (n_spectral_bands // 4)

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, n_classes)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, 1, n_spectral_bands)

        Returns:
            Output logits of shape (batch_size, n_classes)
        """
        # Initial conv layers with ReLU and pooling
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_num_params(self):
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(n_spectral_bands=224, n_classes=11, dropout_rate=0.5):
    """
    Factory function to create P1CH classifier model.

    Args:
        n_spectral_bands: Number of spectral bands
        n_classes: Number of classes (11 for the given dataset)
        dropout_rate: Dropout rate for regularization

    Returns:
        P1CH_Classifier model instance
    """
    model = P1CH_Classifier(n_spectral_bands, n_classes, dropout_rate)
    print(f"Model created with {model.get_num_params():,} trainable parameters")
    return model


if __name__ == "__main__":
    # Test the model
    model = create_model(n_spectral_bands=224, n_classes=11)

    # Test with dummy input
    batch_size = 32
    dummy_input = torch.randn(batch_size, 1, 224)
    output = model(dummy_input)

    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nModel architecture:")
    print(model)
