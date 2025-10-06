"""
Inference pipeline for P1CH plastic classifier.
Performs pixel-level classification on hyperspectral images and generates segmentation maps.
"""

import os
import glob
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, binary_opening, binary_closing


class InferencePipeline:
    """Inference manager for P1CH classifier with post-processing."""

    def __init__(self, model, device='cuda', class_names=None, apply_postprocessing=True):
        """
        Args:
            model: Trained P1CH model
            device: Device to run inference on
            class_names: List of class names
            apply_postprocessing: Whether to apply morphological post-processing
        """
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names or [f'Class_{i}' for i in range(11)]
        self.apply_postprocessing = apply_postprocessing

        # Class colors from labels.json
        self.class_colors = {
            0: (0, 0, 0),       # Background - Black
            1: (255, 0, 0),     # 95PU - Red
            2: (0, 0, 255),     # HIPS - Blue
            3: (255, 125, 125), # HVDF-HFP - Light Red
            4: (255, 255, 0),   # GPSS - Yellow
            5: (0, 125, 125),   # PU - Teal
            6: (0, 200, 255),   # 75PU - Light Blue
            7: (255, 0, 255),   # 85PU - Magenta
            8: (0, 255, 0),     # PETE - Green
            9: (255, 125, 0),   # PET - Orange
            10: (255, 0, 100),  # PMMA - Pink
        }

        self.model.eval()

    def load_hyperspectral_image(self, image_dir, normalize=True):
        """
        Load hyperspectral image from directory of spectral bands.

        Args:
            image_dir: Directory containing ImagesStack*.png files
            normalize: Whether to normalize spectral values

        Returns:
            spectral_cube: Numpy array of shape (n_bands, height, width)
        """
        # Find all spectral band files
        image_files = sorted(glob.glob(os.path.join(image_dir, 'ImagesStack*.png')))

        if len(image_files) == 0:
            raise FileNotFoundError(f"No ImagesStack*.png files found in {image_dir}")

        # Load first image to get dimensions
        first_img = np.array(Image.open(image_files[0]))
        height, width = first_img.shape
        n_bands = len(image_files)

        # Pre-allocate array
        spectral_cube = np.zeros((n_bands, height, width), dtype=np.float32)

        # Load all bands
        print(f"Loading {n_bands} spectral bands from {image_dir}...")
        for i, img_path in enumerate(tqdm(image_files, desc="Loading bands")):
            img = np.array(Image.open(img_path))
            spectral_cube[i] = img.astype(np.float32)

        # Normalize if requested
        if normalize:
            max_val = spectral_cube.max()
            if max_val > 0:
                spectral_cube = spectral_cube / max_val

        print(f"Loaded hyperspectral cube: {spectral_cube.shape}")
        return spectral_cube

    def predict_image(self, spectral_cube, batch_size=640):
        """
        Perform pixel-level classification on hyperspectral image.

        Args:
            spectral_cube: Numpy array of shape (n_bands, height, width)
            batch_size: Batch size for inference

        Returns:
            prediction_map: Numpy array of shape (height, width) with class predictions
        """
        n_bands, height, width = spectral_cube.shape

        # Reshape to (n_pixels, n_bands)
        spectral_pixels = spectral_cube.reshape(n_bands, -1).T

        # Predict in batches
        predictions = []
        n_pixels = spectral_pixels.shape[0]

        with torch.no_grad():
            for i in tqdm(range(0, n_pixels, batch_size), desc="Predicting"):
                batch = spectral_pixels[i:i+batch_size]

                # Convert to tensor and add channel dimension
                batch_tensor = torch.from_numpy(batch).float().unsqueeze(1).to(self.device)

                # Forward pass
                outputs = self.model(batch_tensor)
                _, predicted = outputs.max(1)

                predictions.extend(predicted.cpu().numpy())

        # Reshape to image dimensions
        prediction_map = np.array(predictions).reshape(height, width)

        return prediction_map

    def postprocess(self, prediction_map, median_kernel=5, morph_kernel=3):
        """
        Apply post-processing to reduce noise and smooth boundaries.

        Args:
            prediction_map: Predicted class map
            median_kernel: Kernel size for median filtering
            morph_kernel: Kernel size for morphological operations

        Returns:
            Processed prediction map
        """
        if not self.apply_postprocessing:
            return prediction_map

        processed = prediction_map.copy()

        # Median filter to remove salt-and-pepper noise
        processed = median_filter(processed, size=median_kernel)

        # Morphological opening (erosion + dilation) to remove small objects
        # Morphological closing (dilation + erosion) to fill small holes
        # Apply per class to maintain boundaries
        unique_classes = np.unique(processed)

        for class_id in unique_classes:
            if class_id == 0:  # Skip background
                continue

            # Create binary mask for this class
            mask = (processed == class_id).astype(np.uint8)

            # Opening: remove small objects
            mask = binary_opening(mask, structure=np.ones((morph_kernel, morph_kernel)))

            # Closing: fill small holes
            mask = binary_closing(mask, structure=np.ones((morph_kernel, morph_kernel)))

            # Update processed map
            processed[mask.astype(bool)] = class_id

        return processed

    def prediction_to_rgb(self, prediction_map):
        """
        Convert prediction map to RGB image using class colors.

        Args:
            prediction_map: Predicted class map

        Returns:
            RGB image as numpy array
        """
        height, width = prediction_map.shape
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

        for class_id, color in self.class_colors.items():
            mask = prediction_map == class_id
            rgb_image[mask] = color

        return rgb_image

    def visualize_prediction(self, spectral_cube, prediction_map, save_path=None):
        """
        Visualize prediction results.

        Args:
            spectral_cube: Original spectral cube
            prediction_map: Predicted class map
            save_path: Path to save visualization (optional)
        """
        # Convert prediction to RGB
        rgb_prediction = self.prediction_to_rgb(prediction_map)

        # Use middle spectral band as grayscale reference
        middle_band = spectral_cube[spectral_cube.shape[0] // 2]

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Original (middle band)
        axes[0].imshow(middle_band, cmap='gray')
        axes[0].set_title('Original Image (Middle Band)', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Prediction
        axes[1].imshow(rgb_prediction)
        axes[1].set_title('Predicted Classification', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=tuple(c/255 for c in self.class_colors[i]),
                 label=self.class_names[i])
            for i in range(len(self.class_names))
        ]
        axes[1].legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")

        plt.show()

    def run_inference(self, image_dir, output_dir=None, visualize=True):
        """
        Run complete inference pipeline on a hyperspectral image.

        Args:
            image_dir: Directory containing spectral bands
            output_dir: Directory to save results (optional)
            visualize: Whether to visualize results

        Returns:
            prediction_map: Predicted class map
        """
        # Load image
        spectral_cube = self.load_hyperspectral_image(image_dir, normalize=True)

        # Predict
        print("\nPerforming pixel-level classification...")
        prediction_map = self.predict_image(spectral_cube, batch_size=640)

        # Post-process
        if self.apply_postprocessing:
            print("\nApplying post-processing...")
            prediction_map = self.postprocess(prediction_map)

        # Save results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # Save prediction map as numpy file
            pred_path = os.path.join(output_dir, 'prediction_map.npy')
            np.save(pred_path, prediction_map)
            print(f"✓ Prediction map saved to {pred_path}")

            # Save RGB visualization
            rgb_prediction = self.prediction_to_rgb(prediction_map)
            rgb_path = os.path.join(output_dir, 'prediction_rgb.png')
            Image.fromarray(rgb_prediction).save(rgb_path)
            print(f"✓ RGB prediction saved to {rgb_path}")

            # Save class distribution
            unique, counts = np.unique(prediction_map, return_counts=True)
            dist_path = os.path.join(output_dir, 'class_distribution.txt')
            with open(dist_path, 'w') as f:
                f.write("Class Distribution:\n")
                f.write("="*50 + "\n")
                for class_id, count in zip(unique, counts):
                    percentage = 100 * count / prediction_map.size
                    f.write(f"{self.class_names[class_id]:<15}: {count:>8} pixels ({percentage:>5.2f}%)\n")
            print(f"✓ Class distribution saved to {dist_path}")

        # Visualize
        if visualize:
            vis_path = os.path.join(output_dir, 'visualization.png') if output_dir else None
            self.visualize_prediction(spectral_cube, prediction_map, save_path=vis_path)

        return prediction_map


def load_model_for_inference(checkpoint_path, n_spectral_bands=459, n_classes=11, device='cuda'):
    """Load trained model for inference."""
    from model import create_model

    model = create_model(n_spectral_bands, n_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print(f"✓ Model loaded from {checkpoint_path}")
    return model


if __name__ == '__main__':
    """Run inference on test datasets."""

    # Configuration
    checkpoint_path = 'checkpoints/best_model.pth'
    test_datasets = ['Inference_dataset1', 'Inference_dataset2', 'Inference_dataset3']

    # Class names
    class_names = [
        'Background', '95PU', 'HIPS', 'HVDF-HFP', 'GPSS',
        'PU', '75PU', '85PU', 'PETE', 'PET', 'PMMA'
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load model
    print("Loading model...")
    model = load_model_for_inference(checkpoint_path, n_spectral_bands=459, n_classes=11, device=device)

    # Create inference pipeline
    pipeline = InferencePipeline(
        model,
        device=device,
        class_names=class_names,
        apply_postprocessing=True
    )

    # Run inference on each test dataset
    for dataset_name in test_datasets:
        print(f"\n{'='*70}")
        print(f"Processing {dataset_name}")
        print(f"{'='*70}")

        if not os.path.exists(dataset_name):
            print(f"⚠ {dataset_name} not found, skipping...")
            continue

        output_dir = os.path.join('results', dataset_name)
        prediction_map = pipeline.run_inference(
            dataset_name,
            output_dir=output_dir,
            visualize=True
        )

        print(f"✓ {dataset_name} processed successfully\n")

    print(f"\n{'='*70}")
    print("All inference tasks completed!")
    print(f"{'='*70}")
