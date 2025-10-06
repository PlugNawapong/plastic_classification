"""
Visualize Prediction Noise and Compare Preprocessing Methods

This tool helps identify noise in predictions by:
1. Running inference with different preprocessing methods
2. Visualizing prediction masks with noise highlighting
3. Comparing clean vs noisy band predictions
4. Showing where noise appears (background, edges, between materials)
"""

import numpy as np
import torch
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import median_filter, binary_opening, binary_closing
from typing import Dict, List, Tuple

from preprocessing_pipeline import NoiseRemovalPreprocessor
from model import P1CH_Classifier


class PredictionNoiseAnalyzer:
    """Analyze noise in prediction masks"""

    def __init__(self, model_path: str, dataset_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_path = Path(dataset_path)
        self.preprocessor = NoiseRemovalPreprocessor(dataset_path)

        # Load labels
        with open('labels.json', 'r') as f:
            self.labels = json.load(f)
        self.n_classes = len(self.labels['labels'])

        # Load model
        self.model = P1CH_Classifier(n_spectral_bands=458, n_classes=self.n_classes)
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Loaded model from: {model_path}")
        else:
            print(f"⚠ Model not found: {model_path}")
            print("  Please train the model first or provide correct path")

    def predict_with_preprocessing(self, sample_name: str,
                                   denoise_method: str = 'median',
                                   keep_percentage: float = 75.0) -> Dict:
        """
        Run inference with specified preprocessing

        Returns:
            Dict with prediction mask and intermediate steps
        """
        # Process with pipeline
        results = self.preprocessor.process_with_steps(
            sample_name=sample_name,
            keep_percentage=keep_percentage,
            denoise_method=denoise_method
        )

        # Get normalized hypercube
        hypercube = results['step4_normalized']  # Shape: (n_bands, H, W)
        H, W = hypercube.shape[1:]

        # Prepare for inference
        hypercube_tensor = torch.from_numpy(hypercube).float()  # (n_bands, H, W)
        hypercube_tensor = hypercube_tensor.permute(1, 2, 0)  # (H, W, n_bands)
        hypercube_tensor = hypercube_tensor.reshape(-1, hypercube.shape[0])  # (H*W, n_bands)

        # Run inference
        with torch.no_grad():
            hypercube_tensor = hypercube_tensor.to(self.device)
            logits = self.model(hypercube_tensor)
            predictions = torch.argmax(logits, dim=1)
            confidence = torch.softmax(logits, dim=1).max(dim=1)[0]

        # Reshape to image
        prediction_mask = predictions.cpu().numpy().reshape(H, W)
        confidence_map = confidence.cpu().numpy().reshape(H, W)

        results['prediction_mask'] = prediction_mask
        results['confidence_map'] = confidence_map

        return results

    def analyze_noise_patterns(self, prediction_mask: np.ndarray,
                               ground_truth: np.ndarray = None) -> Dict:
        """
        Analyze noise patterns in prediction mask

        Returns:
            Dict with noise statistics and location
        """
        # Apply morphological operations to identify noise
        # Noise = isolated pixels that differ from neighbors

        # Find connected components
        unique_classes = np.unique(prediction_mask)
        noise_map = np.zeros_like(prediction_mask, dtype=bool)

        for class_id in unique_classes:
            # Get binary mask for this class
            class_mask = (prediction_mask == class_id)

            # Remove small isolated regions (noise)
            # Opening removes small objects
            cleaned_mask = binary_opening(class_mask, structure=np.ones((3, 3)))
            cleaned_mask = binary_closing(cleaned_mask, structure=np.ones((3, 3)))

            # Noise = pixels that were removed
            noise_pixels = class_mask & (~cleaned_mask)
            noise_map |= noise_pixels

        # Classify noise location
        # Dilate prediction mask to find edges
        from scipy.ndimage import binary_dilation

        edge_map = np.zeros_like(prediction_mask, dtype=bool)
        for class_id in unique_classes:
            class_mask = (prediction_mask == class_id)
            dilated = binary_dilation(class_mask, structure=np.ones((3, 3)))
            edge = dilated & (~class_mask)
            edge_map |= edge

        # Background = pixels with class 0 (assuming 0 is background)
        background_map = (prediction_mask == 0)

        # Categorize noise
        noise_in_background = noise_map & background_map
        noise_at_edges = noise_map & edge_map
        noise_in_foreground = noise_map & (~background_map) & (~edge_map)

        total_noise = np.sum(noise_map)
        total_pixels = noise_map.size

        stats = {
            'total_noise_pixels': int(total_noise),
            'noise_percentage': (total_noise / total_pixels) * 100,
            'noise_in_background': int(np.sum(noise_in_background)),
            'noise_at_edges': int(np.sum(noise_at_edges)),
            'noise_in_foreground': int(np.sum(noise_in_foreground)),
            'noise_map': noise_map,
            'edge_map': edge_map,
            'background_map': background_map
        }

        # If ground truth available, calculate accuracy
        if ground_truth is not None:
            correct = (prediction_mask == ground_truth)
            accuracy = np.sum(correct) / correct.size * 100
            stats['accuracy'] = accuracy

        return stats

    def visualize_noise(self, results: Dict, ground_truth: np.ndarray = None,
                       output_path: str = None):
        """
        Visualize prediction with noise highlighting

        Args:
            results: Output from predict_with_preprocessing()
            ground_truth: Optional ground truth mask
            output_path: Where to save visualization
        """
        prediction_mask = results['prediction_mask']
        confidence_map = results['confidence_map']

        # Analyze noise
        noise_stats = self.analyze_noise_patterns(prediction_mask, ground_truth)

        # Create color map for classes
        color_map = np.zeros((self.n_classes, 3), dtype=np.uint8)
        for label_info in self.labels['labels']:
            class_id = label_info['id']
            color_map[class_id] = label_info['color']

        # Convert prediction to RGB
        H, W = prediction_mask.shape
        prediction_rgb = color_map[prediction_mask]

        # Create figure
        n_plots = 6 if ground_truth is not None else 5
        fig, axes = plt.subplots(2, 3, figsize=(20, 13))
        fig.suptitle(f"Prediction Noise Analysis - {results['sample_name']}\n"
                    f"Denoising: {results.get('denoise_method', 'unknown')}",
                    fontsize=16, fontweight='bold')

        # 1. Prediction mask
        ax = axes[0, 0]
        ax.imshow(prediction_rgb)
        ax.set_title(f'Prediction Mask\n({self.n_classes} classes)', fontsize=12, fontweight='bold')
        ax.axis('off')

        # 2. Confidence map
        ax = axes[0, 1]
        im = ax.imshow(confidence_map, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title('Confidence Map\n(Higher = more confident)', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 3. Noise map
        ax = axes[0, 2]
        noise_viz = np.zeros((H, W, 3), dtype=np.uint8)
        noise_viz[noise_stats['noise_map']] = [255, 0, 0]  # Red for noise
        noise_viz[~noise_stats['noise_map']] = [240, 240, 240]  # Light gray for clean

        ax.imshow(noise_viz)
        ax.set_title(f"Noise Pixels (Red)\n"
                    f"Total: {noise_stats['total_noise_pixels']:,} ({noise_stats['noise_percentage']:.2f}%)",
                    fontsize=12, fontweight='bold')
        ax.axis('off')

        # 4. Noise location breakdown
        ax = axes[1, 0]
        noise_location_viz = np.zeros((H, W, 3), dtype=np.uint8)
        noise_location_viz[noise_stats['noise_in_background']] = [255, 100, 100]  # Light red
        noise_location_viz[noise_stats['noise_at_edges']] = [255, 255, 100]  # Yellow
        noise_location_viz[noise_stats['noise_in_foreground']] = [255, 0, 0]  # Red
        noise_location_viz[~noise_stats['noise_map']] = [240, 240, 240]  # Gray

        ax.imshow(noise_location_viz)
        ax.set_title('Noise Location\n'
                    f'Background: {noise_stats["noise_in_background"]:,} | '
                    f'Edges: {noise_stats["noise_at_edges"]:,} | '
                    f'Materials: {noise_stats["noise_in_foreground"]:,}',
                    fontsize=12, fontweight='bold')
        ax.axis('off')

        # Add legend
        legend_elements = [
            Rectangle((0, 0), 1, 1, fc=[1.0, 0.4, 0.4], label='Background noise'),
            Rectangle((0, 0), 1, 1, fc=[1.0, 1.0, 0.4], label='Edge noise'),
            Rectangle((0, 0), 1, 1, fc=[1.0, 0.0, 0.0], label='Material noise')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        # 5. Low confidence regions (likely noise)
        ax = axes[1, 1]
        low_conf_threshold = 0.6
        low_conf_mask = confidence_map < low_conf_threshold

        low_conf_viz = prediction_rgb.copy()
        # Overlay red on low confidence areas
        low_conf_viz[low_conf_mask] = (low_conf_viz[low_conf_mask] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)

        ax.imshow(low_conf_viz)
        ax.set_title(f'Low Confidence Regions (< {low_conf_threshold})\n'
                    f'{np.sum(low_conf_mask):,} pixels ({np.sum(low_conf_mask)/low_conf_mask.size*100:.1f}%)',
                    fontsize=12, fontweight='bold')
        ax.axis('off')

        # 6. Ground truth comparison (if available)
        ax = axes[1, 2]
        if ground_truth is not None:
            error_map = (prediction_mask != ground_truth)
            error_viz = prediction_rgb.copy()
            error_viz[error_map] = [255, 0, 255]  # Magenta for errors

            ax.imshow(error_viz)
            ax.set_title(f'Errors vs Ground Truth\n'
                        f'Accuracy: {noise_stats["accuracy"]:.2f}%',
                        fontsize=12, fontweight='bold')
        else:
            # Show statistics
            ax.axis('off')
            stats_text = f"""
Noise Statistics:

Total pixels:       {H * W:,}
Noise pixels:       {noise_stats['total_noise_pixels']:,}
Noise percentage:   {noise_stats['noise_percentage']:.2f}%

Noise Location:
  Background:       {noise_stats['noise_in_background']:,}
  Edges:            {noise_stats['noise_at_edges']:,}
  Materials:        {noise_stats['noise_in_foreground']:,}

Confidence:
  Mean:             {np.mean(confidence_map):.3f}
  Min:              {np.min(confidence_map):.3f}
  Low conf pixels:  {np.sum(low_conf_mask):,}
            """

            ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
                   fontsize=11, verticalalignment='center', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.axis('off')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Noise visualization saved to: {output_path}")

        plt.show()

        return fig, noise_stats

    def compare_preprocessing_methods(self, sample_name: str,
                                     ground_truth: np.ndarray = None,
                                     output_path: str = None):
        """
        Compare noise in predictions with different preprocessing methods
        """
        print(f"\n{'='*80}")
        print(f"COMPARING PREPROCESSING METHODS: {sample_name}")
        print(f"{'='*80}")

        methods = [
            ('No denoising', None),
            ('Median filter', 'median'),
            ('Bilateral filter', 'bilateral'),
        ]

        results_list = []
        noise_stats_list = []

        for method_name, denoise_method in methods:
            print(f"\n{method_name}...")

            # Run prediction
            if denoise_method:
                results = self.predict_with_preprocessing(
                    sample_name=sample_name,
                    denoise_method=denoise_method,
                    keep_percentage=75.0
                )
            else:
                # No denoising - just normalization
                results = self.preprocessor.process_with_steps(
                    sample_name=sample_name,
                    keep_percentage=75.0,
                    denoise_method='median'  # Will skip in modified version
                )
                # Manually predict without denoising
                hypercube = results['step2_filtered']  # Skip denoising step
                hypercube = self.preprocessor.normalize_bandwise(hypercube)

                H, W = hypercube.shape[1:]
                hypercube_tensor = torch.from_numpy(hypercube).float().permute(1, 2, 0).reshape(-1, hypercube.shape[0])

                with torch.no_grad():
                    hypercube_tensor = hypercube_tensor.to(self.device)
                    logits = self.model(hypercube_tensor)
                    predictions = torch.argmax(logits, dim=1)
                    confidence = torch.softmax(logits, dim=1).max(dim=1)[0]

                results['prediction_mask'] = predictions.cpu().numpy().reshape(H, W)
                results['confidence_map'] = confidence.cpu().numpy().reshape(H, W)

            # Analyze noise
            noise_stats = self.analyze_noise_patterns(results['prediction_mask'], ground_truth)
            noise_stats['method'] = method_name

            results_list.append(results)
            noise_stats_list.append(noise_stats)

        # Visualize comparison
        fig, axes = plt.subplots(3, 3, figsize=(20, 20))
        fig.suptitle(f"Preprocessing Methods Comparison - {sample_name}",
                    fontsize=16, fontweight='bold')

        # Create color map
        color_map = np.zeros((self.n_classes, 3), dtype=np.uint8)
        for label_info in self.labels['labels']:
            class_id = label_info['id']
            color_map[class_id] = label_info['color']

        for idx, (results, noise_stats, (method_name, _)) in enumerate(zip(results_list, noise_stats_list, methods)):
            # Prediction
            ax = axes[idx, 0]
            prediction_rgb = color_map[results['prediction_mask']]
            ax.imshow(prediction_rgb)
            ax.set_title(f'{method_name}\nPrediction', fontsize=12, fontweight='bold')
            ax.axis('off')

            # Noise map
            ax = axes[idx, 1]
            H, W = results['prediction_mask'].shape
            noise_viz = np.zeros((H, W, 3), dtype=np.uint8)
            noise_viz[noise_stats['noise_map']] = [255, 0, 0]
            noise_viz[~noise_stats['noise_map']] = [240, 240, 240]
            ax.imshow(noise_viz)
            ax.set_title(f'Noise: {noise_stats["total_noise_pixels"]:,} pixels\n'
                        f'({noise_stats["noise_percentage"]:.2f}%)',
                        fontsize=12, fontweight='bold')
            ax.axis('off')

            # Confidence
            ax = axes[idx, 2]
            im = ax.imshow(results['confidence_map'], cmap='RdYlGn', vmin=0, vmax=1)
            mean_conf = np.mean(results['confidence_map'])
            ax.set_title(f'Confidence\n(Mean: {mean_conf:.3f})', fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Comparison saved to: {output_path}")

        plt.show()

        # Print summary
        print(f"\n{'='*80}")
        print("NOISE REDUCTION SUMMARY")
        print(f"{'='*80}")
        for noise_stats in noise_stats_list:
            print(f"\n{noise_stats['method']}:")
            print(f"  Noise pixels:    {noise_stats['total_noise_pixels']:,} ({noise_stats['noise_percentage']:.2f}%)")
            print(f"  Background:      {noise_stats['noise_in_background']:,}")
            print(f"  Edges:           {noise_stats['noise_at_edges']:,}")
            print(f"  Materials:       {noise_stats['noise_in_foreground']:,}")
            if 'accuracy' in noise_stats:
                print(f"  Accuracy:        {noise_stats['accuracy']:.2f}%")

        return fig, results_list, noise_stats_list


def main():
    """Example usage"""

    # Initialize analyzer
    analyzer = PredictionNoiseAnalyzer(
        model_path='best_model.pth',  # Adjust if needed
        dataset_path='Inference_dataset1'
    )

    # Get samples
    samples = sorted([d.name for d in Path('Inference_dataset1').iterdir()
                     if d.is_dir() and d.name.startswith('ImageStack')])

    if samples:
        sample_name = samples[0]
        print(f"Analyzing: {sample_name}")

        # Method 1: Analyze single preprocessing method
        results = analyzer.predict_with_preprocessing(
            sample_name=sample_name,
            denoise_method='median',
            keep_percentage=75.0
        )

        analyzer.visualize_noise(
            results=results,
            output_path=f'noise_analysis_{sample_name}.png'
        )

        # Method 2: Compare different preprocessing methods
        analyzer.compare_preprocessing_methods(
            sample_name=sample_name,
            output_path=f'preprocessing_comparison_{sample_name}.png'
        )


if __name__ == '__main__':
    main()
