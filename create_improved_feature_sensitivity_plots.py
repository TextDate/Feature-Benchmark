#!/usr/bin/env python3
"""
Create improved feature domain temporal sensitivity analysis plots.

This script creates clean 3x2 grid plots without the temporal stability subplot,
focusing on the six feature domain performance comparisons.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ImprovedFeatureSensitivityPlotter:
    """Create improved feature domain temporal sensitivity plots with 3x2 grid layout."""

    def __init__(self, base_dir: str):
        """
        Initialize the plotter.

        Args:
            base_dir: Base directory containing Saved_models_binary
        """
        self.base_dir = Path(base_dir)
        self.binary_path = self.base_dir / "Saved_models_binary"

        # Feature types in the desired order for the 3x2 grid
        # Top row: Compression, Lexical Structure, Readability
        # Bottom row: Distance, Neologism, Final Model
        self.feature_types = ['compression', 'lexical_structure', 'readability',
                             'distance', 'neologism', 'final_model']

        self.time_scales = ['decades', 'centuries']
        self.datasets = ['validation', 'test', 'gutenberg']

        # Colors and styles for metrics
        self.metric_colors = {
            'accuracy': '#1f77b4',    # Blue - solid line
            'auc_roc': '#ff7f0e',     # Orange - dashed line
            'auprc': '#2ca02c'        # Green - dotted line
        }

        self.metric_styles = {
            'accuracy': '-',    # solid
            'auc_roc': '--',    # dashed
            'auprc': ':'        # dotted
        }

        # Load binary results
        self.binary_results = self.load_binary_model_results()

    def load_binary_model_results(self) -> dict:
        """Load binary model results from CSV files."""
        logger.info("Loading binary model results...")
        results = {}

        if not self.binary_path.exists():
            logger.warning(f"Binary results directory not found: {self.binary_path}")
            return results

        for feature_type in self.feature_types:
            feature_path = self.binary_path / feature_type
            if not feature_path.exists():
                continue

            results[feature_type] = {}
            for time_scale in self.time_scales:
                time_path = feature_path / time_scale
                if not time_path.exists():
                    continue

                results[feature_type][time_scale] = {}
                for dataset in self.datasets:
                    dataset_file = time_path / f"binary_results_{dataset}.csv"
                    if dataset_file.exists():
                        try:
                            df = pd.read_csv(dataset_file)
                            results[feature_type][time_scale][dataset] = df
                        except Exception as e:
                            logger.warning(f"Error loading {dataset_file}: {e}")

        logger.info(f"Loaded binary results for {len(results)} feature types")
        return results

    def create_feature_domain_sensitivity_plot(self, time_scale: str = 'decades',
                                             figsize: tuple = (18, 12)) -> plt.Figure:
        """
        Create improved feature domain temporal sensitivity plot with 3x2 grid.

        Args:
            time_scale: 'decades' or 'centuries'
            figsize: Figure size tuple

        Returns:
            Figure object
        """
        logger.info(f"Creating improved feature domain sensitivity plot for {time_scale}")

        # Create 3x2 subplot grid
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'Feature Domain Temporal Sensitivity Analysis ({time_scale.title()})',
                    fontsize=20, fontweight='bold', y=0.98)

        # Adjust spacing for better layout
        plt.subplots_adjust(hspace=0.35, wspace=0.25, top=0.85, bottom=0.10, left=0.08, right=0.95)

        # Define subplot positions for 3x2 grid
        # Top row: [0,0], [0,1], [0,2] - Compression, Lexical Structure, Readability
        # Bottom row: [1,0], [1,1], [1,2] - Distance, Neologism, Final Model
        subplot_positions = [
            (0, 0), (0, 1), (0, 2),  # Top row
            (1, 0), (1, 1), (1, 2)   # Bottom row
        ]

        # Plot each feature type
        for idx, feature_type in enumerate(self.feature_types):
            row, col = subplot_positions[idx]
            ax = axes[row, col]

            if (feature_type in self.binary_results and
                time_scale in self.binary_results[feature_type]):

                # Use test dataset for primary analysis
                if 'test' in self.binary_results[feature_type][time_scale]:
                    df = self.binary_results[feature_type][time_scale]['test']

                    # Filter for CatBoost model (most stable performer)
                    model_data = df[df['model'].str.contains('catboost', na=False)]

                    if not model_data.empty:
                        # Get thresholds
                        thresholds = model_data['threshold'].values
                        if time_scale == 'centuries':
                            thresholds = thresholds / 100  # Convert to century scale

                        # Plot the three main metrics
                        for metric in ['accuracy', 'auc_roc', 'auprc']:
                            if metric in model_data.columns:
                                metric_values = model_data[metric].values

                                ax.plot(thresholds, metric_values,
                                       color=self.metric_colors[metric],
                                       linestyle=self.metric_styles[metric],
                                       linewidth=3.0, alpha=0.85,
                                       label=metric.replace('_', ' ').upper(),
                                       zorder=2)

                        # Add random baseline if available
                        random_data = df[df['model'] == 'random']
                        if not random_data.empty and 'accuracy' in random_data.columns:
                            random_thresholds = random_data['threshold'].values
                            if time_scale == 'centuries':
                                random_thresholds = random_thresholds / 100
                            random_accuracy = random_data['accuracy'].values

                            ax.plot(random_thresholds, random_accuracy,
                                   color='red', linestyle='-', alpha=0.5, linewidth=2,
                                   label='Random', zorder=1)

            # Customize subplot
            ax.set_title(feature_type.replace('_', ' ').title(),
                        fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Temporal Threshold', fontsize=12)
            ax.set_ylabel('Performance Score', fontsize=12)
            ax.grid(True, alpha=0.3, color='gray', linewidth=0.5, zorder=0)
            ax.set_ylim(0.0, 1.05)
            ax.tick_params(axis='both', labelsize=11)

            # Add legend only to the first subplot (top-left)
            if idx == 0:
                legend = ax.legend(bbox_to_anchor=(0.02, 0.98), loc='upper left',
                                 fontsize=11, frameon=True, fancybox=True, shadow=True)
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_alpha(0.9)

        return fig

    def save_improved_plots(self, output_dir: str = None):
        """
        Create and save improved plots for both decades and centuries.

        Args:
            output_dir: Directory to save plots (default: base_dir/Thesis/figs)
        """
        if output_dir is None:
            output_dir = self.base_dir / "Thesis" / "figs"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        for time_scale in self.time_scales:
            logger.info(f"Creating improved plot for {time_scale}")

            # Create the plot
            fig = self.create_feature_domain_sensitivity_plot(time_scale, figsize=(18, 12))

            # Save the plot
            filename = f"feature_domain_sensitivity_{time_scale}_fixed_3x2.png"
            save_path = output_dir / filename

            fig.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close(fig)

            logger.info(f"Saved improved plot: {save_path}")

        logger.info("All improved plots created successfully!")


def main():
    """Main function."""
    # Set base directory
    base_dir = "/data/home/paulojnpinto02/TextDate/Feature-Benchmark"

    # Initialize plotter
    plotter = ImprovedFeatureSensitivityPlotter(base_dir)

    # Create and save improved plots
    plotter.save_improved_plots()

    logger.info("Improved feature domain sensitivity plots completed!")


if __name__ == "__main__":
    main()