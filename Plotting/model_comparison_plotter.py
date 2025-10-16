"""
Model Comparison Plotting Script for TextDate Feature-Benchmark Project

This script analyzes and visualizes model results to compare:
- Between Datasets: test vs gutenberg vs validation datasets
- Between Feature Types: compression, lexical_structure, readability, distance, final_model
- Between Models: random_forest, xgboost, catboost, svm, gnb, knn
- Between Time Scales: decades vs centuries performance
- Binary vs Base Models: base model results vs binary model results
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
from itertools import combinations
import logging

# Check for required packages
required_packages = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'matplotlib.pyplot': 'matplotlib',
    'seaborn': 'seaborn',
    'sklearn.metrics': 'scikit-learn'
}

missing_packages = []
for module_name, package_name in required_packages.items():
    try:
        __import__(module_name)
    except ImportError:
        missing_packages.append(package_name)

if missing_packages:
    print("Error: Missing required packages. Please install them using:")
    print(f"pip install {' '.join(missing_packages)}")
    print("\nOr install from requirements.txt:")
    print("pip install -r requirements.txt")
    sys.exit(1)

# Import packages after verification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelResultsAnalyzer:
    """Main class for analyzing and plotting model comparison results."""

    def __init__(self, base_dir: str, output_dir: str = None):
        """
        Initialize the analyzer with base directory containing results.

        Args:
            base_dir: Base directory containing Saved_models_results and Saved_models_binary
            output_dir: Directory to save plots (default: base_dir/comparison_plots)
        """
        
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir) if output_dir else self.base_dir / "comparison_plots"
        self.output_dir.mkdir(exist_ok=True)

        # Define constants first
        self.FEATURE_TYPES = ['compression', 'lexical_structure', 'readability', 'distance', 'neologism', 'final_model']
        self.TIME_SCALES = ['decades', 'centuries']
        self.BASE_MODELS = ['random_forest', 'xgboost', 'catboost', 'knn','svm', 'gnb']
        self.DATASETS = ['validation','test', 'gutenberg']

        # Create subdirectories for different plot types
        self.heatmaps_dir = self.output_dir / "heatmaps"
        self.binary_dir = self.output_dir / "binary_comparisons"
        self.dataset_dir = self.output_dir / "dataset_comparisons"
        self.timescale_dir = self.output_dir / "time_scale_comparisons"

        # Create main directories
        for plot_dir in [self.heatmaps_dir, self.binary_dir, self.dataset_dir, self.timescale_dir]:
            plot_dir.mkdir(exist_ok=True)

        # Create subdirectories for heatmaps (by dataset)
        for dataset in self.DATASETS:
            (self.heatmaps_dir / dataset).mkdir(exist_ok=True)

        # Create subdirectories for binary and dataset comparisons (by time scale)
        for time_scale in self.TIME_SCALES:
            (self.binary_dir / time_scale).mkdir(exist_ok=True)
            (self.dataset_dir / time_scale).mkdir(exist_ok=True)

        # Create subdirectories for time scale comparisons (by model)
        for model in self.BASE_MODELS:
            (self.timescale_dir / model).mkdir(exist_ok=True)

        # Metrics available in different model types
        self.BASE_METRICS = [
            'accuracy', 'f1_macro', 'f1_weighted', 'recall_macro', 'recall_weighted',
            'precision_macro', 'precision_weighted', 'rmse', 'mae', 'r2', 'auc_roc',
            'auprc', 'top3_accuracy', 'top5_accuracy', 'top10_accuracy'
        ]

        self.BINARY_METRICS = [
            'accuracy', 'auc_roc', 'auprc', 'f1_macro', 'f1_weighted',
            'recall_macro', 'recall_weighted', 'precision_macro', 'precision_weighted'
        ]

        # Color schemes
        self.COLORS = {
            'feature_types': sns.color_palette("Set2", len(self.FEATURE_TYPES)),
            'models': sns.color_palette("tab10", len(self.BASE_MODELS)),
            'datasets': sns.color_palette("viridis", len(self.DATASETS)),
            'time_scales': ['#FF6B6B', '#4ECDC4']
        }

        # Storage for loaded data
        self.base_results = {}
        self.binary_results = {}

    def load_base_model_results(self) -> Dict:
        """Load base model results from CSV files."""
        logger.info("Loading base model results...")
        results = {}

        base_path = self.base_dir / "Saved_models_results"
        if not base_path.exists():
            logger.warning(f"Base results directory not found: {base_path}")
            return results

        for feature_type in self.FEATURE_TYPES:
            feature_path = base_path / feature_type
            if not feature_path.exists():
                continue

            results[feature_type] = {}
            for time_scale in self.TIME_SCALES:
                time_path = feature_path / time_scale
                if not time_path.exists():
                    continue

                results[feature_type][time_scale] = {}
                for dataset in self.DATASETS:
                    results[feature_type][time_scale][dataset] = {}
                    for model in self.BASE_MODELS:
                        model_file = time_path / f"{model}_model_results_{dataset}.csv"
                        if model_file.exists():
                            try:
                                df = pd.read_csv(model_file)
                                if len(df) > 0:
                                    results[feature_type][time_scale][dataset][model] = df.iloc[0].to_dict()
                            except Exception as e:
                                logger.warning(f"Error loading {model_file}: {e}")

        logger.info(f"Loaded base results for {len(results)} feature types")
        return results

    def load_binary_model_results(self) -> Dict:
        """Load binary model results from CSV files."""
        logger.info("Loading binary model results...")
        results = {}

        binary_path = self.base_dir / "Saved_models_binary"
        if not binary_path.exists():
            logger.warning(f"Binary results directory not found: {binary_path}")
            return results

        for feature_type in self.FEATURE_TYPES:
            feature_path = binary_path / feature_type
            if not feature_path.exists():
                continue

            results[feature_type] = {}
            for time_scale in self.TIME_SCALES:
                time_path = feature_path / time_scale
                if not time_path.exists():
                    continue

                results[feature_type][time_scale] = {}
                for dataset in self.DATASETS:
                    dataset_file = time_path / f"binary_results_{dataset}.csv"
                    if dataset_file.exists():
                        try:
                            df = pd.read_csv(dataset_file)
                            results[feature_type][time_scale][dataset] = df
                        except Exception as e:
                            logger.warning(f"Error loading {dataset_file}: {e}")

        logger.info(f"Loaded binary results for {len(results)} feature types")
        return results

    def load_all_data(self):
        """Load all available data."""
        self.base_results = self.load_base_model_results()
        self.binary_results = self.load_binary_model_results()
        logger.info("All data loaded successfully")

    def create_base_model_heatmap(self, metric: str = 'accuracy', time_scale: str = 'decades',
                                  dataset: str = 'test', figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Create heatmap comparing base models across feature types."""
        logger.info(f"Creating base model heatmap for {metric} ({time_scale})")

        # Prepare data matrix
        data_matrix = []
        feature_labels = []
        model_labels = self.BASE_MODELS

        for feature_type in self.FEATURE_TYPES:
            if (feature_type in self.base_results and
                time_scale in self.base_results[feature_type] and
                dataset in self.base_results[feature_type][time_scale]):
                row = []
                for model in self.BASE_MODELS:
                    if model in self.base_results[feature_type][time_scale][dataset]:
                        value = self.base_results[feature_type][time_scale][dataset][model].get(metric, np.nan)
                        row.append(value)
                    else:
                        row.append(np.nan)
                data_matrix.append(row)
                feature_labels.append(feature_type.replace('_', ' ').title())

        if not data_matrix:
            logger.warning(f"No data available for {metric} heatmap")
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        data_df = pd.DataFrame(data_matrix, index=feature_labels, columns=model_labels)
        sns.heatmap(data_df, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0.5,
                   square=True, cbar_kws={'label': metric.replace('_', ' ').title()}, ax=ax)

        # Customize plot
        ax.set_title(f'Base Model Performance Comparison - {metric.replace("_", " ").title()}\n'
                    f'Time Scale: {time_scale.title()}, Dataset: {dataset.title()}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature Types', fontsize=12, fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save plot
        filename = f"base_model_heatmap_{metric}_{time_scale}.png"
        plt.savefig(self.heatmaps_dir / dataset / filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved heatmap: heatmaps/{dataset}/{filename}")

        return fig

    def create_binary_model_comparison(self, metric: str = 'accuracy', time_scale: str = 'decades',
                                     figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Create comparison plot for binary models across datasets."""
        logger.info(f"Creating binary model comparison for {metric} ({time_scale})")

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        for idx, feature_type in enumerate(self.FEATURE_TYPES):
            if idx >= len(axes):
                break

            ax = axes[idx]

            if (feature_type in self.binary_results and
                time_scale in self.binary_results[feature_type]):

                data_for_plot = []
                models_for_plot = []
                datasets_for_plot = []

                for dataset in self.DATASETS:
                    if dataset in self.binary_results[feature_type][time_scale]:
                        df = self.binary_results[feature_type][time_scale][dataset]
                        if metric in df.columns:
                            for _, row in df.iterrows():
                                model = row['model'].replace('_model', '')
                                if model in self.BASE_MODELS:
                                    data_for_plot.append(row[metric])
                                    models_for_plot.append(model)
                                    datasets_for_plot.append(dataset)

                if data_for_plot:
                    # Create DataFrame for plotting
                    plot_df = pd.DataFrame({
                        'Model': models_for_plot,
                        'Dataset': datasets_for_plot,
                        metric: data_for_plot
                    })

                    # Create bar plot
                    sns.barplot(data=plot_df, x='Model', y=metric, hue='Dataset', ax=ax)
                    ax.set_title(f'{feature_type.replace("_", " ").title()}', fontweight='bold')
                    ax.set_xlabel('Model')
                    ax.set_ylabel(metric.replace('_', ' ').title())
                    ax.tick_params(axis='x', rotation=45)

                    if idx == 0:  # Only show legend for first subplot
                        ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
                    else:
                        ax.legend().remove()
            else:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{feature_type.replace("_", " ").title()}', fontweight='bold')

        # Remove extra subplots
        for idx in range(len(self.FEATURE_TYPES), len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle(f'Binary Model Performance Comparison - {metric.replace("_", " ").title()}\n'
                    f'Time Scale: {time_scale.title()}', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save plot
        filename = f"binary_model_comparison_{metric}.png"
        plt.savefig(self.binary_dir / time_scale / filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved binary comparison: binary_comparisons/{time_scale}/{filename}")

        return fig

    def create_performance_summary_table(self, time_scale: str = 'decades') -> pd.DataFrame:
        """Create summary table of best performing models."""
        logger.info(f"Creating performance summary table for {time_scale}")

        summary_data = []

        # Base model summary
        for feature_type in self.FEATURE_TYPES:
            if (feature_type in self.base_results and
                time_scale in self.base_results[feature_type]):

                feature_data = self.base_results[feature_type][time_scale]

                for metric in ['accuracy', 'f1_macro', 'auc_roc']:
                    if metric in self.BASE_METRICS:
                        best_model = None
                        best_score = -1

                        for model, results in feature_data.items():
                            if metric in results and not pd.isna(results[metric]):
                                if results[metric] > best_score:
                                    best_score = results[metric]
                                    best_model = model

                        if best_model:
                            summary_data.append({
                                'Feature_Type': feature_type,
                                'Time_Scale': time_scale,
                                'Model_Type': 'Base',
                                'Metric': metric,
                                'Best_Model': best_model,
                                'Best_Score': best_score,
                                'Dataset': 'N/A'
                            })

        # Binary model summary
        for feature_type in self.FEATURE_TYPES:
            if (feature_type in self.binary_results and
                time_scale in self.binary_results[feature_type]):

                for dataset in self.DATASETS:
                    if dataset in self.binary_results[feature_type][time_scale]:
                        df = self.binary_results[feature_type][time_scale][dataset]

                        for metric in ['accuracy', 'f1_macro', 'auc_roc']:
                            if metric in df.columns:
                                best_idx = df[metric].idxmax()
                                if not pd.isna(best_idx):
                                    best_row = df.loc[best_idx]
                                    summary_data.append({
                                        'Feature_Type': feature_type,
                                        'Time_Scale': time_scale,
                                        'Model_Type': 'Binary',
                                        'Metric': metric,
                                        'Best_Model': best_row['model'],
                                        'Best_Score': best_row[metric],
                                        'Dataset': dataset
                                    })

        summary_df = pd.DataFrame(summary_data)

        # Save summary table
        filename = f"performance_summary_{time_scale}.csv"
        summary_df.to_csv(self.output_dir / filename, index=False)
        logger.info(f"Saved summary table: {filename}")

        return summary_df

    def create_dataset_comparison_plot(self, metric: str = 'accuracy', time_scale: str = 'decades',
                                     figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
        """Create comparison plot between test and gutenberg datasets."""
        logger.info(f"Creating dataset comparison plot for {metric} ({time_scale})")

        fig, ax = plt.subplots(figsize=figsize)

        # Collect data for comparison
        comparison_data = []

        for feature_type in self.FEATURE_TYPES:
            if (feature_type in self.binary_results and
                time_scale in self.binary_results[feature_type]):

                feature_results = self.binary_results[feature_type][time_scale]

                # Get test and gutenberg results
                test_data = feature_results.get('test')
                gutenberg_data = feature_results.get('gutenberg')

                if test_data is not None and gutenberg_data is not None:
                    if metric in test_data.columns and metric in gutenberg_data.columns:
                        # Match models between datasets
                        test_models = set(test_data['model'].values)
                        gutenberg_models = set(gutenberg_data['model'].values)
                        common_models = test_models.intersection(gutenberg_models)

                        for model in common_models:
                            test_score = test_data[test_data['model'] == model][metric].iloc[0]
                            gutenberg_score = gutenberg_data[gutenberg_data['model'] == model][metric].iloc[0]

                            comparison_data.append({
                                'Feature_Type': feature_type.replace('_', ' ').title(),
                                'Model': model.replace('_model', ''),
                                'Test': test_score,
                                'Gutenberg': gutenberg_score,
                                'Difference': gutenberg_score - test_score
                            })

        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)

            # Create scatter plot
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.FEATURE_TYPES)))

            for i, feature_type in enumerate(comp_df['Feature_Type'].unique()):
                feature_data = comp_df[comp_df['Feature_Type'] == feature_type]
                ax.scatter(feature_data['Test'], feature_data['Gutenberg'],
                          label=feature_type, alpha=0.7, s=100, color=colors[i])

            # Add diagonal line (x=y)
            min_val = min(comp_df['Test'].min(), comp_df['Gutenberg'].min())
            max_val = max(comp_df['Test'].max(), comp_df['Gutenberg'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Equal Performance')

            ax.set_xlabel(f'Test Dataset {metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'Gutenberg Dataset {metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_title(f'Dataset Performance Comparison - {metric.replace("_", " ").title()}\n'
                        f'Time Scale: {time_scale.title()}', fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        filename = f"dataset_comparison_{metric}.png"
        plt.savefig(self.dataset_dir / time_scale / filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved dataset comparison: dataset_comparisons/{time_scale}/{filename}")

        return fig

    def create_time_scale_comparisons_per_model(self, metric: str = 'accuracy', dataset: str = 'test',
                                              figsize: Tuple[int, int] = (10, 6)) -> List[plt.Figure]:
        """Compare performance between decades and centuries, creating separate plots for each model."""
        logger.info(f"Creating time scale comparisons per model for {metric}")

        figures = []

        # Collect all data first
        all_data = {}
        for feature_type in self.FEATURE_TYPES:
            if feature_type in self.base_results:
                decades_data = self.base_results[feature_type].get('decades', {}).get(dataset, {})
                centuries_data = self.base_results[feature_type].get('centuries', {}).get(dataset, {})

                # Find common models
                common_models = set(decades_data.keys()).intersection(set(centuries_data.keys()))

                for model in common_models:
                    if (metric in decades_data[model] and metric in centuries_data[model] and
                        not pd.isna(decades_data[model][metric]) and not pd.isna(centuries_data[model][metric])):

                        if model not in all_data:
                            all_data[model] = []

                        all_data[model].append({
                            'Feature_Type': feature_type.replace('_', ' ').title(),
                            'Decades': decades_data[model][metric],
                            'Centuries': centuries_data[model][metric],
                            'Difference': centuries_data[model][metric] - decades_data[model][metric]
                        })

        # Create separate plot for each model
        for model, model_data in all_data.items():
            if not model_data:
                continue

            fig, ax = plt.subplots(figsize=figsize)

            model_df = pd.DataFrame(model_data)
            x_pos = np.arange(len(model_df))
            width = 0.35

            # Create bars
            ax.bar(x_pos - width/2, model_df['Decades'], width, label='Decades',
                  alpha=0.8, color='#FF6B6B')
            ax.bar(x_pos + width/2, model_df['Centuries'], width, label='Centuries',
                  alpha=0.8, color='#4ECDC4')

            # Customize plot
            ax.set_xlabel('Feature Types', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_title(f'{model.replace("_", " ").title()} - Time Scale Comparison\n{metric.replace("_", " ").title()}',
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_df['Feature_Type'], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()

            # Save plot
            filename = f"time_scale_comparison_{metric}.png"
            plt.savefig(self.timescale_dir / model / filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved time scale comparison: time_scale_comparisons/{model}/{filename}")

            figures.append(fig)

        return figures

    def create_comprehensive_report(self, metrics: List[str] = None):
        """Create comprehensive report with all visualizations."""
        if metrics is None:
            # Use all available metrics from both base and binary models
            metrics = list(set(self.BASE_METRICS + self.BINARY_METRICS))

        logger.info("Creating comprehensive report...")

        # Load all data
        self.load_all_data()

        report_summary = {
            'plots_created': [],
            'summary_tables': [],
            'metrics_analyzed': metrics,
            'total_plots': 0
        }

        for time_scale in self.TIME_SCALES:
            # Create base model heatmaps
            for metric in metrics:
                # Create heatmaps for both test and gutenberg datasets
                for dataset in ['test', 'gutenberg']:
                    fig = self.create_base_model_heatmap(metric, time_scale, dataset)
                    if fig is not None:
                        report_summary['plots_created'].append(f'heatmaps/{dataset}/base_model_heatmap_{metric}_{time_scale}.png')
                        report_summary['total_plots'] += 1
                    plt.close(fig)

            # Create binary model comparisons (only for binary metrics)
            for metric in metrics:
                if metric in self.BINARY_METRICS:
                    fig = self.create_binary_model_comparison(metric, time_scale)
                    if fig is not None:
                        report_summary['plots_created'].append(f'binary_comparisons/{time_scale}/binary_model_comparison_{metric}.png')
                        report_summary['total_plots'] += 1
                    plt.close(fig)

            # Create dataset comparisons (only for binary metrics)
            for metric in metrics:
                if metric in self.BINARY_METRICS:
                    fig = self.create_dataset_comparison_plot(metric, time_scale)
                    if fig is not None:
                        report_summary['plots_created'].append(f'dataset_comparisons/{time_scale}/dataset_comparison_{metric}.png')
                        report_summary['total_plots'] += 1
                    plt.close(fig)

            # Create summary table
            summary_df = self.create_performance_summary_table(time_scale)
            if not summary_df.empty:
                report_summary['summary_tables'].append(f'performance_summary_{time_scale}.csv')

        # Create time scale comparisons per model
        for metric in metrics:
            figures = self.create_time_scale_comparisons_per_model(metric)
            for model in self.BASE_MODELS:
                # Add expected plot paths for each model
                model_plot_path = f'time_scale_comparisons/{model}/time_scale_comparison_{metric}.png'
                report_summary['plots_created'].append(model_plot_path)
                report_summary['total_plots'] += 1
            # Close all figures
            for fig in figures:
                plt.close(fig)

        # Save report summary
        with open(self.output_dir / 'report_summary.json', 'w') as f:
            json.dump(report_summary, f, indent=2)

        logger.info(f"Comprehensive report created with {report_summary['total_plots']} plots")
        logger.info(f"All outputs saved to: {self.output_dir}")

        return report_summary


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Model Comparison Plotting Script for TextDate Feature-Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--base_dir', type=str, required=True,
                       help='Base directory containing Saved_models_results and Saved_models_binary')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: base_dir/comparison_plots)')
    parser.add_argument('--metrics', nargs='+', default=None,
                       help='Metrics to analyze (default: all available metrics)')
    parser.add_argument('--time_scales', nargs='+', choices=['decades', 'centuries'],
                       default=['decades', 'centuries'],
                       help='Time scales to analyze (default: both)')
    parser.add_argument('--feature_types', nargs='+',
                       choices=['compression', 'lexical_structure', 'readability', 'distance', 'neologism', 'final_model'],
                       default=['compression', 'lexical_structure', 'readability', 'distance', 'neologism', 'final_model'],
                       help='Feature types to analyze (default: all)')
    parser.add_argument('--plot_types', nargs='+',
                       choices=['heatmap', 'binary_comparison', 'dataset_comparison', 'time_comparison', 'all'],
                       default=['all'],
                       help='Types of plots to create (default: all)')
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 8],
                       help='Figure size as width height (default: 12 8)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved figures (default: 300)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize analyzer
    analyzer = ModelResultsAnalyzer(args.base_dir, args.output_dir)

    # Filter feature types if specified
    if args.feature_types != analyzer.FEATURE_TYPES:
        analyzer.FEATURE_TYPES = args.feature_types

    try:
        if 'all' in args.plot_types:
            # Create comprehensive report
            analyzer.create_comprehensive_report(args.metrics)
        else:
            # Load data
            analyzer.load_all_data()

            # Create specific plots
            for time_scale in args.time_scales:
                for metric in args.metrics:
                    if 'heatmap' in args.plot_types and metric in analyzer.BASE_METRICS:
                        # Default to test dataset, or allow specification
                        fig = analyzer.create_base_model_heatmap(metric, time_scale, 'test', tuple(args.figsize))
                        if fig:
                            plt.close(fig)

                    if 'binary_comparison' in args.plot_types and metric in analyzer.BINARY_METRICS:
                        fig = analyzer.create_binary_model_comparison(metric, time_scale, tuple(args.figsize))
                        if fig:
                            plt.close(fig)

                    if 'dataset_comparison' in args.plot_types and metric in analyzer.BINARY_METRICS:
                        fig = analyzer.create_dataset_comparison_plot(metric, time_scale, tuple(args.figsize))
                        if fig:
                            plt.close(fig)

            for metric in args.metrics:
                if 'time_comparison' in args.plot_types and metric in analyzer.BASE_METRICS:
                    fig = analyzer.create_time_scale_comparison(metric, tuple(args.figsize))
                    if fig:
                        plt.close(fig)

        logger.info("Analysis completed successfully!")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()