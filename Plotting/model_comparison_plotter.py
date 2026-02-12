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
            base_dir: Base directory containing Results/saved_models_results and Saved_models_binary
            output_dir: Directory to save plots (default: base_dir/Results/comparison_plots)
        """
        
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir) if output_dir else self.base_dir / "Results" / "comparison_plots"
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
            'auprc', "top2_accuracy" ,'top3_accuracy', 'top5_accuracy', 'top10_accuracy'
        ]

        self.BINARY_METRICS = [
            'accuracy', 'auc_roc', 'auprc', 'f1_macro', 'f1_weighted',
            'recall_macro', 'recall_weighted', 'precision_macro', 'precision_weighted',
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

        base_path = self.base_dir / "Results" / "saved_models_results"
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

    def create_2x2_ranking_heatmaps(self, dataset: str = 'test', figsize: Tuple[int, int] = (20, 16)) -> plt.Figure:
        """Create 2x2 grid of AUC-ROC and AUPRC heatmaps with centuries on top row, decades on bottom row."""
        logger.info(f"Creating 2x2 ranking heatmaps for dataset: {dataset}")

        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Define custom color palette: purple (0.0) → blue → green → yellow → orange → red (1.0)
        cmap = 'RdYlBu_r'

        # Define metrics and time scales
        metrics = ['auc_roc', 'auprc']
        time_scales = ['centuries', 'decades']

        # Fix acronym formatting to match dissertation
        metric_labels = {'auc_roc': 'AUC-ROC', 'auprc': 'AUPRC'}

        for row, time_scale in enumerate(time_scales):
            for col, metric in enumerate(metrics):
                ax = axes[row, col]

                # Prepare data matrix
                data_matrix = []
                feature_labels = []
                model_labels = self.BASE_MODELS

                for feature_type in self.FEATURE_TYPES:
                    if (feature_type in self.base_results and
                        time_scale in self.base_results[feature_type] and
                        dataset in self.base_results[feature_type][time_scale]):
                        row_data = []
                        for model in self.BASE_MODELS:
                            if model in self.base_results[feature_type][time_scale][dataset]:
                                value = self.base_results[feature_type][time_scale][dataset][model].get(metric, np.nan)
                                row_data.append(value)
                            else:
                                row_data.append(np.nan)
                        data_matrix.append(row_data)
                        feature_labels.append(feature_type.replace('_', ' ').title())
                        

                if data_matrix:
                    # Create heatmap with 2 decimal places
                    data_df = pd.DataFrame(data_matrix, index=feature_labels, columns=model_labels)
                    sns.heatmap(data_df, annot=True, fmt='.2f', cmap=cmap,
                               square=True, cbar_kws={'label': metric_labels[metric]}, ax=ax,
                               vmin=0, vmax=1)  # Set consistent scale for ranking metrics

                    # Customize subplot
                    ax.set_title(f'{metric_labels[metric]} - {time_scale.title()}',
                                fontsize=14, fontweight='bold', pad=10)
                    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Feature Types' if col == 0 else '', fontsize=12, fontweight='bold')

                    # Rotate labels for better readability
                    ax.tick_params(axis='x', rotation=45, labelsize=10)
                    ax.tick_params(axis='y', rotation=0, labelsize=10)

        plt.suptitle(f'Model Performance Comparison - Ranking Metrics\nDataset: {dataset.title()}',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

        # Save plot
        filename = f"ranking_metrics_2x2_{dataset}.png"
        plt.savefig(self.heatmaps_dir / dataset / filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved 2x2 ranking metrics plot: heatmaps/{dataset}/{filename}")

        return fig

    def create_2x2_error_heatmaps(self, dataset: str = 'test', figsize: Tuple[int, int] = (20, 16)) -> plt.Figure:
        """Create 2x2 grid of MAE and RMSE heatmaps with centuries on top row, decades on bottom row."""
        logger.info(f"Creating 2x2 error heatmaps for dataset: {dataset}")

        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        cmap = 'RdYlBu_r'

        # Define metrics and time scales
        metrics = ['mae', 'rmse']
        time_scales = ['centuries', 'decades']

        # Metric labels
        metric_labels = {'mae': 'MAE', 'rmse': 'RMSE'}

        for row, time_scale in enumerate(time_scales):
            for col, metric in enumerate(metrics):
                ax = axes[row, col]

                # Prepare data matrix
                data_matrix = []
                feature_labels = []
                model_labels = self.BASE_MODELS

                for feature_type in self.FEATURE_TYPES:
                    if (feature_type in self.base_results and
                        time_scale in self.base_results[feature_type] and
                        dataset in self.base_results[feature_type][time_scale]):
                        row_data = []
                        for model in self.BASE_MODELS:
                            if model in self.base_results[feature_type][time_scale][dataset]:
                                value = self.base_results[feature_type][time_scale][dataset][model].get(metric, np.nan)
                                row_data.append(value)
                            else:
                                row_data.append(np.nan)
                        data_matrix.append(row_data)
                        feature_labels.append(feature_type.replace('_', ' ').title())

                if data_matrix:
                    print(f"Data matrix for {metric} - {time_scale}:", flush=True)
                    print(data_matrix, flush=True)
                    # Create heatmap with 2 decimal places
                    data_df = pd.DataFrame(data_matrix, index=feature_labels, columns=model_labels)
                    sns.heatmap(data_df, annot=True, fmt='.2f', cmap=cmap,
                               square=True, cbar_kws={'label': metric_labels[metric]}, ax=ax)

                    # Customize subplot
                    ax.set_title(f'{metric_labels[metric]} - {time_scale.title()}',
                                fontsize=14, fontweight='bold', pad=10)
                    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Feature Types' if col == 0 else '', fontsize=12, fontweight='bold')

                    # Rotate labels for better readability
                    ax.tick_params(axis='x', rotation=45, labelsize=10)
                    ax.tick_params(axis='y', rotation=0, labelsize=10)

        plt.suptitle(f'Model Performance Comparison - Error Metrics\nDataset: {dataset.title()}',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

        # Save plot
        filename = f"error_metrics_2x2_{dataset}.png"
        plt.savefig(self.heatmaps_dir / dataset / filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved 2x2 error metrics plot: heatmaps/{dataset}/{filename}")

        return fig

    def create_topk_combined_plot(self, time_scale: str = 'decades', dataset: str = 'test',
                                  figsize: Tuple[int, int] = (20, 6)) -> plt.Figure:
        """Create combined top-k accuracy plot in 3-line format (top 3, 5, 10)."""
        logger.info(f"Creating combined top-k plot for {time_scale} on {dataset}")

        # Create figure with 1x3 subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Define top-k values and custom shared color palette
        topk_values = [3, 5, 10]
        shared_palette = 'RdYlBu_r'

        for idx, k in enumerate(topk_values):
            ax = axes[idx]
            metric = f'top{k}_accuracy'

            # Prepare data matrix
            data_matrix = []
            feature_labels = []
            model_labels = self.BASE_MODELS

            for feature_type in self.FEATURE_TYPES:
                if (feature_type in self.base_results and
                    time_scale in self.base_results[feature_type] and
                    dataset in self.base_results[feature_type][time_scale]):
                    row_data = []
                    for model in self.BASE_MODELS:
                        if model in self.base_results[feature_type][time_scale][dataset]:
                            value = self.base_results[feature_type][time_scale][dataset][model].get(metric, np.nan)
                            row_data.append(value)
                        else:
                            row_data.append(np.nan)
                    data_matrix.append(row_data)
                    feature_labels.append(feature_type.replace('_', ' ').title())

            if data_matrix:
                # Create heatmap
                data_df = pd.DataFrame(data_matrix, index=feature_labels, columns=model_labels)
                sns.heatmap(data_df, annot=True, fmt='.2f', cmap=shared_palette,
                           square=True, cbar_kws={'label': f'Top-{k} Accuracy'}, ax=ax,
                           vmin=0, vmax=1)

                # Customize subplot
                ax.set_title(f'Top-{k} Accuracy', fontsize=14, fontweight='bold', pad=10)
                ax.set_xlabel('Models', fontsize=12, fontweight='bold')
                ax.set_ylabel('Feature Types' if idx == 0 else '', fontsize=12, fontweight='bold')

                # Rotate labels
                ax.tick_params(axis='x', rotation=45, labelsize=10)
                ax.tick_params(axis='y', rotation=0, labelsize=10)

        plt.suptitle(f'Top-K Accuracy Analysis - {time_scale.title()} Classification\nDataset: {dataset.title()}',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        # Save plot
        filename = f"topk_combined_{time_scale}_{dataset}.png"
        plt.savefig(self.heatmaps_dir / dataset / filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved combined top-k plot: heatmaps/{dataset}/{filename}")

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

    def create_binary_threshold_performance_plot_individual(self, feature_type: str = 'final_model',
                                                          time_scale: str = 'decades',
                                                          model: str = 'catboost',
                                                          figsize: Tuple[int, int] = (24, 8)) -> plt.Figure:
        """
        Create clean binary classification performance plot for a single model across datasets.
        Shows how performance varies across temporal thresholds.

        Args:
            feature_type: Feature type to analyze
            time_scale: 'decades' or 'centuries'
            model: Single model to analyze ('catboost', 'xgboost', etc.)
            figsize: Figure size

        Returns:
            Figure with threshold performance plots
        """
        logger.info(f"Creating binary threshold performance plot for {model} {feature_type} ({time_scale})")

        if (feature_type not in self.binary_results or
            time_scale not in self.binary_results[feature_type]):
            logger.warning(f"No binary results found for {feature_type} {time_scale}")
            return None

        # Create wider subplot grid with more horizontal space
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Increase spacing between subplots - maximize plot area with minimal top spacing
        plt.subplots_adjust(hspace=0.3, wspace=0.2, top=0.92, bottom=0.15, left=0.06, right=0.94)

        # Enable subtle grid that appears behind lines
        plt.rcParams['axes.grid'] = True

        # Define metrics to plot with clear descriptions
        metrics = ['accuracy', 'f1_macro']
        colors = {'accuracy': '#1f77b4', 'f1_macro': '#ff7f0e'}

        # Define datasets
        datasets = self.DATASETS
        ax2_first = None  # Store reference to first subplot's secondary axis for legend

        for dataset_idx, dataset in enumerate(datasets):
            ax = axes[dataset_idx]

            if dataset in self.binary_results[feature_type][time_scale]:
                df = self.binary_results[feature_type][time_scale][dataset]
                model_data = df[df['model'].str.contains(model, na=False)]

                if not model_data.empty:
                    # Get all data for this dataset (not just the specific model)
                    full_data = df

                    # Extract thresholds and convert to appropriate scale
                    if 'threshold' in full_data.columns:
                        thresholds = sorted(full_data['threshold'].unique())
                        if time_scale == 'centuries':
                            # Convert year thresholds to century scale
                            thresholds = [t/100 for t in thresholds]

                    # Plot performance metrics using real data for the specific model
                    model_subset = full_data[full_data['model'].str.contains(model, na=False)]
                    if not model_subset.empty:
                        for metric in metrics:
                            if metric in model_subset.columns:
                                # Get threshold values for plotting
                                plot_thresholds = model_subset['threshold'].values
                                if time_scale == 'centuries':
                                    plot_thresholds = plot_thresholds / 100

                                metric_values = model_subset[metric].values

                                # Set line style based on metric
                                if metric == 'accuracy':
                                    linestyle = '-'  # solid line
                                elif metric == 'f1_macro':
                                    linestyle = ':'  # dotted line
                                else:
                                    linestyle = '-'

                                ax.plot(plot_thresholds, metric_values,
                                       label=metric.replace('_', ' ').title(),
                                       color=colors[metric], linewidth=3.5, alpha=0.9,
                                       linestyle=linestyle, zorder=2)

                    # Plot random baseline using real random classifier results
                    random_data = full_data[full_data['model'] == 'random']
                    if not random_data.empty and 'accuracy' in random_data.columns:
                        random_thresholds = random_data['threshold'].values
                        if time_scale == 'centuries':
                            random_thresholds = random_thresholds / 100
                        random_accuracy = random_data['accuracy'].values

                        ax.plot(random_thresholds, random_accuracy, '-', color='red',
                               alpha=0.7, label='Random', linewidth=3, zorder=2)

                    # Add real percentage of texts using actual data
                    if '%_of_texts' in model_subset.columns:
                        plot_thresholds = model_subset['threshold'].values
                        if time_scale == 'centuries':
                            plot_thresholds = plot_thresholds / 100
                        pct_texts = model_subset['%_of_texts'].values

                        ax2 = ax.twinx()
                        ax2.plot(plot_thresholds, pct_texts, '--', color='green', alpha=0.8,
                               label='% of Texts', linewidth=3, zorder=2)
                        ax2.set_ylabel('% of Texts', fontsize=13, fontweight='bold', alpha=0.8)
                        ax2.set_ylim(0, 1)
                        ax2.tick_params(axis='y', labelsize=11)
                        ax2.grid(False)  # Disable grid on secondary axis to avoid overlap

                        # Store reference for legend
                        if dataset_idx == 0:
                            ax2_first = ax2

            # Customize subplot with larger fonts and better spacing
            ax.set_title(f'{dataset.title()} Dataset', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Temporal Threshold', fontsize=14, fontweight='bold')
            if dataset_idx == 0:
                ax.set_ylabel('Performance Score', fontsize=14, fontweight='bold')

            # Clean styling with subtle grid behind lines
            ax.set_ylim(0, 1.05)
            ax.tick_params(axis='both', labelsize=12)
            ax.grid(True, alpha=0.3, color='gray', linewidth=0.5, zorder=0)  # Subtle grid behind lines

            # Add subtle background
            ax.set_facecolor('#fafafa')

            # Add legend to the first subplot at top left, aligned with title
            if dataset_idx == 0:  # First subplot
                # Get legends from main axis
                lines1, labels1 = ax.get_legend_handles_labels()

                # Get legends from secondary axis if it exists
                lines2, labels2 = [], []
                if ax2_first is not None:
                    lines2, labels2 = ax2_first.get_legend_handles_labels()

                # Combine legends from both axes
                all_lines = lines1 + lines2
                all_labels = labels1 + labels2

                legend = ax.legend(all_lines, all_labels, bbox_to_anchor=(0, 1.30), loc='upper left', fontsize=12,
                                 frameon=True, fancybox=True, shadow=True, ncol=3)
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_alpha(0.95)

        # Add title centered at the top
        main_title = f'{model.upper()} Binary Classification Performance\n{feature_type.replace("_", " ").title()} Features - {time_scale.title()} Scale'

        # Position title using fig.text at the very top edge
        fig.text(0.5, 1.06, main_title, ha='center', va='bottom', fontsize=20, fontweight='bold')

        # Don't use tight_layout since we're manually adjusting

        # Save plot
        filename = f"binary_threshold_performance_{model}_{feature_type}_{time_scale}.png"
        save_path = self.binary_dir / time_scale / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved binary threshold plot: binary_comparisons/{time_scale}/{filename}")

        # Restore default grid setting
        plt.rcParams['axes.grid'] = True

        return fig

    def create_enhanced_binary_threshold_plots(self, feature_type: str = 'final_model',
                                             time_scale: str = 'decades', model: str = 'catboost',
                                             figsize: Tuple[int, int] = (24, 8)) -> plt.Figure:
        """
        Create enhanced binary threshold plots with bigger labels for readability.
        """
        logger.info(f"Creating enhanced binary threshold plot for {model} {feature_type} ({time_scale})")

        if (feature_type not in self.binary_results or
            time_scale not in self.binary_results[feature_type]):
            logger.warning(f"No binary results found for {feature_type} {time_scale}")
            return None

        # Create wider subplot grid with larger fonts
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        plt.subplots_adjust(hspace=0.3, wspace=0.25, top=0.90, bottom=0.15, left=0.08, right=0.94)

        # Define metrics with bigger labels
        metrics = ['accuracy', 'f1_macro']
        colors = {'accuracy': '#1f77b4', 'f1_macro': '#ff7f0e'}
        datasets = self.DATASETS

        for dataset_idx, dataset in enumerate(datasets):
            ax = axes[dataset_idx]

            if dataset in self.binary_results[feature_type][time_scale]:
                df = self.binary_results[feature_type][time_scale][dataset]
                model_data = df[df['model'].str.contains(model, na=False)]

                if not model_data.empty:
                    # Plot with enhanced styling
                    for metric in metrics:
                        if metric in model_data.columns:
                            plot_thresholds = model_data['threshold'].values
                            if time_scale == 'centuries':
                                plot_thresholds = plot_thresholds / 100

                            metric_values = model_data[metric].values
                            linestyle = '-' if metric == 'accuracy' else ':'

                            ax.plot(plot_thresholds, metric_values,
                                   label=metric.replace('_', ' ').title(),
                                   color=colors[metric], linewidth=4, alpha=0.9,
                                   linestyle=linestyle, zorder=2)

                    # Add random baseline
                    random_data = df[df['model'] == 'random']
                    if not random_data.empty and 'accuracy' in random_data.columns:
                        random_thresholds = random_data['threshold'].values
                        if time_scale == 'centuries':
                            random_thresholds = random_thresholds / 100
                        random_accuracy = random_data['accuracy'].values

                        ax.plot(random_thresholds, random_accuracy, '-', color='red',
                               alpha=0.7, label='Random', linewidth=4, zorder=2)

                    # Add percentage of texts with bigger styling
                    if '%_of_texts' in model_data.columns:
                        plot_thresholds = model_data['threshold'].values
                        if time_scale == 'centuries':
                            plot_thresholds = plot_thresholds / 100
                        pct_texts = model_data['%_of_texts'].values

                        ax2 = ax.twinx()
                        ax2.plot(plot_thresholds, pct_texts, '--', color='green', alpha=0.8,
                               label='% of Texts', linewidth=4, zorder=2)
                        ax2.set_ylabel('% of Texts', fontsize=16, fontweight='bold')  # Bigger font
                        ax2.set_ylim(0, 1)
                        ax2.tick_params(axis='y', labelsize=14)  # Bigger ticks

            # Enhanced styling with bigger fonts
            ax.set_title(f'{dataset.title()} Dataset', fontsize=20, fontweight='bold', pad=25)  # Bigger title
            ax.set_xlabel('Temporal Threshold', fontsize=18, fontweight='bold')  # Bigger labels
            if dataset_idx == 0:
                ax.set_ylabel('Performance Score', fontsize=18, fontweight='bold')

            ax.set_ylim(0, 1.05)
            ax.tick_params(axis='both', labelsize=14)  # Bigger tick labels
            ax.grid(True, alpha=0.3, color='gray', linewidth=0.5, zorder=0)

            # Bigger legend
            if dataset_idx == 0:
                lines1, labels1 = ax.get_legend_handles_labels()
                if 'ax2' in locals():
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    all_lines = lines1 + lines2
                    all_labels = labels1 + labels2
                else:
                    all_lines = lines1
                    all_labels = labels1

                legend = ax.legend(all_lines, all_labels, bbox_to_anchor=(0, 1.30),
                                 loc='upper left', fontsize=14,  # Bigger legend font
                                 frameon=True, fancybox=True, shadow=True, ncol=3)

        # Enhanced main title
        main_title = f'{model.upper()} Binary Classification Performance\n{feature_type.replace("_", " ").title()} Features - {time_scale.title()} Scale'
        fig.text(0.5, 1.08, main_title, ha='center', va='bottom',
                fontsize=24, fontweight='bold')  # Much bigger title

        # Save with enhanced naming
        filename = f"enhanced_binary_threshold_{model}_{feature_type}_{time_scale}.png"
        save_path = self.binary_dir / time_scale / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved enhanced binary plot: binary_comparisons/{time_scale}/{filename}")

        return fig

    def create_temporal_evolution_analysis(self, feature_type: str = 'final_model',
                                         time_scale: str = 'decades',
                                         figsize: Tuple[int, int] = (20, 12)) -> plt.Figure:
        """
        Temporal evolution analysis showing performance across historical periods.

        Args:
            feature_type: Feature type to analyze
            time_scale: 'decades' or 'centuries'
            figsize: Figure size

        Returns:
            Figure with temporal evolution analysis
        """
        logger.info(f"Creating temporal evolution analysis for {feature_type} ({time_scale})")

        if (feature_type not in self.binary_results or
            time_scale not in self.binary_results[feature_type]):
            logger.warning(f"No binary results found for {feature_type} {time_scale}")
            return None

        # Create subplot grid: 3 rows, 2 columns
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle(f'Temporal Evolution Analysis: {feature_type.replace("_", " ").title()} Features ({time_scale.title()})',
                    fontsize=20, fontweight='bold', y=0.98)

        # Adjust spacing
        plt.subplots_adjust(hspace=0.35, wspace=0.25, top=0.93, bottom=0.08, left=0.08, right=0.95)

        # Define metrics to analyze
        metrics = ['accuracy', 'f1_macro', 'auc_roc', 'auprc']
        metric_titles = ['Accuracy', 'F1-Macro', 'AUC-ROC', 'AUPRC']
        datasets = self.DATASETS
        colors = {'validation': '#1f77b4', 'test': '#ff7f0e', 'gutenberg': '#2ca02c'}

        # Plot each metric (4 metrics, so use 4 of the 6 subplots)
        for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]

            for dataset in datasets:
                if dataset in self.binary_results[feature_type][time_scale]:
                    df = self.binary_results[feature_type][time_scale][dataset]

                    # Use CatBoost as primary model for temporal analysis
                    model_data = df[df['model'].str.contains('catboost', na=False)]

                    if not model_data.empty and metric in model_data.columns:
                        thresholds = model_data['threshold'].values
                        if time_scale == 'centuries':
                            thresholds = thresholds / 100  # Convert to century scale

                        metric_values = model_data[metric].values

                        ax.plot(thresholds, metric_values, '-',
                               color=colors[dataset], linewidth=2.5, alpha=0.8,
                               label=f'{dataset.title()}', zorder=2)

            # Customize subplot
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Temporal Threshold', fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.grid(True, alpha=0.3, color='gray', linewidth=0.5, zorder=0)
            ax.legend()
            ax.set_ylim(0, 1.05)

        # Use remaining 2 subplots for special analysis
        # Row 2, Col 0: Class Distribution Evolution
        ax_dist = axes[2, 0]
        for dataset in datasets:
            if dataset in self.binary_results[feature_type][time_scale]:
                df = self.binary_results[feature_type][time_scale][dataset]
                model_data = df[df['model'].str.contains('catboost', na=False)]

                if not model_data.empty and '%_of_texts' in model_data.columns:
                    thresholds = model_data['threshold'].values
                    if time_scale == 'centuries':
                        thresholds = thresholds / 100

                    pct_texts = model_data['%_of_texts'].values

                    ax_dist.plot(thresholds, pct_texts, '--',
                               color=colors[dataset], linewidth=2.5, alpha=0.8,
                               label=f'{dataset.title()}', zorder=2)

        ax_dist.set_title('Class Distribution Evolution\n(% Texts Labeled as "Old")', fontsize=14, fontweight='bold')
        ax_dist.set_xlabel('Temporal Threshold', fontsize=12)
        ax_dist.set_ylabel('Percentage of Texts', fontsize=12)
        ax_dist.grid(True, alpha=0.3, color='gray', linewidth=0.5, zorder=0)
        ax_dist.legend()
        ax_dist.set_ylim(0, 1.05)

        # Row 2, Col 1: Performance Stability Analysis (coefficient of variation)
        ax_stability = axes[2, 1]
        stability_metrics = ['accuracy', 'f1_macro', 'auc_roc', 'auprc']
        dataset_stability = {}

        for dataset in datasets:
            if dataset in self.binary_results[feature_type][time_scale]:
                df = self.binary_results[feature_type][time_scale][dataset]
                model_data = df[df['model'].str.contains('catboost', na=False)]

                if not model_data.empty:
                    cv_values = []
                    for metric in stability_metrics:
                        if metric in model_data.columns:
                            values = model_data[metric].values
                            cv = np.std(values) / np.mean(values)  # Coefficient of variation
                            cv_values.append(cv)

                    dataset_stability[dataset] = np.mean(cv_values)

        # Bar plot of stability
        if dataset_stability:
            datasets_stable = list(dataset_stability.keys())
            cv_values = list(dataset_stability.values())
            bars = ax_stability.bar(datasets_stable, cv_values,
                                  color=[colors[d] for d in datasets_stable], alpha=0.7)

            ax_stability.set_title('Performance Stability Across Thresholds\n(Lower = More Stable)',
                                 fontsize=14, fontweight='bold')
            ax_stability.set_ylabel('Coefficient of Variation', fontsize=12)
            ax_stability.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, val in zip(bars, cv_values):
                ax_stability.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        return fig

    def create_cross_dataset_performance_comparison(self, feature_type: str = 'final_model',
                                                  time_scale: str = 'decades',
                                                  figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create cross-dataset performance comparison highlighting Gutenberg differences
        and explaining domain adaptation challenges.

        Args:
            feature_type: Feature type to analyze
            time_scale: 'decades' or 'centuries'
            figsize: Figure size

        Returns:
            Figure with cross-dataset analysis
        """
        logger.info(f"Creating cross-dataset comparison for {feature_type} ({time_scale})")

        if (feature_type not in self.binary_results or
            time_scale not in self.binary_results[feature_type]):
            logger.warning(f"No binary results found for {feature_type} {time_scale}")
            return None

        # Create subplot grid: 2 rows, 3 columns (but only use 5 subplots)
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'Cross-Dataset Performance Analysis: {feature_type.replace("_", " ").title()} ({time_scale.title()})',
                    fontsize=18, fontweight='bold', y=0.95)

        # Remove the last subplot (bottom right) that was used for optimal thresholds
        fig.delaxes(axes[1, 2])

        # Adjust spacing
        plt.subplots_adjust(hspace=0.35, wspace=0.25, top=0.90, bottom=0.08, left=0.08, right=0.95)

        datasets = self.DATASETS
        colors = {'validation': '#1f77b4', 'test': '#ff7f0e', 'gutenberg': '#d62728'}
        metrics = ['accuracy', 'f1_macro', 'auc_roc']

        # First row: Performance curves for key metrics
        for idx, metric in enumerate(metrics):
            ax = axes[0, idx]

            for dataset in datasets:
                if dataset in self.binary_results[feature_type][time_scale]:
                    df = self.binary_results[feature_type][time_scale][dataset]
                    model_data = df[df['model'].str.contains('catboost', na=False)]

                    if not model_data.empty and metric in model_data.columns:
                        thresholds = model_data['threshold'].values
                        if time_scale == 'centuries':
                            thresholds = thresholds / 100

                        metric_values = model_data[metric].values

                        # Highlight Gutenberg with different line style
                        linestyle = '--' if dataset == 'gutenberg' else '-'
                        linewidth = 3 if dataset == 'gutenberg' else 2.5

                        ax.plot(thresholds, metric_values, linestyle,
                               color=colors[dataset], linewidth=linewidth, alpha=0.8,
                               label=f'{dataset.title()}', zorder=2)

            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Temporal Threshold', fontsize=12)
            ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
            ax.grid(True, alpha=0.3, color='gray', linewidth=0.5, zorder=0)
            ax.legend()
            ax.set_ylim(0, 1.05)

            # Set custom x-axis labels for centuries
            if time_scale == 'centuries':
                ax.set_xticks([0.18, 0.19, 0.20])
                ax.set_xticklabels(['18', '19', '20'])

        # Second row: Specialized analysis
        # Performance gap analysis
        ax_gap = axes[1, 0]
        for metric in ['accuracy', 'f1_macro']:
            test_data = self.binary_results[feature_type][time_scale].get('test')
            gutenberg_data = self.binary_results[feature_type][time_scale].get('gutenberg')

            if test_data is not None and gutenberg_data is not None:
                test_model = test_data[test_data['model'].str.contains('catboost', na=False)]
                gutenberg_model = gutenberg_data[gutenberg_data['model'].str.contains('catboost', na=False)]

                if not test_model.empty and not gutenberg_model.empty and metric in test_model.columns:
                    test_thresholds = test_model['threshold'].values
                    if time_scale == 'centuries':
                        test_thresholds = test_thresholds / 100

                    test_values = test_model[metric].values
                    gutenberg_values = gutenberg_model[metric].values

                    # Calculate performance gap
                    gap = test_values - gutenberg_values

                    ax_gap.plot(test_thresholds, gap, '-',
                               linewidth=2.5, alpha=0.8,
                               label=f'{metric.replace("_", " ").title()} Gap', zorder=2)

        ax_gap.set_title('Performance Gap\n(Test - Gutenberg)', fontsize=14, fontweight='bold')
        ax_gap.set_xlabel('Temporal Threshold', fontsize=12)
        ax_gap.set_ylabel('Performance Gap', fontsize=12)
        ax_gap.grid(True, alpha=0.3, color='gray', linewidth=0.5, zorder=0)
        ax_gap.legend()
        ax_gap.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Set custom x-axis labels for centuries
        if time_scale == 'centuries':
            ax_gap.set_xticks([0.18, 0.19, 0.20])
            ax_gap.set_xticklabels(['18', '19', '20'])

        # Class distribution comparison
        ax_class = axes[1, 1]
        for dataset in datasets:
            if dataset in self.binary_results[feature_type][time_scale]:
                df = self.binary_results[feature_type][time_scale][dataset]
                model_data = df[df['model'].str.contains('catboost', na=False)]

                if not model_data.empty and '%_of_texts' in model_data.columns:
                    thresholds = model_data['threshold'].values
                    if time_scale == 'centuries':
                        thresholds = thresholds / 100

                    pct_texts = model_data['%_of_texts'].values

                    linestyle = '--' if dataset == 'gutenberg' else '-'
                    linewidth = 3 if dataset == 'gutenberg' else 2.5

                    ax_class.plot(thresholds, pct_texts, linestyle,
                                color=colors[dataset], linewidth=linewidth, alpha=0.8,
                                label=f'{dataset.title()}', zorder=2)

        ax_class.set_title('Class Distribution Comparison\n(% Texts Labeled "Old")', fontsize=14, fontweight='bold')
        ax_class.set_xlabel('Temporal Threshold', fontsize=12)
        ax_class.set_ylabel('Percentage', fontsize=12)
        ax_class.grid(True, alpha=0.3, color='gray', linewidth=0.5, zorder=0)
        ax_class.legend()
        ax_class.set_ylim(0, 1.05)

        # Set custom x-axis labels for centuries
        if time_scale == 'centuries':
            ax_class.set_xticks([0.18, 0.19, 0.20])
            ax_class.set_xticklabels(['18', '19', '20'])

        return fig

    def create_feature_domain_temporal_sensitivity(self, time_scale: str = 'decades',
                                                 figsize: Tuple[int, int] = (20, 14)) -> plt.Figure:
        """
        Create feature domain temporal sensitivity analysis showing how different
        feature types respond to temporal boundary changes.

        Args:
            time_scale: 'decades' or 'centuries'
            figsize: Figure size

        Returns:
            Figure with feature domain sensitivity analysis
        """
        logger.info(f"Creating feature domain temporal sensitivity analysis ({time_scale})")

        # Get available feature types (excluding optimal which might not have full data)
        feature_types = [ft for ft in self.binary_results.keys() if ft != 'optimal']

        if not feature_types:
            logger.warning(f"No feature types found for {time_scale}")
            return None

        # Create subplot grid: 3 rows, 3 columns
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        fig.suptitle(f'Feature Domain Temporal Sensitivity Analysis ({time_scale.title()})',
                    fontsize=20, fontweight='bold', y=0.98)

        # Adjust spacing
        plt.subplots_adjust(hspace=0.4, wspace=0.25, top=0.93, bottom=0.08, left=0.08, right=0.95)

        # Define colors for feature types
        feature_colors = {
            'compression': '#1f77b4',
            'lexical_structure': '#ff7f0e',
            'readability': '#2ca02c',
            'distance': '#d62728',
            'neologism': '#9467bd',
            'final_model': '#8c564b'
        }

        # First 6 subplots: Individual feature domain analysis
        for idx, feature_type in enumerate(feature_types[:6]):
            if idx >= 6:  # Only use first 6 subplots
                break

            row = idx // 3
            col = idx % 3
            ax = axes[row, col]

            if (feature_type in self.binary_results and
                time_scale in self.binary_results[feature_type]):

                # Use test dataset for primary analysis
                if 'test' in self.binary_results[feature_type][time_scale]:
                    df = self.binary_results[feature_type][time_scale]['test']
                    model_data = df[df['model'].str.contains('catboost', na=False)]

                    if not model_data.empty:
                        thresholds = model_data['threshold'].values
                        if time_scale == 'centuries':
                            thresholds = thresholds / 100

                        # Plot multiple metrics for this feature
                        for metric, alpha in [('accuracy', 1.0), ('auc_roc', 0.8), ('auprc', 0.6)]:
                            if metric in model_data.columns:
                                metric_values = model_data[metric].values
                                ax.plot(thresholds, metric_values, '-',
                                       color=feature_colors.get(feature_type, '#000000'),
                                       linewidth=2.5, alpha=alpha,
                                       label=metric.replace('_', ' ').title(), zorder=2)

            ax.set_title(f'{feature_type.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Temporal Threshold', fontsize=10)
            ax.set_ylabel('Performance', fontsize=10)
            ax.grid(True, alpha=0.3, color='gray', linewidth=0.5, zorder=0)
            ax.legend(fontsize=9)
            ax.set_ylim(0, 1.05)

        # Remaining subplots: Comparative analysis
        # Feature stability comparison (row 2, col 0)
        ax_stability = axes[2, 0]
        stability_scores = {}

        for feature_type in feature_types:
            if (feature_type in self.binary_results and
                time_scale in self.binary_results[feature_type] and
                'test' in self.binary_results[feature_type][time_scale]):

                df = self.binary_results[feature_type][time_scale]['test']
                model_data = df[df['model'].str.contains('catboost', na=False)]

                if not model_data.empty and 'accuracy' in model_data.columns:
                    accuracy_values = model_data['accuracy'].values
                    # Calculate coefficient of variation as stability measure
                    cv = np.std(accuracy_values) / np.mean(accuracy_values)
                    stability_scores[feature_type] = cv

        if stability_scores:
            features = list(stability_scores.keys())
            cv_values = list(stability_scores.values())
            bars = ax_stability.bar(features, cv_values,
                                  color=[feature_colors.get(f, '#000000') for f in features], alpha=0.7)

            ax_stability.set_title('Temporal Stability\n(Lower = More Stable)', fontsize=12, fontweight='bold')
            ax_stability.set_ylabel('Coefficient of Variation', fontsize=10)
            ax_stability.tick_params(axis='x', rotation=45, labelsize=9)

            for bar, val in zip(bars, cv_values):
                ax_stability.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

        # Optimal threshold comparison (row 2, col 1)
        ax_optimal = axes[2, 1]
        optimal_thresholds = {}

        for feature_type in feature_types:
            if (feature_type in self.binary_results and
                time_scale in self.binary_results[feature_type] and
                'test' in self.binary_results[feature_type][time_scale]):

                df = self.binary_results[feature_type][time_scale]['test']
                model_data = df[df['model'].str.contains('catboost', na=False)]

                if not model_data.empty and 'accuracy' in model_data.columns:
                    thresholds = model_data['threshold'].values
                    if time_scale == 'centuries':
                        thresholds = thresholds / 100

                    accuracy_values = model_data['accuracy'].values
                    optimal_idx = np.argmax(accuracy_values)
                    optimal_threshold = thresholds[optimal_idx]
                    optimal_thresholds[feature_type] = optimal_threshold

        if optimal_thresholds:
            features = list(optimal_thresholds.keys())
            thresholds_opt = list(optimal_thresholds.values())
            bars = ax_optimal.bar(features, thresholds_opt,
                                color=[feature_colors.get(f, '#000000') for f in features], alpha=0.7)

            ax_optimal.set_title('Optimal Temporal Thresholds\nby Feature Domain', fontsize=12, fontweight='bold')
            ax_optimal.set_ylabel('Optimal Threshold', fontsize=10)
            ax_optimal.tick_params(axis='x', rotation=45, labelsize=9)

            for bar, threshold in zip(bars, thresholds_opt):
                ax_optimal.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                              f'{threshold:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

        # Peak performance comparison (row 2, col 2)
        ax_peak = axes[2, 2]
        peak_performance = {}

        for feature_type in feature_types:
            if (feature_type in self.binary_results and
                time_scale in self.binary_results[feature_type] and
                'test' in self.binary_results[feature_type][time_scale]):

                df = self.binary_results[feature_type][time_scale]['test']
                model_data = df[df['model'].str.contains('catboost', na=False)]

                if not model_data.empty and 'accuracy' in model_data.columns:
                    accuracy_values = model_data['accuracy'].values
                    peak_performance[feature_type] = np.max(accuracy_values)

        if peak_performance:
            features = list(peak_performance.keys())
            peak_values = list(peak_performance.values())
            bars = ax_peak.bar(features, peak_values,
                             color=[feature_colors.get(f, '#000000') for f in features], alpha=0.7)

            ax_peak.set_title('Peak Performance\nby Feature Domain', fontsize=12, fontweight='bold')
            ax_peak.set_ylabel('Peak Accuracy', fontsize=10)
            ax_peak.tick_params(axis='x', rotation=45, labelsize=9)
            ax_peak.set_ylim(0.5, 1.05)

            for bar, peak in zip(bars, peak_values):
                ax_peak.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{peak:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

        return fig

    def create_3x2_feature_sensitivity_plot(self, time_scale: str = 'decades',
                                           figsize: tuple = (18, 12)) -> plt.Figure:
        """
        Create the exact 3x2 feature sensitivity plot that matches the image with fixed legend.
        """
        logger.info(f"Creating 3x2 feature domain sensitivity plot for {time_scale}")

        # Feature types in the desired order for the 3x2 grid
        feature_types = ['compression', 'lexical_structure', 'readability',
                        'distance', 'neologism', 'final_model']

        # Colors and styles for metrics (matching the separate script)
        metric_colors = {
            'accuracy': '#1f77b4',    # Blue - solid line
            'auc_roc': '#ff7f0e',     # Orange - dashed line
            'auprc': '#2ca02c'        # Green - dotted line
        }

        metric_styles = {
            'accuracy': '-',    # solid
            'auc_roc': '--',    # dashed
            'auprc': ':'        # dotted
        }

        # Create 3x2 subplot grid
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'Feature Domain Temporal Sensitivity Analysis ({time_scale.title()})',
                    fontsize=20, fontweight='bold', y=0.96)

        # Adjust spacing for better layout
        plt.subplots_adjust(hspace=0.35, wspace=0.25, top=0.85, bottom=0.10, left=0.08, right=0.95)

        # Define subplot positions for 3x2 grid
        subplot_positions = [
            (0, 0), (0, 1), (0, 2),  # Top row
            (1, 0), (1, 1), (1, 2)   # Bottom row
        ]

        # Plot each feature type
        for idx, feature_type in enumerate(feature_types):
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
                                       color=metric_colors[metric],
                                       linestyle=metric_styles[metric],
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

            # Set proper x-axis formatting for decades/centuries
            if time_scale == 'centuries':
                # For centuries, show century labels from 18 to 20
                ax.set_xlim(0.18, 0.20)
                ax.set_xticks([0.18, 0.19, 0.20])
                ax.set_xticklabels(['18', '19', '20'])
            # For decades, let matplotlib handle the range automatically

        # Add shared legend positioned to the left of the title (FIXED POSITION)
        # Create legend handles for the metrics
        legend_handles = []
        legend_labels = []

        # Create legend entries for the metrics shown
        for metric in ['accuracy', 'auc_roc', 'auprc']:
            line = plt.Line2D([0], [0],
                            color=metric_colors[metric],
                            linestyle=metric_styles[metric],
                            linewidth=3.0, alpha=0.85)
            legend_handles.append(line)
            legend_labels.append(metric.replace('_', ' ').upper())

        # Add random baseline
        random_line = plt.Line2D([0], [0], color='red', linestyle='-',
                               alpha=0.5, linewidth=2)
        legend_handles.append(random_line)
        legend_labels.append('Random')

        # Position legend to the left of the title
        fig.legend(legend_handles, legend_labels, bbox_to_anchor=(0.01, 0.96),
                  loc='upper left', fontsize=12, frameon=True, fancybox=True,
                  shadow=True, facecolor='white', framealpha=0.9, ncol=2)

        return fig

    def _simulate_threshold_curve(self, thresholds: np.ndarray, base_performance: float,
                                time_scale: str) -> np.ndarray:
        """
        Simulate realistic threshold performance curve.
        In real implementation, this would use actual threshold sweep data.
        """
        n_points = len(thresholds)

        if time_scale == 'decades':
            # For decades, performance varies more with threshold
            # Create a curve that shows optimal performance around 1850-1900
            center = len(thresholds) // 2
            distance_from_center = np.abs(np.arange(n_points) - center)

            # Bell-like curve with some noise
            curve = base_performance * (1 - 0.3 * (distance_from_center / center) ** 2)
            noise = np.random.normal(0, 0.02, n_points)
            curve += noise

        else:
            # For centuries, more stable performance
            curve = np.full(n_points, base_performance)
            # Add slight variation
            noise = np.random.normal(0, 0.01, n_points)
            curve += noise

        # Ensure values are in valid range
        curve = np.clip(curve, 0.3, 1.0)

        return curve

    def create_comprehensive_report(self, metrics: List[str] = None):
        """Create report with all visualizations."""
        if metrics is None:
            # Use all available metrics from both base and binary models
            metrics = list(set(self.BASE_METRICS + self.BINARY_METRICS))

        logger.info("Creating report...")

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

            # Create individual binary threshold performance plots for each model
            for model in ['catboost', 'xgboost']:
                fig = self.create_binary_threshold_performance_plot_individual(
                    feature_type='final_model',
                    time_scale=time_scale,
                    model=model
                )
                if fig is not None:
                    report_summary['plots_created'].append(f'binary_comparisons/{time_scale}/binary_threshold_performance_{model}_final_model_{time_scale}.png')
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

        logger.info(f"Report created with {report_summary['total_plots']} plots")
        logger.info(f"All outputs saved to: {self.output_dir}")

        return report_summary

    def create_enhanced_chapter6_plots(self):
        """Create enhanced plots for Chapter 6 with improved formatting and layout."""
        logger.info("Creating enhanced Chapter 6 plots...")

        self.load_all_data()

        # 1. Create 2x2 ranking metrics (AUC-ROC and AUPRC)
        for dataset in ['test', 'gutenberg']:
            fig = self.create_2x2_ranking_heatmaps(dataset)
            if fig:
                plt.close(fig)
                logger.info(f"Created 2x2 ranking heatmaps for {dataset}")

        # 2. Create 2x2 error metrics (MAE and RMSE)
        for dataset in ['test', 'gutenberg']:
            fig = self.create_2x2_error_heatmaps(dataset)
            if fig:
                plt.close(fig)
                logger.info(f"Created 2x2 error heatmaps for {dataset}")

        # 3. Create combined top-k plots
        for time_scale in ['decades', 'centuries']:
            for dataset in ['test', 'gutenberg']:
                fig = self.create_topk_combined_plot(time_scale, dataset)
                if fig:
                    plt.close(fig)
                    logger.info(f"Created combined top-k plot for {time_scale} {dataset}")

        # 4. Create enhanced binary classification plots with bigger labels
        for time_scale in ['decades', 'centuries']:
            for model in ['catboost', 'xgboost']:
                fig = self.create_enhanced_binary_threshold_plots('final_model', time_scale, model)
                if fig:
                    plt.close(fig)
                    logger.info(f"Created enhanced binary plot for {model} {time_scale}")

        # 5. Create the 3x2 feature sensitivity plots that match the image you showed
        for time_scale in ['decades', 'centuries']:
            fig = self.create_3x2_feature_sensitivity_plot(time_scale)
            if fig:
                # Save to Thesis figs directory to replace existing plots
                filename = f"feature_domain_sensitivity_{time_scale}_fixed_3x2.png"
                fig_path = Path(f"/data/home/paulojnpinto02/TextDate/Feature-Benchmark/Thesis/figs/{filename}")
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Updated feature sensitivity plot with fixed legend: {fig_path}")

        logger.info("All enhanced Chapter 6 plots created successfully!")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Model comparison plotting for TextDate Feature-Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--base_dir', type=str, required=True,
                       help='Base directory containing Results/saved_models_results and Saved_models_binary')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: base_dir/Results/comparison_plots)')
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
                       choices=['heatmap', 'binary_comparison', 'dataset_comparison', 'time_comparison', 'chapter6_enhanced', 'all'],
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
        if 'chapter6_enhanced' in args.plot_types:
            # Create enhanced Chapter 6 plots
            analyzer.create_enhanced_chapter6_plots()
        elif 'all' in args.plot_types:
            # Create all plots
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