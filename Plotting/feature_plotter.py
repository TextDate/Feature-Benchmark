import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import numpy as np
from tqdm import tqdm

def get_memory_usage():
    """Get current memory usage in MB"""
    return psutil.Process().memory_info().rss / 1024 / 1024

def should_process_chunked(dfs, threshold_mb=1000):
    """Check if datasets are large enough to warrant chunked processing"""
    total_memory = sum(df.memory_usage(deep=True).sum() for df in dfs.values())
    return total_memory / 1024 / 1024 > threshold_mb

def validate_dataframes(dfs, target_col):
    """Validate input dataframes and target column"""
    errors = []

    for name, df in dfs.items():
        if df.empty:
            errors.append(f"Dataset '{name}' is empty")
        if target_col not in df.columns:
            errors.append(f"Target column '{target_col}' not found in '{name}'")
        if target_col in df.columns and df[target_col].isna().all():
            errors.append(f"Target column '{target_col}' is all NaN in '{name}'")

        # Check for numeric columns
        numeric_cols = df.select_dtypes(include='number').columns
        if len(numeric_cols) == 0:
            errors.append(f"Dataset '{name}' has no numeric columns to plot")

    if errors:
        raise ValueError("Validation failed:\n" + "\n".join(errors))

def plot_feature_means_and_boxes(dfs, target_col, output_dir, exclude_values=None):
    os.makedirs(output_dir, exist_ok=True)

    # Set up modern plot aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = plt.cm.Set2(np.linspace(0, 1, 8))

    # Validate input data
    validate_dataframes(dfs, target_col)

    # Create proper DataFrame copies to avoid SettingWithCopyWarning
    dfs = {name: df.copy() for name, df in dfs.items()}

    # Memory monitoring
    initial_memory = get_memory_usage()
    print(f"[INFO] Initial memory usage: {initial_memory:.1f} MB", flush=True)

    if should_process_chunked(dfs):
        print(f"[INFO] Large dataset detected, using memory-efficient processing", flush=True)

    if exclude_values:
        for name, df in dfs.items():
            original_len = len(df)
            dfs[name] = df[~df[target_col].isin(exclude_values)]
            removed = original_len - len(dfs[name])
            print(f"[INFO] [{name}] Excluded {target_col}s: {exclude_values} ({removed} rows removed)", flush=True)

    min_max = {}
    for name, df in dfs.items():
        for col in df.select_dtypes(include='number').columns:
            if col == target_col:
                continue
            q_low = df[col].quantile(0.01)
            q_high = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=q_low, upper=q_high)
            current_min = df[col].min()
            current_max = df.groupby(target_col)[col].mean().max()
            range_buffer = (current_max - current_min) * 0.02
            existing_min, existing_max = min_max.get(col, (float("inf"), float("-inf")))
            min_max[col] = (
                min(existing_min, current_min - range_buffer),
                max(existing_max, current_max + range_buffer)
            )

    for col in tqdm(min_max, desc="Processing features"):
        for name, df in dfs.items():
            os.makedirs(os.path.join(output_dir, name), exist_ok=True)
            grouped = df.groupby(target_col).mean(numeric_only=True).sort_index()

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('white')
            ax.plot(grouped.index, grouped[col], marker='o', linewidth=2, markersize=6, color=colors[0])
            ax.set_title(f"{name}: Mean of '{col}' per {target_col}", fontsize=14, fontweight='bold')
            ax.set_xlabel(target_col.capitalize(), fontsize=12)
            ax.set_ylabel(f"Mean {col}", fontsize=12)
            ymin, ymax = min_max[col]
            ax.set_ylim(ymin, ymax)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            line_path = os.path.join(output_dir, name, f"{col}_mean_by_{target_col}.png")
            fig.savefig(line_path)
            plt.close(fig)
            print(f"[INFO] Saved line plot to {line_path}", flush=True)

            fig, ax = plt.subplots(figsize=(20, 6))
            fig.patch.set_facecolor('white')
            df.boxplot(column=col, by=target_col, grid=True, ax=ax)
            ax.set_title(f"{name}: Box Plot of '{col}' by {target_col.capitalize()}", fontsize=14, fontweight='bold')
            fig.suptitle("")
            ax.set_xlabel(target_col.capitalize(), fontsize=12)
            ax.set_ylabel(col, fontsize=12)
            ymin, ymax = min_max[col]
            ax.set_ylim(ymin, ymax)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            box_path = os.path.join(output_dir,name ,f"{col}_box_by_{target_col}.png")
            fig.savefig(box_path)
            plt.close(fig)
            print(f"[INFO] Saved box plot to {box_path}", flush=True)

            fig, ax = plt.subplots(figsize=(20, 6))
            fig.patch.set_facecolor('white')
            data_to_plot = []
            x_labels = []
            for label, group in df.groupby(target_col):
                data_to_plot.append(group[col])
                x_labels.append(label)
            parts = ax.violinplot(data_to_plot, showmeans=True)
            # Color the violin plots
            for pc in parts['bodies']:
                pc.set_facecolor(colors[1])
                pc.set_alpha(0.7)
            ax.set_xticks(range(1, len(x_labels) + 1))
            ax.set_xticklabels(x_labels, rotation=45)
            ax.set_title(f"{name}: Violin Plot of '{col}' by {target_col}", fontsize=14, fontweight='bold')
            ax.set_xlabel(target_col.capitalize(), fontsize=12)
            ax.set_ylabel(f"{col}", fontsize=12)
            ymin, ymax = min_max[col]
            ax.set_ylim(ymin, ymax)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            violin_path = os.path.join(output_dir,name ,f"{col}_violin_by_{target_col}.png")
            fig.savefig(violin_path)
            plt.close(fig)
            print(f"[INFO] Saved violin plot to {violin_path}", flush=True)

    # Final memory usage report
    final_memory = get_memory_usage()
    print(f"[INFO] Final memory usage: {final_memory:.1f} MB (Δ{final_memory-initial_memory:+.1f} MB)", flush=True)


def main(args):
    datasets = dict()
    for label, path in zip(args.labels, args.csvs):
        df = pd.read_csv(path)
        drop_cols = [c.strip() for group in args.drop_cols for c in group.split(",")]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        datasets[label] = df

    plot_feature_means_and_boxes(datasets, args.target, args.output_dir, args.exclude_decades)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot feature means by decade")
    parser.add_argument("--csvs", nargs="+", required=True, help="Paths to CSV files (e.g. train.csv val.csv test.csv)")
    parser.add_argument("--labels", nargs="+", required=True, help="Labels for the datasets (e.g. train val test)")
    parser.add_argument("--target", default="decade", help="Target column to group by (e.g., 'decade')")
    parser.add_argument("--drop_cols", nargs="+", default=[], help="Comma-separated columns to drop")
    parser.add_argument("--output_dir", default="feature_mean_plots", help="Directory to save the plots")
    parser.add_argument("--exclude_decades", type=int, nargs='*', help="Decades to exclude, e.g. 1600 1610 1620")

    args = parser.parse_args()

    main(args)
