import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_means_and_boxes(dfs, target_col, output_dir, exclude_values=None):
    os.makedirs(output_dir, exist_ok=True)
    
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
            df.loc[:, col] = df[col].clip(lower=q_low, upper=q_high)
            current_min = df[col].min()
            current_max = df.groupby(target_col)[col].mean().max()
            range_buffer = (current_max - current_min) * 0.02
            existing_min, existing_max = min_max.get(col, (float("inf"), float("-inf")))
            min_max[col] = (
                min(existing_min, current_min - range_buffer),
                max(existing_max, current_max + range_buffer)
            )

    for col in min_max:
        for name, df in dfs.items():
            os.makedirs(os.path.join(output_dir, name), exist_ok=True)
            grouped = df.groupby(target_col).mean(numeric_only=True).sort_index()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(grouped.index, grouped[col], marker='o')
            ax.set_title(f"{name}: Mean of '{col}' per {target_col}")
            ax.set_xlabel(target_col.capitalize())
            ax.set_ylabel(f"Mean {col}")
            ymin, ymax = min_max[col]
            ax.set_ylim(ymin, ymax)
            ax.grid(True)
            fig.tight_layout()
            line_path = os.path.join(output_dir, name, f"{col}_mean_by_{target_col}.png")
            fig.savefig(line_path)
            plt.close(fig)
            print(f"[INFO] Saved line plot to {line_path}", flush=True)

            fig, ax = plt.subplots(figsize=(20, 6))
            df.boxplot(column=col, by=target_col, grid=True, ax=ax)
            ax.set_title(f"{name}: Box Plot of '{col}' by {target_col.capitalize()}")
            fig.suptitle("")
            ax.set_xlabel(target_col.capitalize())
            ax.set_ylabel(col)
            ymin, ymax = min_max[col]
            ax.set_ylim(ymin, ymax)
            fig.tight_layout()
            box_path = os.path.join(output_dir,name ,f"{col}_box_by_{target_col}.png")
            fig.savefig(box_path)
            plt.close(fig)
            print(f"[INFO] Saved box plot to {box_path}", flush=True)

            fig, ax = plt.subplots(figsize=(20, 6))
            data_to_plot = []
            x_labels = []
            for label, group in df.groupby(target_col):
                data_to_plot.append(group[col])
                x_labels.append(label)
            ax.violinplot(data_to_plot, showmeans=True)
            ax.set_xticks(range(1, len(x_labels) + 1))
            ax.set_xticklabels(x_labels, rotation=45)
            ax.set_title(f"{name}: Violin Plot of '{col}' by {target_col}")
            ax.set_xlabel(target_col.capitalize())
            ax.set_ylabel(f"{col}")
            ymin, ymax = min_max[col]
            ax.set_ylim(ymin, ymax)
            ax.grid(True)
            fig.tight_layout()
            violin_path = os.path.join(output_dir,name ,f"{col}_violin_by_{target_col}.png")
            fig.savefig(violin_path)
            plt.close(fig)
            print(f"[INFO] Saved violin plot to {violin_path}", flush=True)


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
