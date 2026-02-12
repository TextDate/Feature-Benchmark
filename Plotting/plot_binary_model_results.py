import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class BinaryModelResultsPlotter:
    def __init__(self, csv_files, dataset_names, output_dir="binary_model_plots"):
        self.csv_files = csv_files
        self.dataset_names = dataset_names
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.metrics_to_plot = [
            "accuracy", "auc_roc", "auprc",
            "f1_macro", "recall_macro", "precision_macro", "%_of_texts"
        ]

    def plot_individual_metrics(self, df, dataset_name):
        for metric in self.metrics_to_plot:
            plt.figure()
            for model_name, group in df.groupby("model"):
                plt.plot(group["threshold"], group[metric], label=model_name)
            plt.xlabel("Threshold")
            plt.ylabel(metric.replace("_", " ").title())
            plt.ylim(0, 1)
            plt.title(f"{dataset_name}: {metric.replace('_', ' ').title()} vs Threshold")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir,f"{dataset_name}_{metric}_vs_threshold.png"))
            plt.close()

    def plot_combined_metrics(self, df, dataset_name):
        num_metrics = len(self.metrics_to_plot)
        rows = (num_metrics + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(16, 4 * rows))
        axes = axes.flatten()
        os.makedirs(os.path.join(self.output_dir, "catboostVSrandom"), exist_ok=True)

        for idx, metric in enumerate(self.metrics_to_plot):
            ax = axes[idx]
            for model_name, group in df.groupby("model"):
                if model_name == "xgboost_model":
                    continue
                ax.plot(group["threshold"], group[metric], label=model_name)
            clean_label = metric.replace("_", " ").title()
            ax.set_title(f"{clean_label} vs Threshold", fontsize=14)
            ax.set_xlabel("Threshold", fontsize=12)
            ax.set_ylabel(clean_label, fontsize=12)
            ax.set_ylim(0, 1)
            ax.grid(True)
            ax.legend(fontsize=10)

        for i in range(len(self.metrics_to_plot), len(axes)):
            fig.delaxes(axes[i])

        fig.suptitle(f"{dataset_name} - Combined Metric Trends", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(self.output_dir, "catboostVSrandom",f"{dataset_name}_combined_metrics_grid.png"))
        plt.close()

    def plot_across_datasets_comparison(self):
        os.makedirs(os.path.join(self.output_dir, "across_datasets_comparison"), exist_ok=True)
        data_per_dataset = {}
        for csv_file, dataset_name in zip(self.csv_files, self.dataset_names):
            df = pd.read_csv(csv_file)
            df_filtered = df[df["model"].isin(["catboost_model", "random"])]
            data_per_dataset[dataset_name] = df_filtered

        for metric in self.metrics_to_plot:
            plt.figure()
            for dataset_name, df in data_per_dataset.items():
                for model_name, group in df.groupby("model"):
                    plt.plot(group["threshold"], group[metric], label=f"{dataset_name} - {model_name}")
            plt.xlabel("Threshold")
            plt.ylabel(metric.replace("_", " ").title())
            plt.ylim(0, 1)
            plt.title(f"Comparison of {metric.replace('_', ' ').title()} across Datasets (catboost_model vs random)")
            plt.legend(fontsize=8)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "across_datasets_comparison", f"across_datasets_{metric}_comparison.png"))
            plt.close()

    def plot_combined_across_datasets_metrics(self):
        os.makedirs(os.path.join(self.output_dir, "combined_across_datasets_grid"), exist_ok=True)
        data_per_dataset = {}
        for csv_file, dataset_name in zip(self.csv_files, self.dataset_names):
            df = pd.read_csv(csv_file)
            df_filtered = df[df["model"].isin(["catboost_model", "random"])]
            data_per_dataset[dataset_name] = df_filtered

        num_metrics = len(self.metrics_to_plot)
        rows = (num_metrics + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(16, 4 * rows))
        axes = axes.flatten()

        for idx, metric in enumerate(self.metrics_to_plot):
            ax = axes[idx]
            for dataset_name, df in data_per_dataset.items():
                for model_name, group in df.groupby("model"):
                    ax.plot(group["threshold"], group[metric], label=f"{dataset_name} - {model_name}")
            clean_label = metric.replace("_", " ").title()
            ax.set_title(f"{clean_label} vs Threshold", fontsize=14)
            ax.set_xlabel("Threshold", fontsize=12)
            ax.set_ylabel(clean_label, fontsize=12)
            ax.set_ylim(0, 1)
            ax.grid(True)
            ax.legend(fontsize=10)

        for i in range(len(self.metrics_to_plot), len(axes)):
            fig.delaxes(axes[i])

        fig.suptitle("Combined Metric Comparison Across Datasets (catboost_model vs random)", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(self.output_dir, "combined_across_datasets_grid", "all_metrics_combined_across_datasets.png"))
        plt.close()
    
    def plot_model_comparison_across_datasets(self):
        os.makedirs(os.path.join(self.output_dir, "per_model_across_datasets"), exist_ok=True)
        data_per_dataset = {
            name: pd.read_csv(csv).query("model in ['catboost_model', 'random', 'xgboost_model']")
            for csv, name in zip(self.csv_files, self.dataset_names)
        }

        selected_metrics = ["accuracy", "%_of_texts", "f1_macro"]
        line_styles = {
            "accuracy": "-",
            "%_of_texts": "--",
            "f1_macro": "-."
        }

        models = ["catboost_model", "xgboost_model"]
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True, sharex=True)

        for i, model in enumerate(models):
            for j, (dataset_name, df) in enumerate(data_per_dataset.items()):
                ax = axes[i, j]
                model_data = df[df["model"] == model]
                for metric in selected_metrics:
                    ax.plot(
                        model_data["threshold"],
                        model_data[metric],
                        label=metric if i == 0 else None,
                        linestyle=line_styles.get(metric, "-")
                    )
                random_data = df[df["model"] == "random"]
                ax.plot(
                    random_data["threshold"],
                    random_data["accuracy"],
                    label="random" if i == 0 else None,
                    linestyle="-"
                )

                ax.set_title(f"{dataset_name} - {model.split('_')[0].capitalize()}")
                ax.set_xlabel("Threshold")
                ax.grid(True)

        axes[0, 0].set_ylabel("Metric Value")
        axes[1, 0].set_ylabel("Metric Value")
        for ax in axes[0]:
            ax.set_ylim(0, 1)
        fig.suptitle("CatBoost and XGBoost - Accuracy, % of Texts, and F1 Score Across Datasets", fontsize=16)
        axes[0, -1].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(self.output_dir, "per_model_across_datasets", "catboost_xgboost_side_by_side_metrics.png"))
        plt.close()
        
    def plot_au_metrics_per_model_across_datasets(self):
        os.makedirs(os.path.join(self.output_dir, "per_model_au_metrics_across_datasets"), exist_ok=True)
        data_per_dataset = {
            name: pd.read_csv(csv).query("model in ['catboost_model', 'random', 'xgboost_model']")
            for csv, name in zip(self.csv_files, self.dataset_names)
        }

        selected_metrics = ["auprc", "%_of_texts" ,"auc_roc"]
        line_styles = {
            "auprc": "-",
            "%_of_texts": "--",
            "auc_roc": "--"
        }

        models = ["catboost_model", "xgboost_model"]
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True, sharex=True)
        for i, model in enumerate(models):
            for j, (dataset_name, df) in enumerate(data_per_dataset.items()):
                ax = axes[i, j]
                model_data = df[df["model"] == model]
                for metric in selected_metrics:
                    ax.plot(
                        model_data["threshold"],
                        model_data[metric],
                        label=metric if i == 0 else None,
                        linestyle=line_styles.get(metric, "-")
                    )
                random_data = df[df["model"] == "random"]
                ax.plot(
                    random_data["threshold"],
                    random_data["auc_roc"],
                    label="random_auc_roc" if i == 0 else None,
                    linestyle=":"
                )
                ax.plot(
                    random_data["threshold"],
                    random_data["auprc"],
                    label="random_auprc" if i == 0 else None,
                    linestyle=":"
                )
                ax.set_title(f"{dataset_name} - {model.split('_')[0].capitalize()}")
                ax.set_xlabel("Threshold")
                ax.grid(True)

        axes[0, 0].set_ylabel("Metric Value")
        axes[1, 0].set_ylabel("Metric Value")
        for ax in axes[0]:
            ax.set_ylim(0, 1)
        fig.suptitle("CatBoost and XGBoost - AUPRC and AUROC Across Datasets", fontsize=16)
        axes[0, -1].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(self.output_dir, "per_model_au_metrics_across_datasets", "catboost_xgboost_au_metrics_side_by_side.png"))
        plt.close()

    def run(self):
        for csv_file, dataset_name in zip(self.csv_files, self.dataset_names):
            df = pd.read_csv(csv_file)
            self.plot_individual_metrics(df, dataset_name)
            self.plot_combined_metrics(df, dataset_name)
        self.plot_across_datasets_comparison()
        self.plot_combined_across_datasets_metrics()
        self.plot_model_comparison_across_datasets()
        self.plot_au_metrics_per_model_across_datasets()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot binary model results from multiple CSVs")
    parser.add_argument("--csv_files", type=str, nargs='+', required=True, help="List of CSV files with model results")
    parser.add_argument("--dataset_names", type=str, nargs='+', required=True, help="List of dataset names corresponding to CSV files")
    parser.add_argument("--output_dir", type=str, default="binary_model_plots", help="Directory to save output plots")
    args = parser.parse_args()

    if len(args.csv_files) != len(args.dataset_names):
        raise ValueError("Number of CSV files must match number of dataset names.")

    plotter = BinaryModelResultsPlotter(csv_files=args.csv_files, dataset_names=args.dataset_names, output_dir=args.output_dir)
    plotter.run()