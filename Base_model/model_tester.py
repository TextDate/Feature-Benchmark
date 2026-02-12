import argparse
import os
import json
import glob
import traceback
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, roc_auc_score, average_precision_score,
    root_mean_squared_error, mean_absolute_error, r2_score,
    recall_score, precision_score
)
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning)
import seaborn as sns


def load_test_data(test_file, target_col, drop_cols, label_encoder, feature_names, use_centuries, exclude_centuries):
    df = pd.read_csv(test_file)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in test file.")

    if use_centuries:
        if "century" not in df.columns:
            raise ValueError("'century' column must be present in test file when using centuries.")

    if exclude_centuries:
        original_len = len(df)
        df = df[~df["century"].astype(int).isin(exclude_centuries)]
        removed = original_len - len(df)
        print(f"Excluded centuries: {exclude_centuries} ({removed} samples removed)", flush=True)

    # Extract target column before dropping it
    y = df[target_col]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    X = df.drop(columns=[target_col], errors='ignore')

    X = X[feature_names]
    y_encoded = label_encoder.transform(y)

    return X, y_encoded


def evaluate(model, X, y, model_name, label_encoder, args):
    y_pred = model.predict(X)
    
    results = {
        "model": model_name,
        "accuracy": accuracy_score(y, y_pred),
        "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y, y_pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y, y_pred, average="weighted", zero_division=0),
        "rmse": root_mean_squared_error(y, y_pred),
        "mae": mean_absolute_error(y, y_pred),
        "r2": r2_score(y, y_pred)
    }
    
    if args.predict_distribution:
        unique, counts = np.unique(y_pred, return_counts=True)
        prediction_distribution = dict(zip(unique, counts))
        results["prediction_distribution"] = prediction_distribution

    if args.save_confusion_matrix:

        cm = confusion_matrix(y, y_pred)
        labels = label_encoder.classes_
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        plt.figure(figsize=(20, 16))
        sns.heatmap(cm_df, cmap="Blues", cbar=True, xticklabels=True, yticklabels=True)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        cm_path = os.path.join(args.output_dir, f"{model_name}_confusion_matrix.png")
        os.makedirs(args.output_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        print(f"[INFO] Confusion matrix saved to {cm_path}")

    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X)
            results["auc_roc"] = roc_auc_score(y, y_prob, multi_class="ovr", average="macro")
            results["auprc"] = average_precision_score(y, y_prob, average="macro")

            def compute_topk_accuracy(y_true, y_prob, k):
                topk_correct = 0
                for true, probs in zip(y_true, y_prob):
                    topk = np.argsort(probs)[-k:]
                    if true in topk:
                        topk_correct += 1
                return topk_correct / len(y_true)

            if args.use_centuries:
                results["top2_accuracy"] = compute_topk_accuracy(y, y_prob, 2)
            else:
                results["top3_accuracy"] = compute_topk_accuracy(y, y_prob, 3)
                results["top5_accuracy"] = compute_topk_accuracy(y, y_prob, 5)
                results["top10_accuracy"] = compute_topk_accuracy(y, y_prob, 10)

        except:
            results["auc_roc"] = None
            results["auprc"] = None
    else:
        results["auc_roc"] = None
        results["auprc"] = None

    return results

def main(args):
    drop_cols = [col.strip() for group in args.drop_cols for col in group.split(",")]

    for model_path in args.models:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        print(f"[INFO] Loading model: {model_name}", flush=True)

        try:
            model, label_encoder, feature_names = joblib.load(model_path)
        except Exception as e:
            print(f"[ERROR] Failed to load model {model_path}: {e}")
            continue

        X, y = load_test_data(args.test_file, args.target, drop_cols, label_encoder, feature_names, args.use_centuries, args.exclude_centuries)
        result = evaluate(model, X, y, model_name, label_encoder, args)

        print(f"----- {model_name} Results -----", flush=True)
        for k, v in result.items():
            if isinstance(v, (int, float)):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
        print("------------------------------", flush=True)


        df_result = pd.DataFrame([result])
        os.makedirs(args.output_dir, exist_ok=True)
        suffix = f"_{args.dataset_suffix}" if args.dataset_suffix else ""
        output_csv = os.path.join(args.output_dir, f"{model_name}_results{suffix}.csv")
        df_result.to_csv(output_csv, index=False)
        print(f"[INFO] Results saved to {output_csv}\n")
        
        if args.predict_distribution:
            print(f"[INFO] Prediction distribution for {model_name}:")
            for cls, count in result["prediction_distribution"].items():
                print(f"  {label_encoder.inverse_transform([cls])[0]}: {count}")
            print("------------------------------", flush=True)


# =============================================================================
# Multi-seed testing functions (from multi_seed_tester.py)
# =============================================================================

def run_multi_seed_testing(args):
    """
    Test all models trained with different seeds and aggregate results

    Args:
        args: Testing arguments (similar to original tester)
    """

    # Find all trained models for this feature subset and model type
    model_pattern = f"Results/multi_seed/{args.feature_subset}/{args.time_scale}/{args.model}_seed_*_model.joblib"
    model_files = glob.glob(model_pattern)

    if not model_files:
        raise FileNotFoundError(f"No trained models found with pattern: {model_pattern}")

    print(f"Found {len(model_files)} trained models")

    # Storage for all test results
    all_test_results = []

    # Process each seed's model
    for model_file in sorted(model_files):
        try:
            # Extract seed from filename - use the last occurrence of '_seed_'
            filename = os.path.basename(model_file)  # Get just the filename, not full path

            if '_seed_' not in filename:
                print(f"Warning: Skipping file without seed pattern: {filename}")
                continue

            # Split by '_seed_' and get the part after it
            parts_after_seed = filename.split('_seed_')[1]
            # Extract just the numeric seed part (before next underscore or file extension)
            seed_str = parts_after_seed.split('_')[0]
            seed = int(seed_str)
            print(f"Extracted seed {seed} from {filename}")
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not extract seed from filename {model_file}")
            print(f"  Error: {e}")
            print(f"  Filename only: {os.path.basename(model_file)}")
            continue

        print(f"\n=== Testing seed {seed} ===")

        try:
            # Load model artifacts
            base_path = model_file.replace('_model.joblib', '')

            model = joblib.load(model_file)
            label_encoder = joblib.load(f"{base_path}_label_encoder.joblib")

            with open(f"{base_path}_feature_names.json", 'r') as f:
                feature_names = json.load(f)

            # Load test data
            X_test, y_test = load_test_data(
                args.test_file, args.target_col, args.drop_cols,
                label_encoder, feature_names, args.use_centuries,
                args.exclude_centuries
            )

            # Evaluate model
            test_results = evaluate(model, X_test, y_test, args.model, label_encoder, args)

            # Add seed information
            test_results['seed'] = seed
            test_results['feature_subset'] = args.feature_subset
            test_results['time_scale'] = args.time_scale
            test_results['test_dataset'] = args.dataset_name

            all_test_results.append(test_results)

            print(f"Seed {seed} - Test Accuracy: {test_results['accuracy']:.4f}")

        except Exception as e:
            print(f"ERROR testing seed {seed}: {e}")
            print("Full error traceback:")
            traceback.print_exc()
            continue

    if not all_test_results:
        raise RuntimeError("No models could be successfully tested")

    # Save individual results
    output_dir = f"Results/multi_seed/{args.feature_subset}/{args.time_scale}"
    os.makedirs(output_dir, exist_ok=True)

    results_file = f"{output_dir}/{args.model}_{args.dataset_name}_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_test_results, f, indent=2, default=str)

    # Compute and save test statistics
    test_statistics = compute_test_statistics(all_test_results, output_dir, args.model, args.dataset_name)

    print(f"\n=== MULTI-SEED TESTING COMPLETED ===")
    print(f"Successfully tested {len(all_test_results)} models")
    print(f"Results saved to: {results_file}")

    return all_test_results, test_statistics

def compute_test_statistics(test_results, output_dir, model_name, dataset_name):
    """Compute statistics across all test runs"""

    if not test_results:
        print("No test results to compute statistics")
        return {}

    # Extract metrics from all seeds
    metrics_data = {}

    # Get all possible metric names from the first result
    metric_names = [key for key in test_results[0].keys()
                   if key not in ['seed', 'feature_subset', 'time_scale', 'test_dataset', 'model']]

    for metric in metric_names:
        values = []
        for result in test_results:
            if metric in result and result[metric] is not None:
                # Handle different data types
                val = result[metric]
                if isinstance(val, (int, float)):
                    values.append(float(val))

        if values:
            metrics_data[metric] = {
                'values': values,
                'mean': np.mean(values),
                'std': np.std(values, ddof=1) if len(values) > 1 else 0,
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'count': len(values)
            }

            # 95% confidence interval
            if len(values) > 1:
                sem = np.std(values, ddof=1) / np.sqrt(len(values))
                ci_95 = 1.96 * sem
                metrics_data[metric]['sem'] = sem
                metrics_data[metric]['ci_95'] = ci_95
                metrics_data[metric]['ci_95_lower'] = metrics_data[metric]['mean'] - ci_95
                metrics_data[metric]['ci_95_upper'] = metrics_data[metric]['mean'] + ci_95

    # Save statistics
    stats_file = f"{output_dir}/{model_name}_{dataset_name}_test_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(metrics_data, f, indent=2, default=str)

    # Create summary DataFrame
    summary_data = []
    for metric, stats in metrics_data.items():
        summary_data.append({
            'metric': metric,
            'mean': stats['mean'],
            'std': stats['std'],
            'min': stats['min'],
            'max': stats['max'],
            'median': stats['median'],
            'count': stats['count'],
            'sem': stats.get('sem', np.nan),
            'ci_95_lower': stats.get('ci_95_lower', np.nan),
            'ci_95_upper': stats.get('ci_95_upper', np.nan)
        })

    summary_df = pd.DataFrame(summary_data)
    summary_csv = f"{output_dir}/{model_name}_{dataset_name}_test_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"\nTest Statistics Summary ({dataset_name}):")
    print(f"{'Metric':<20} {'Mean':<10} {'Std':<10} {'95% CI':<25}")
    print("-" * 65)
    for metric, stats in metrics_data.items():
        if 'ci_95_lower' in stats and not np.isnan(stats['ci_95_lower']):
            ci_str = f"[{stats['ci_95_lower']:.4f}, {stats['ci_95_upper']:.4f}]"
        else:
            ci_str = "N/A"
        print(f"{metric:<20} {stats['mean']:<10.4f} {stats['std']:<10.4f} {ci_str}")

    return metrics_data


# =============================================================================
# Aggregation functions (from aggregate_multi_seed_results.py)
# =============================================================================

def aggregate_all_results(base_dir="Results/multi_seed"):
    """Aggregate all multi-seed results into summary files"""

    print("Starting aggregation of multi-seed results...")

    # Find all result directories
    if not os.path.exists(base_dir):
        print(f"Results directory {base_dir} not found!")
        return

    # Storage for aggregated data
    training_summary = []
    testing_summary = []

    # Pattern to find all feature subsets and time scales
    pattern = f"{base_dir}/*/*"
    result_dirs = glob.glob(pattern)

    print(f"Found {len(result_dirs)} result directories")

    for result_dir in result_dirs:
        if not os.path.isdir(result_dir):
            continue

        # Extract feature_subset and time_scale from path
        path_parts = result_dir.split(os.sep)
        if len(path_parts) >= 3:
            feature_subset = path_parts[-2]
            time_scale = path_parts[-1]
        else:
            continue

        print(f"Processing: {feature_subset}/{time_scale}")

        # Process training results
        training_files = glob.glob(f"{result_dir}/*_statistics.json")
        for train_file in training_files:
            model_name = os.path.basename(train_file).replace("_statistics.json", "")
            training_summary.extend(
                process_training_results(train_file, model_name, feature_subset, time_scale)
            )

        # Process testing results
        test_files = glob.glob(f"{result_dir}/*_test_statistics.json")
        for test_file in test_files:
            base_name = os.path.basename(test_file).replace("_test_statistics.json", "")
            parts = base_name.split("_")
            if len(parts) >= 2:
                model_name = parts[0]
                dataset_name = "_".join(parts[1:])
            else:
                continue

            testing_summary.extend(
                process_testing_results(test_file, model_name, dataset_name, feature_subset, time_scale)
            )

    # Create summary DataFrames
    if training_summary:
        training_df = pd.DataFrame(training_summary)
        training_df.to_csv(f"{base_dir}/training_summary_all.csv", index=False)
        print(f"Training summary saved: {len(training_summary)} records")

        # Create pivot tables for easy analysis
        create_training_pivot_tables(training_df, base_dir)

    if testing_summary:
        testing_df = pd.DataFrame(testing_summary)
        testing_df.to_csv(f"{base_dir}/testing_summary_all.csv", index=False)
        print(f"Testing summary saved: {len(testing_summary)} records")

        # Create pivot tables for easy analysis
        create_testing_pivot_tables(testing_df, base_dir)

    # Create comparison tables
    if training_summary and testing_summary:
        create_comparison_tables(training_df, testing_df, base_dir)

    print("Aggregation completed!")

def process_training_results(stats_file, model_name, feature_subset, time_scale):
    """Process individual training statistics file"""
    try:
        with open(stats_file, 'r') as f:
            stats = json.load(f)

        results = []
        for metric_name, metric_stats in stats.items():
            result = {
                'model': model_name,
                'feature_subset': feature_subset,
                'time_scale': time_scale,
                'phase': 'training',
                'metric': metric_name,
                'mean': metric_stats.get('mean', np.nan),
                'std': metric_stats.get('std', np.nan),
                'min': metric_stats.get('min', np.nan),
                'max': metric_stats.get('max', np.nan),
                'median': metric_stats.get('median', np.nan),
                'count': metric_stats.get('count', 0),
                'sem': metric_stats.get('sem', np.nan),
                'ci_95_lower': metric_stats.get('ci_95_lower', np.nan),
                'ci_95_upper': metric_stats.get('ci_95_upper', np.nan)
            }
            results.append(result)
        return results

    except Exception as e:
        print(f"Error processing {stats_file}: {e}")
        return []

def process_testing_results(stats_file, model_name, dataset_name, feature_subset, time_scale):
    """Process individual testing statistics file"""
    try:
        with open(stats_file, 'r') as f:
            stats = json.load(f)

        results = []
        for metric_name, metric_stats in stats.items():
            result = {
                'model': model_name,
                'feature_subset': feature_subset,
                'time_scale': time_scale,
                'dataset': dataset_name,
                'phase': 'testing',
                'metric': metric_name,
                'mean': metric_stats.get('mean', np.nan),
                'std': metric_stats.get('std', np.nan),
                'min': metric_stats.get('min', np.nan),
                'max': metric_stats.get('max', np.nan),
                'median': metric_stats.get('median', np.nan),
                'count': metric_stats.get('count', 0),
                'sem': metric_stats.get('sem', np.nan),
                'ci_95_lower': metric_stats.get('ci_95_lower', np.nan),
                'ci_95_upper': metric_stats.get('ci_95_upper', np.nan)
            }
            results.append(result)
        return results

    except Exception as e:
        print(f"Error processing {stats_file}: {e}")
        return []

def create_training_pivot_tables(df, base_dir):
    """Create pivot tables for training results"""

    # Accuracy pivot table
    accuracy_df = df[df['metric'] == 'Accuracy'].copy()
    if not accuracy_df.empty:
        accuracy_pivot = accuracy_df.pivot_table(
            values='mean',
            index=['feature_subset', 'time_scale'],
            columns='model',
            fill_value=np.nan
        )
        accuracy_pivot.to_csv(f"{base_dir}/training_accuracy_pivot.csv")

        # Standard deviation pivot table
        accuracy_std_pivot = accuracy_df.pivot_table(
            values='std',
            index=['feature_subset', 'time_scale'],
            columns='model',
            fill_value=np.nan
        )
        accuracy_std_pivot.to_csv(f"{base_dir}/training_accuracy_std_pivot.csv")

def create_testing_pivot_tables(df, base_dir):
    """Create pivot tables for testing results"""

    # Accuracy pivot table by dataset
    accuracy_df = df[df['metric'] == 'accuracy'].copy()
    if not accuracy_df.empty:
        for dataset in accuracy_df['dataset'].unique():
            dataset_df = accuracy_df[accuracy_df['dataset'] == dataset]

            accuracy_pivot = dataset_df.pivot_table(
                values='mean',
                index=['feature_subset', 'time_scale'],
                columns='model',
                fill_value=np.nan
            )
            accuracy_pivot.to_csv(f"{base_dir}/testing_accuracy_pivot_{dataset}.csv")

            # Standard deviation pivot table
            accuracy_std_pivot = dataset_df.pivot_table(
                values='std',
                index=['feature_subset', 'time_scale'],
                columns='model',
                fill_value=np.nan
            )
            accuracy_std_pivot.to_csv(f"{base_dir}/testing_accuracy_std_pivot_{dataset}.csv")

def create_comparison_tables(training_df, testing_df, base_dir):
    """Create comparison tables between training and testing"""

    # Best models summary
    best_models = []

    for feature_subset in training_df['feature_subset'].unique():
        for time_scale in training_df['time_scale'].unique():
            # Get training accuracy
            train_acc = training_df[
                (training_df['feature_subset'] == feature_subset) &
                (training_df['time_scale'] == time_scale) &
                (training_df['metric'] == 'Accuracy')
            ]

            if train_acc.empty:
                continue

            # Find best model by training accuracy
            best_model = train_acc.loc[train_acc['mean'].idxmax()]

            # Get corresponding test accuracies
            for dataset in ['test', 'gutenberg']:
                test_acc = testing_df[
                    (testing_df['feature_subset'] == feature_subset) &
                    (testing_df['time_scale'] == time_scale) &
                    (testing_df['model'] == best_model['model']) &
                    (testing_df['dataset'] == dataset) &
                    (testing_df['metric'] == 'accuracy')
                ]

                if not test_acc.empty:
                    test_result = test_acc.iloc[0]
                    best_models.append({
                        'feature_subset': feature_subset,
                        'time_scale': time_scale,
                        'model': best_model['model'],
                        'dataset': dataset,
                        'train_accuracy_mean': best_model['mean'],
                        'train_accuracy_std': best_model['std'],
                        'test_accuracy_mean': test_result['mean'],
                        'test_accuracy_std': test_result['std'],
                        'generalization_gap': best_model['mean'] - test_result['mean']
                    })

    if best_models:
        best_models_df = pd.DataFrame(best_models)
        best_models_df.to_csv(f"{base_dir}/best_models_comparison.csv", index=False)

    print("Pivot tables and comparison tables created")


# =============================================================================
# Multi-seed testing main entry point
# =============================================================================

def multi_seed_main(args):
    """Entry point for multi-seed subcommand"""
    test_results, statistics = run_multi_seed_testing(args)
    return test_results, statistics


# =============================================================================
# Aggregate main entry point
# =============================================================================

def aggregate_main(args):
    """Entry point for aggregate subcommand"""
    aggregate_all_results(base_dir=args.base_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model testing and multi-seed evaluation")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # --- test subcommand (original model_tester.py behavior) ---
    test_parser = subparsers.add_parser("test", help="Evaluate saved models on test data")
    test_parser.add_argument("--test_file", required=True, help="CSV file with test data")
    test_parser.add_argument("--target", required=True, help="Name of target column (e.g. 'decade')")
    test_parser.add_argument("--models", nargs='+', required=True, help="Path(s) to model .pkl files")
    test_parser.add_argument("--drop_cols", nargs='+', required=True, help='Columns to drop (comma-separated)')
    test_parser.add_argument("--output_dir", default="model_eval_results", help="Where to save CSV with results")
    test_parser.add_argument("--save_confusion_matrix", action="store_true", help="Save confusion matrix plot")
    test_parser.add_argument("--use_centuries", action="store_true", help="Convert decades to centuries for evaluation")
    test_parser.add_argument("--exclude_centuries", type=int, nargs='*', help="Centuries to exclude (e.g. 17 18)")
    test_parser.add_argument("--predict_distribution", action="store_true", help="Print prediction distribution for each class")
    test_parser.add_argument("--dataset_suffix", default="", help="Suffix to identify dataset type (e.g., 'test', 'gutenberg')")

    # --- multi-seed subcommand (from multi_seed_tester.py) ---
    ms_parser = subparsers.add_parser("multi-seed", help="Multi-seed model testing")
    ms_parser.add_argument("--test_file", required=True, help="Test CSV file")
    ms_parser.add_argument("--feature_subset", required=True, help="Feature subset name")
    ms_parser.add_argument("--model", required=True, help="Model name to test")
    ms_parser.add_argument("--time_scale", required=True, choices=["decades", "centuries"], help="Time scale")
    ms_parser.add_argument("--target_col", default="target", help="Target column name")
    ms_parser.add_argument("--drop_cols", nargs="*", default=["text", "title", "year", "dataset"], help="Columns to drop")
    ms_parser.add_argument("--use_centuries", action="store_true", help="Use century target")
    ms_parser.add_argument("--exclude_centuries", nargs="*", type=int, help="Centuries to exclude")
    ms_parser.add_argument("--dataset_name", default="test", help="Name of test dataset (for labeling)")
    ms_parser.add_argument("--save_plots", action="store_true", help="Save confusion matrix and other plots")
    ms_parser.add_argument("--predict_distribution", action="store_true", help="Show prediction distribution")
    ms_parser.add_argument("--save_confusion_matrix", action="store_true", help="Save confusion matrix plot")
    ms_parser.add_argument("--plot_format", default="png", choices=["png", "pdf", "svg"], help="Plot format")

    # --- aggregate subcommand (from aggregate_multi_seed_results.py) ---
    agg_parser = subparsers.add_parser("aggregate", help="Aggregate multi-seed results")
    agg_parser.add_argument("--base_dir", default="Results/multi_seed", help="Base directory for multi-seed results")

    args = parser.parse_args()

    if args.command == "test":
        main(args)
    elif args.command == "multi-seed":
        multi_seed_main(args)
    elif args.command == "aggregate":
        aggregate_main(args)
    else:
        parser.print_help()

