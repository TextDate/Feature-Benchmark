import argparse
import os
import json
import time
import joblib
import psutil
import pandas as pd
import numpy as np
import warnings

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    root_mean_squared_error, mean_absolute_error, r2_score,
    precision_score, recall_score
)
from sklearn.exceptions import DataConversionWarning
from torch import zero_
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


MODELS = {
    "random_forest": RandomForestClassifier(random_state=42, max_depth=None, class_weight="balanced", verbose=0, n_jobs=-1),
    "gradient_boosting": HistGradientBoostingClassifier(random_state=42, verbose=0, class_weight="balanced"),
    "xgboost": XGBClassifier(random_state=42, eval_metric='mlogloss', verbosity=0, n_jobs=-1),
    "lightgbm": LGBMClassifier(random_state=42, verbose=0, n_jobs=-1, class_weight="balanced"),
    "catboost": CatBoostClassifier(random_state=42, verbose=0, thread_count=-1),
    "svm": SVC(probability=True, random_state=42),
    "gnb": GaussianNB(),
    "knn": KNeighborsClassifier(n_jobs=-1)
}


def get_feature_subset_drop_cols(feature_subset):
    """Get drop columns for feature subset from feature_configs.json"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'feature_configs.json')

    with open(config_path, 'r') as f:
        feature_configs = json.load(f)

    if feature_subset not in feature_configs:
        raise ValueError(f"Unknown feature subset: {feature_subset}. Available subsets: {list(feature_configs.keys())}")

    return feature_configs[feature_subset]


def filter_features_by_subset(X_train, X_val, feature_subset):
    """
    Filter training and validation datasets by dropping columns for the specified subset.
    Uses the same drop-cols approach as run_all_feature_models.sh

    Args:
        X_train: Training feature DataFrame
        X_val: Validation feature DataFrame
        feature_subset: Name of the feature subset to use

    Returns:
        Filtered X_train, X_val, and list of remaining feature names
    """
    # Get drop columns for this feature subset
    drop_cols = get_feature_subset_drop_cols(feature_subset)

    # Filter out drop columns that don't exist in the dataset
    existing_drop_cols = [col for col in drop_cols if col in X_train.columns]

    # Drop the specified columns
    X_train_filtered = X_train.drop(columns=existing_drop_cols, errors='ignore')
    X_val_filtered = X_val.drop(columns=existing_drop_cols, errors='ignore')

    remaining_features = list(X_train_filtered.columns)

    if not remaining_features:
        raise ValueError(f"No features remaining after dropping columns for subset '{feature_subset}'. All columns were dropped.")

    print(f"Dropped {len(existing_drop_cols)} columns for subset '{feature_subset}'")
    print(f"Remaining {len(remaining_features)} features: {remaining_features[:5]}{'...' if len(remaining_features) > 5 else ''}")

    return X_train_filtered, X_val_filtered, remaining_features


def load_data(train_file, val_file, target_col, drop_cols, use_smote, use_tomek, use_centuries, exclude_decades=None, exclude_centuries=None):
    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)

    if use_centuries:
        if "century" not in df_train.columns or "century" not in df_val.columns:
            raise ValueError("'century' column must exist in both train and validation files when use_centuries is enabled.")
        if exclude_centuries:
            df_train = df_train[~df_train["century"].astype(int).isin(exclude_centuries)]
            df_val = df_val[~df_val["century"].astype(int).isin(exclude_centuries)]
            print(f"Excluded centuries: {exclude_centuries}", flush=True)
            
    elif exclude_decades:
        df_train = df_train[~df_train["decade"].astype(int).isin(exclude_decades)]
        df_val = df_val[~df_val["decade"].astype(int).isin(exclude_decades)]
        print(f"Excluded decades: {exclude_decades}", flush=True)

    # Extract target columns before dropping them
    y_train_df = df_train[[target_col]]
    y_val_df = df_val[[target_col]]

    df_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns], errors='ignore')
    df_val = df_val.drop(columns=[c for c in drop_cols if c in df_val.columns], errors='ignore')

    common_features = df_train.columns.intersection(df_val.columns).tolist()

    X_train, y_train = df_train[common_features], y_train_df
    X_val, y_val = df_val[common_features], y_val_df

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train.values.ravel())
    y_val = label_encoder.transform(y_val.values.ravel())

    if use_smote:
        print("Applying SMOTE...", flush=True)
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    if use_tomek:
        print("Applying TomekLinks...", flush=True)
        tomek = TomekLinks()
        X_train, y_train = tomek.fit_resample(X_train, y_train)

    return X_train, y_train, X_val, y_val, label_encoder


def evaluate_model(model, X_val, y_val, label_encoder, args):
    y_pred = model.predict(X_val)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    y_val_decoded = label_encoder.inverse_transform(y_val)

    results = {
        "Accuracy": accuracy_score(y_val_decoded, y_pred_decoded),
        "F1 Macro": f1_score(y_val_decoded, y_pred_decoded, average="macro", zero_division=0),
        "F1 Weighted": f1_score(y_val_decoded, y_pred_decoded, average="weighted", zero_division=0),
        "Recall Macro": recall_score(y_val_decoded, y_pred_decoded, average="macro", zero_division=0),
        "Recall Weighted": recall_score(y_val_decoded, y_pred_decoded, average="weighted", zero_division=0),
        "Precision Macro": precision_score(y_val_decoded, y_pred_decoded, average="macro", zero_division=0),
        "Precision Weighted": precision_score(y_val_decoded, y_pred_decoded, average="weighted", zero_division=0),
        "RMSE": root_mean_squared_error(y_val, y_pred),  # Use encoded class indices, not decoded values
        "MAE": mean_absolute_error(y_val, y_pred),               # Use encoded class indices, not decoded values
        "R2": r2_score(y_val, y_pred)                           # Use encoded class indices, not decoded values
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val)
        results["AUC-ROC"] = roc_auc_score(y_val, y_proba, multi_class="ovr", average="macro")
        results["AUPRC"] = average_precision_score(y_val, y_proba, average="macro")

        def compute_topk_accuracy(y_true, y_prob, k):
            topk_correct = 0
            for true, probs in zip(y_true, y_prob):
                topk = np.argsort(probs)[-k:]
                if true in topk:
                    topk_correct += 1
            return topk_correct / len(y_true)

        if args.use_centuries:
            results["Top-2 Accuracy"] = compute_topk_accuracy(y_val, y_proba, 2)
        else:
            results["Top-3 Accuracy"] = compute_topk_accuracy(y_val, y_proba, 3)
            results["Top-5 Accuracy"] = compute_topk_accuracy(y_val, y_proba, 5)
            results["Top-10 Accuracy"] = compute_topk_accuracy(y_val, y_proba, 10)

    print("----- Validation Results -----", flush=True)
    for k, v in results.items():
        print(f"{k}: {v:.4f}", flush=True)
    print("------------------------------", flush=True)
    return results


def run_multi_seed_training(args, seeds=None):
    """
    Run model training with multiple random seeds and store results

    Args:
        args: Training arguments (same as original trainer)
        seeds: List of random seeds to use. If None, generates default seeds
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 2024, 2025, 3141, 7777, 9999, 13579]

    print(f"Starting multi-seed training with seeds: {seeds}")
    print(f"Model: {args.model}, Feature subset: {args.feature_subset}")

    # Load data once (seeds only affect model initialization, not data splits)
    print("Loading data...")
    X_train, y_train, X_val, y_val, label_encoder = load_data(
        args.train_file, args.val_file, args.target_col, args.drop_cols,
        args.use_smote, args.use_tomek, args.use_centuries, args.exclude_decades, args.exclude_centuries
    )

    # Apply feature subset filtering
    print(f"Applying feature subset filtering for: {args.feature_subset}")
    X_train, X_val, feature_names = filter_features_by_subset(X_train, X_val, args.feature_subset)

    print(f"Training with {len(feature_names)} features: {feature_names[:5]}{'...' if len(feature_names) > 5 else ''}")

    # Storage for all results across seeds
    all_results = []

    # Create output directories
    base_output_dir = f"Results/multi_seed/{args.feature_subset}/{args.time_scale}"
    os.makedirs(base_output_dir, exist_ok=True)

    for i, seed in enumerate(seeds):
        print(f"\n=== SEED {seed} ({i+1}/{len(seeds)}) ===")

        # Update model with new seed
        base_model = MODELS[args.model]
        model_config = base_model.get_params()

        # Update random state for models that support it
        if 'random_state' in model_config:
            model_config['random_state'] = seed

        # Handle special cases for different model types
        if args.model == "gnb":
            # GaussianNB doesn't have random_state
            from sklearn.naive_bayes import GaussianNB
            model = GaussianNB()
        elif args.model == "knn":
            # KNN doesn't have random_state
            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier(n_jobs=-1)
        else:
            # Create new model instance with updated seed
            model_class = type(base_model)
            model = model_class(**model_config)
            print(f"{args.model} configured with random_state={seed}")

        try:
            # Train model
            print(f"Training {args.model} with seed {seed}...")

            process = psutil.Process()
            start_ram = process.memory_info().rss / (1024 ** 3)
            start_time = time.time()

            model.fit(X_train, y_train)

            train_time = time.time() - start_time
            end_ram = process.memory_info().rss / (1024 ** 3)

            # Evaluate model
            training_metrics = evaluate_model(model, X_val, y_val, label_encoder, args)
            training_metrics["Training Time"] = train_time
            training_metrics["RAM Usage (GB)"] = end_ram - start_ram

            # Store results
            result = {
                'seed': seed,
                'model_name': args.model,
                'feature_subset': args.feature_subset,
                'time_scale': args.time_scale,
                'training_metrics': training_metrics
            }

            all_results.append(result)

            # Save individual model
            model_filename = f"{args.model}_seed_{seed}"
            save_model_and_artifacts(
                model, training_metrics, label_encoder, feature_names,
                base_output_dir, model_filename
            )

            print(f"Seed {seed} completed. Validation accuracy: {training_metrics['Accuracy']:.4f}")

        except Exception as e:
            print(f"ERROR with seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save aggregated results
    results_file = f"{base_output_dir}/{args.model}_multi_seed_summary.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Compute and save statistics
    compute_and_save_statistics(all_results, base_output_dir, args.model)

    print(f"\n=== MULTI-SEED TRAINING COMPLETED ===")
    print(f"Results saved to: {base_output_dir}")
    print(f"Summary statistics computed for {len(all_results)} successful runs")

    return all_results

def save_model_and_artifacts(model, metrics, label_encoder, feature_names, output_dir, model_filename):
    """Save model and associated artifacts"""

    # Save model
    model_path = f"{output_dir}/{model_filename}_model.joblib"
    joblib.dump(model, model_path)

    # Save label encoder
    encoder_path = f"{output_dir}/{model_filename}_label_encoder.joblib"
    joblib.dump(label_encoder, encoder_path)

    # Save feature names
    features_path = f"{output_dir}/{model_filename}_feature_names.json"
    with open(features_path, 'w') as f:
        json.dump(feature_names, f)

    # Save metrics
    metrics_path = f"{output_dir}/{model_filename}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

def compute_and_save_statistics(results, output_dir, model_name):
    """Compute mean, std, and confidence intervals across seeds"""

    if not results:
        print("No results to compute statistics")
        return

    # Extract metrics from all seeds
    metrics_data = {}
    metric_names = list(results[0]['training_metrics'].keys())

    for metric in metric_names:
        values = [r['training_metrics'][metric] for r in results if r['training_metrics'].get(metric) is not None]
        if values:
            metrics_data[metric] = {
                'values': values,
                'mean': np.mean(values),
                'std': np.std(values, ddof=1),  # Sample standard deviation
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'count': len(values)
            }

            # 95% confidence interval (assuming normal distribution)
            if len(values) > 1:
                sem = np.std(values, ddof=1) / np.sqrt(len(values))  # Standard error of mean
                ci_95 = 1.96 * sem  # 95% CI
                metrics_data[metric]['sem'] = sem
                metrics_data[metric]['ci_95'] = ci_95
                metrics_data[metric]['ci_95_lower'] = metrics_data[metric]['mean'] - ci_95
                metrics_data[metric]['ci_95_upper'] = metrics_data[metric]['mean'] + ci_95

    # Save statistics
    stats_file = f"{output_dir}/{model_name}_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(metrics_data, f, indent=2, default=str)

    # Create summary DataFrame and save as CSV
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
    summary_csv = f"{output_dir}/{model_name}_summary_statistics.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"\nStatistics summary:")
    print(f"{'Metric':<20} {'Mean':<10} {'Std':<10} {'95% CI':<20}")
    print("-" * 60)
    for metric, stats in metrics_data.items():
        if 'ci_95_lower' in stats:
            ci_str = f"[{stats['ci_95_lower']:.4f}, {stats['ci_95_upper']:.4f}]"
        else:
            ci_str = "N/A"
        print(f"{metric:<20} {stats['mean']:<10.4f} {stats['std']:<10.4f} {ci_str}")


def train_and_save_models(args):
    X_train, y_train, X_val, y_val, label_encoder = load_data(
        args.train_file, args.val_file, args.target,
        args.drop_cols.split(","), args.use_smote, args.use_tomek,
        args.use_centuries, args.exclude_centuries, args.exclude_decades,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for model_name in args.models:
        print(f"\n=== Training {model_name} with {args.n_estimators} estimators ===", flush=True)
        base_model = MODELS[model_name]

        if model_name == "gradient_boosting":
            model = HistGradientBoostingClassifier(max_iter=args.n_estimators, random_state=42, verbose=0)
        else:
            model_class = base_model.__class__
            params = base_model.get_params()
            if "n_estimators" in model_class().get_params():
                params["n_estimators"] = args.n_estimators
            model = model_class(**params)

        process = psutil.Process()
        start_ram = process.memory_info().rss / (1024 ** 3)
        start_time = time.time()

        model.fit(X_train, y_train)

        train_time = time.time() - start_time
        end_ram = process.memory_info().rss / (1024 ** 3)
        ram_used = end_ram - start_ram

        print(f"Training completed in {train_time:.2f}s, RAM used: {ram_used:.2f} GB", flush=True)
        results = evaluate_model(model, X_val, y_val, label_encoder, args)

        output_path = os.path.join(args.output_dir, f"{model_name}_model.pkl")
        joblib.dump((model, label_encoder, X_train.columns.tolist()), output_path)
        print(f"Model saved to: {output_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save multiple tree-based models.")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # --- train subcommand (existing behavior) ---
    train_parser = subparsers.add_parser("train", help="Train and save models (original workflow)")
    train_parser.add_argument("--train_file", type=str, required=True)
    train_parser.add_argument("--val_file", type=str, required=True)
    train_parser.add_argument("--target", type=str, required=True)
    train_parser.add_argument("--models", type=str, nargs='+', required=True, choices=MODELS.keys())
    train_parser.add_argument("--n_estimators", type=int, required=True)
    train_parser.add_argument("--drop_cols", type=str, required=True)
    train_parser.add_argument("--output_dir", type=str, required=True)
    train_parser.add_argument("--use_smote", action="store_true")
    train_parser.add_argument("--use_tomek", action="store_true")
    train_parser.add_argument("--exclude_decades", type=int, nargs='*', help="Decades to exclude, e.g. 1600 1610 1620")
    train_parser.add_argument("--use_centuries", action="store_true")
    train_parser.add_argument("--exclude_centuries", type=int, nargs='*', help="Centuries to exclude (e.g. 17 18)")

    # --- multi-seed subcommand ---
    ms_parser = subparsers.add_parser("multi-seed", help="Multi-seed model training workflow")
    ms_parser.add_argument("--train_file", required=True, help="Training CSV file")
    ms_parser.add_argument("--val_file", required=True, help="Validation CSV file")
    ms_parser.add_argument("--feature_subset", required=True, help="Feature subset name")
    ms_parser.add_argument("--model", required=True, choices=MODELS.keys(), help="Model to train")
    ms_parser.add_argument("--time_scale", required=True, choices=["decades", "centuries"], help="Time scale")
    ms_parser.add_argument("--target_col", default="target", help="Target column name")
    ms_parser.add_argument("--drop_cols", nargs="*", default=["text", "title", "year", "dataset"], help="Columns to drop")
    ms_parser.add_argument("--use_smote", action="store_true", help="Use SMOTE for oversampling")
    ms_parser.add_argument("--use_tomek", action="store_true", help="Use Tomek links for undersampling")
    ms_parser.add_argument("--use_centuries", action="store_true", help="Use century target instead of decade")
    ms_parser.add_argument("--exclude_decades", nargs="*", type=int, help="Decades to exclude")
    ms_parser.add_argument("--exclude_centuries", nargs="*", type=int, help="Centuries to exclude")
    ms_parser.add_argument("--seeds", nargs="*", type=int, help="Custom list of seeds")
    ms_parser.add_argument("--n_seeds", type=int, default=10, help="Number of random seeds to generate")

    args = parser.parse_args()

    if args.command == "train":
        train_and_save_models(args)
    elif args.command == "multi-seed":
        # Generate seeds if not provided
        if args.seeds is None:
            np.random.seed(42)  # For reproducible seed generation
            args.seeds = np.random.randint(1, 10000, size=args.n_seeds).tolist()
        run_multi_seed_training(args, args.seeds)
    else:
        parser.print_help()
