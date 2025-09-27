import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, roc_auc_score, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score,
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
        "rmse": mean_squared_error(y, y_pred),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved models on test data")
    parser.add_argument("--test_file", required=True, help="CSV file with test data")
    parser.add_argument("--target", required=True, help="Name of target column (e.g. 'decade')")
    parser.add_argument("--models", nargs='+', required=True, help="Path(s) to model .pkl files")
    parser.add_argument("--drop_cols", nargs='+', required=True, help='Columns to drop (comma-separated)')
    parser.add_argument("--output_dir", default="model_eval_results", help="Where to save CSV with results")
    parser.add_argument("--save_confusion_matrix", action="store_true", help="Save confusion matrix plot")
    parser.add_argument("--use_centuries", action="store_true", help="Convert decades to centuries for evaluation")
    parser.add_argument("--exclude_centuries", type=int, nargs='*', help="Centuries to exclude (e.g. 17 18)")
    parser.add_argument("--predict_distribution", action="store_true", help="Print prediction distribution for each class")
    parser.add_argument("--dataset_suffix", default="", help="Suffix to identify dataset type (e.g., 'test', 'gutenberg')")
    args = parser.parse_args()
    main(args)
    
