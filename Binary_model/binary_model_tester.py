import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from re import X
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score, average_precision_score,
    f1_score, recall_score, precision_score,
)
import joblib
import os
import numpy as np


class BinaryModelTester:
    def __init__(self, models, target, drop_cols, output_dir, output_file, label_encoder, feature_names):
        self.models = models
        self.target = target
        self.drop_cols = drop_cols
        self.output_dir = output_dir
        self.output_file = output_file
        self.label_encoder = label_encoder
        self.feature_names = feature_names
    
    def plot_confusion_matrix_for_threshold(self, model_name, threshold):
        """
        Plot and save confusion matrix for the given model and threshold.
        """
        decoded_threshold = self.label_encoder.inverse_transform([threshold])[0]
        y_test_bin = (self.y_test >= threshold).astype(int)

        if model_name == "random":
            y_pred = self.predict_random_baseline(len(self.X_test))
            y_pred_bin = (y_pred >= threshold).astype(int)
        else:
            try:
                model, _, _ = joblib.load(model_name)
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                return

            y_pred = model.predict(self.X_test)
            y_pred_bin = (y_pred >= threshold).astype(int)

        cm = confusion_matrix(y_test_bin, y_pred_bin)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Older", "Newer"], yticklabels=["Older", "Newer"])
        plt.title(f"Confusion Matrix for {os.path.basename(model_name)}\nThreshold: {decoded_threshold}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()

        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(
            self.output_dir,
            f"confusion_matrix_{os.path.basename(model_name)}_thresh{decoded_threshold}.png"
        )
        plt.savefig(save_path)
        plt.close()
    
    def load_data(self, file_path, drop_cols, target, use_centuries=False, exclude_centuries=None):
        df = pd.read_csv(file_path)
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in test file.")

        if use_centuries:
            if "century" not in df.columns:
                raise ValueError("'century' column must be present in test file when using centuries.")

        if exclude_centuries:
            original_len = len(df)
            df = df[~df["century"].astype(int).isin(exclude_centuries)]
            removed = original_len - len(df)
            print(f"Excluded centuries: {exclude_centuries} ({removed} samples removed)", flush=True)

        X = df.drop(columns=[target])
        y = df[target]

        X = X[self.feature_names]
        y_encoded = self.label_encoder.transform(y)

        return X, y_encoded

    def predict_random_baseline(self, size):
        unique_decades = np.unique(self.y_test)
        return np.random.choice(unique_decades, size=size)

    def run(self):
        results = []
        thresholds = sorted(np.unique(self.y_test))

        for threshold in thresholds[1:-1]:  # Skip the first and last thresholds to avoid trivial cases
            decoded_threshold = self.label_encoder.inverse_transform([threshold])[0]
            print(f"\nEvaluating for threshold: {decoded_threshold}", flush=True)
            
            y_test_bin = (self.y_test >= threshold).astype(int)
            
            for model_path in self.models:
                model_name = os.path.splitext(os.path.basename(model_path))[0]
                if model_name == "random":
                    random_decades = self.predict_random_baseline(len(self.X_test))
                    y_pred_bin = (random_decades >= threshold).astype(int)
                    y_score_binary = y_pred_bin.astype(float)
                    print(f"[Random Debug] Threshold {decoded_threshold} - Accuracy: {(y_test_bin == y_pred_bin).mean():.4f}, "
                          f"Positives in test: {y_test_bin.mean():.4f}, Positives in pred: {y_pred_bin.mean():.4f}", flush=True)
                else:
                    try:
                        model, label_encoder, feature_names = joblib.load(model_path)
                    except Exception as e:
                        print(f"Failed to load model from {model_path}: {e}", flush=True)
                        continue

                    y_pred = model.predict(self.X_test)
                    y_pred_bin = (y_pred >= threshold).astype(int)

                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(self.X_test)
                        class_indices = model.classes_
                        threshold_class = class_indices[class_indices >= threshold]
                        if len(threshold_class) == 0 or len(threshold_class) == len(class_indices):
                            y_score_binary = np.zeros(len(self.X_test))
                        else:
                            idx = np.isin(class_indices, threshold_class)
                            y_score_binary = proba[:, idx].sum(axis=1)
                    else:
                        y_score_binary = model.decision_function(self.X_test)

                equal_or_older = (self.y_test <= threshold).mean()

                result = {
                    "model": model_name,
                    "threshold": f"{decoded_threshold}",
                    "accuracy": accuracy_score(y_test_bin, y_pred_bin),
                    "auc_roc": roc_auc_score(y_test_bin, y_score_binary) if len(np.unique(y_test_bin)) > 1 else np.nan,
                    "auprc": average_precision_score(y_test_bin, y_score_binary) if len(np.unique(y_test_bin)) > 1 else np.nan,
                    "f1_macro": f1_score(y_test_bin, y_pred_bin, average="macro", zero_division=0),
                    "f1_weighted": f1_score(y_test_bin, y_pred_bin, average="weighted", zero_division=0),
                    "recall_macro": recall_score(y_test_bin, y_pred_bin, average="macro", zero_division=0),
                    "recall_weighted": recall_score(y_test_bin, y_pred_bin, average="weighted", zero_division=0),
                    "precision_macro": precision_score(y_test_bin, y_pred_bin, average="macro", zero_division=0),
                    "precision_weighted": precision_score(y_test_bin, y_pred_bin, average="weighted", zero_division=0),
                    "%_of_texts": equal_or_older,
                }
                results.append(result)
                
                self.plot_confusion_matrix_for_threshold(model_path, threshold)

                print(f"----- Results for {model_name} at threshold {decoded_threshold} -----", flush=True)
                for k, v in result.items():
                    print(f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}", flush=True)

        os.makedirs(self.output_dir, exist_ok=True)
        pd.DataFrame(results).to_csv(os.path.join(self.output_dir, self.output_file), index=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Binary evaluation of models across thresholds.")
    parser.add_argument("--test_file", type=str, required=True, help="CSV file for test set")
    parser.add_argument("--target", type=str, choices=["decade", "century"], default="decade", help="Target type for thresholding")
    parser.add_argument("--models", nargs="+", required=True, help="Paths to trained model .pkl files")
    parser.add_argument("--drop_cols", nargs="+", required=True, help="Columns to drop from input features")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the results CSV files")
    parser.add_argument("--output_file", type=str, nargs="?", default="binary_results.csv", help="Output file name for results")

    args = parser.parse_args()

    drop_cols = [col.strip() for col in args.drop_cols]

    first_model_path = args.models[0]
    try:
        _, label_encoder, feature_names = joblib.load(first_model_path)
    except Exception as e:
        print(f"Failed to load model from {first_model_path}: {e}", flush=True)
        exit(1)

    tester = BinaryModelTester(
        models=args.models,
        target=args.target,
        drop_cols=drop_cols,
        output_dir=args.output_dir,
        output_file=args.output_file,
        label_encoder=label_encoder,
        feature_names=feature_names
    )

    X_test, y_test = tester.load_data(args.test_file, drop_cols, args.target)

    tester.X_test = X_test
    tester.y_test = y_test

    tester.run()

    