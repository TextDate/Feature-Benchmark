import argparse
import os
import time
import joblib
import psutil
import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    root_mean_squared_error, mean_absolute_error, r2_score,
    precision_score, recall_score
)
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


MODELS = {
    "random_forest": RandomForestClassifier(random_state=42, max_depth=None, class_weight="balanced", verbose=0, n_jobs=-1),
    "gradient_boosting": HistGradientBoostingClassifier(random_state=42, verbose=0, class_weight="balanced"),
    "xgboost": XGBClassifier(random_state=42, eval_metric='mlogloss', verbosity=0, n_jobs=-1),
    "lightgbm": LGBMClassifier(random_state=42, verbose=0, n_jobs=-1, class_weight="balanced"),
    "catboost": CatBoostClassifier(random_state=42, verbose=0, thread_count=-1)
    
}


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


    df_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns], errors='ignore')
    df_val = df_val.drop(columns=[c for c in drop_cols if c in df_val.columns], errors='ignore')

    common_features = df_train.columns.intersection(df_val.columns).tolist()
    common_features.remove(target_col)

    X_train, y_train = df_train[common_features], df_train[[target_col]]
    X_val, y_val = df_val[common_features], df_val[[target_col]]

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
        "F1 Macro": f1_score(y_val_decoded, y_pred_decoded, average="macro"),
        "F1 Weighted": f1_score(y_val_decoded, y_pred_decoded, average="weighted"),
        "Recall Macro": recall_score(y_val_decoded, y_pred_decoded, average="macro"),
        "Recall Weighted": recall_score(y_val_decoded, y_pred_decoded, average="weighted"),
        "Precision Macro": precision_score(y_val_decoded, y_pred_decoded, average="macro"),
        "Precision Weighted": precision_score(y_val_decoded, y_pred_decoded, average="weighted"),
        "RMSE": root_mean_squared_error(y_val_decoded, y_pred_decoded),
        "MAE": mean_absolute_error(y_val_decoded, y_pred_decoded),
        "R2": r2_score(y_val_decoded, y_pred_decoded)
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
        params = base_model.get_params()
        params.pop("n_estimators", None)

        if model_name == "gradient_boosting":
            model = HistGradientBoostingClassifier(max_iter=args.n_estimators, random_state=42, verbose=0)
        else:
            model = base_model.__class__(**params, n_estimators=args.n_estimators)

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
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--models", type=str, nargs='+', required=True, choices=MODELS.keys())
    parser.add_argument("--n_estimators", type=int, required=True)
    parser.add_argument("--drop_cols", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_smote", action="store_true")
    parser.add_argument("--use_tomek", action="store_true")
    parser.add_argument("--exclude_decades", type=int, nargs='*', help="Decades to exclude, e.g. 1600 1610 1620")
    parser.add_argument("--use_centuries", action="store_true")
    parser.add_argument("--exclude_centuries", type=int, nargs='*', help="Centuries to exclude (e.g. 17 18)")
    args = parser.parse_args()

    train_and_save_models(args)
