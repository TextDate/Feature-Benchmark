import argparse
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import seaborn as sns
from tqdm import tqdm


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance


class FeatureImportanceAnalyzer:
    def __init__(self, model_path, test_csv, target_col, drop_cols):
        self.model_path = model_path
        self.test_csv = test_csv
        self.target_col = target_col
        self.drop_cols = drop_cols
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.X = None
        self.y = None
        self._load_model()
        self._load_data()

    def _load_model(self):
        print(f"[INFO] Loading model from {self.model_path}", flush=True)
        loaded = joblib.load(self.model_path)
        if len(loaded) == 3:
            self.model, self.label_encoder, self.feature_names = loaded
        else:
            raise ValueError("Model must contain (model, label_encoder, feature_names)")

    def _load_data(self):
        df = pd.read_csv(self.test_csv)
        df = df.drop(columns=[col for col in self.drop_cols if col in df.columns], errors='ignore')
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found.")

        self.X = df.drop(columns=[self.target_col])[self.feature_names]
        self.y = self.label_encoder.transform(df[self.target_col])

    def plot_tree_feature_importance(self, output_dir):
        print("[INFO] Plotting tree-based feature importances...", flush=True)
        if not hasattr(self.model, "feature_importances_"):
            print("[WARNING] Model does not support feature_importances_", flush=True)
            return

        importances = self.model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        top_features = [self.feature_names[i] for i in sorted_idx]
        top_importances = importances[sorted_idx]

        plt.figure(figsize=(12, 6))
        plt.bar(top_features, top_importances)
        plt.xticks(rotation=45, ha='right')
        plt.title("Tree-Based Feature Importances")
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "tree_feature_importances.png"))
        plt.close()
        print("[INFO] Saved tree-based feature importances.", flush=True)

    def plot_permutation_importance(self, output_dir):
        print("[INFO] Calculating permutation importances...", flush=True)
        result = permutation_importance(self.model, self.X, self.y, n_repeats=10, random_state=42, n_jobs=2)
        sorted_idx = result.importances_mean.argsort()[::-1]
        top_features = [self.feature_names[i] for i in sorted_idx]
        top_scores = result.importances_mean[sorted_idx]

        plt.figure(figsize=(12, 6))
        plt.bar(top_features, top_scores)
        plt.xticks(rotation=45, ha='right')
        plt.title("Permutation Importances")
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "permutation_importances.png"))
        plt.close()
        print("[INFO] Saved permutation importances.", flush=True)

    def plot_pca_loadings(self, output_dir):
        print("[INFO] Performing PCA analysis...", flush=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        pca = PCA(n_components=min(X_scaled.shape[0], X_scaled.shape[1]))
        pca.fit(X_scaled)

        loadings = pd.DataFrame(
            pca.components_.T,
            index=self.feature_names,
            columns=[f"PC{i+1}" for i in range(pca.n_components_)]
        )
        mean_abs_loading = loadings.abs().mean(axis=1).sort_values(ascending=False)

        plt.figure(figsize=(12, 6))
        mean_abs_loading.plot(kind='bar')
        plt.title("PCA-Based Feature Contribution (Mean Absolute Loading)")
        plt.ylabel("Mean |Loading|")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "pca_feature_contributions.png"))
        plt.close()
        print("[INFO] Saved PCA feature contributions.", flush=True)
        
        
    def compute_shap_values_with_progress(self, explainer, X, chunk_size=1000):
        print(f"[INFO] Total samples: {X.shape[0]}", flush=True)
        
        shap_preview = explainer.shap_values(X.iloc[:1])
        if isinstance(shap_preview, list) or (isinstance(shap_preview, np.ndarray) and shap_preview.ndim == 3):
            print("[INFO] Detected multiclass model. Computing SHAP values per class...", flush=True)
            num_classes = len(shap_preview) if isinstance(shap_preview, list) else shap_preview.shape[2]
            shap_values_per_class = [[] for _ in range(num_classes)]

            for i in tqdm(range(0, X.shape[0], chunk_size), desc="Computing SHAP values"):
                X_chunk = X.iloc[i:i + chunk_size]
                shap_chunk = explainer.shap_values(X_chunk)
                if isinstance(shap_chunk, list):
                    for class_idx in range(num_classes):
                        shap_values_per_class[class_idx].append(shap_chunk[class_idx])
                else:
                    for class_idx in range(num_classes):
                        shap_values_per_class[class_idx].append(shap_chunk[:, :, class_idx])

            return [np.vstack(class_chunks) for class_chunks in shap_values_per_class]

        else:
            print("[INFO] Detected single output model. Computing SHAP values...", flush=True)
            shap_values = []
            for i in tqdm(range(0, X.shape[0], chunk_size), desc="Computing SHAP values"):
                X_chunk = X.iloc[i:i + chunk_size]
                shap_values.append(explainer.shap_values(X_chunk))
            return np.vstack(shap_values)



    def plot_shap_summary(self, output_dir):
        print("[INFO] Generating SHAP summary plot...", flush=True)
        os.makedirs(output_dir, exist_ok=True)

        try:
            explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            print(f"[ERROR] Failed to create SHAP explainer: {e}", flush=True)
            return

        try:
            print("[INFO] Calculating SHAP values...", flush=True)
            shap_values = self.compute_shap_values_with_progress(explainer, self.X, chunk_size=1000)

            if isinstance(shap_values, list):
                print("[INFO] Detected multiclass SHAP values. Global summary skipped.", flush=True)
            elif len(shap_values.shape) == 3:
                print("[INFO] Multiclass SHAP values detected. Skipping global summary plot and proceeding to per-class summaries.", flush=True)
            else:
                shap.summary_plot(shap_values, features=self.X, feature_names=self.feature_names, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"))
                plt.close()
                print("[INFO] Saved SHAP summary plot.", flush=True)
            
            print(isinstance(shap_values, list))
            print(hasattr(self.label_encoder, 'classes_'))
            if isinstance(shap_values, list) and hasattr(self.label_encoder, 'classes_'):
                for i, cls in tqdm(enumerate(self.label_encoder.classes_), total=len(self.label_encoder.classes_), desc="SHAP class-wise plots"):
                    sv = shap_values[i]

                    print(f"[DEBUG] SHAP class {cls}: shape {sv.shape}, X shape {self.X.shape}", flush=True)

                    if sv.shape[1] != self.X.shape[1]:
                        print(f"[WARNING] Skipping class {cls}: feature mismatch (SHAP {sv.shape[1]} vs X {self.X.shape[1]})", flush=True)
                        continue

                    try:
                        shap.summary_plot(sv, features=self.X, feature_names=self.feature_names, show=False)
                        plt.title(f"SHAP Summary for class {cls}")
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f"shap_summary_decade_{cls}.png"))
                        plt.close()
                        print(f"[INFO] Saved SHAP summary plot for class {cls}.", flush=True)
                    except Exception as e:
                        print(f"[ERROR] Failed to plot SHAP summary for class {cls}: {e}", flush=True)


        except Exception as e:
            print(f"[ERROR] Failed to generate SHAP plot: {e}", flush=True)



    def plot_correlation_matrix(self, output_dir):
        print("[INFO] Generating correlation matrix...", flush=True)
        df = pd.DataFrame(self.X, columns=self.feature_names)
        corr = df.corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, cmap='coolwarm', annot=False, square=True, fmt=".2f", cbar=True)
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
        plt.close()
        print("[INFO] Saved correlation matrix.", flush=True)


def main(args):
    drop_cols = [col.strip() for group in args.drop_cols for col in group.split(",")]
    analyzer = FeatureImportanceAnalyzer(args.model, args.test_csv, args.target, drop_cols)

    if args.tree:
        analyzer.plot_tree_feature_importance(args.output_dir)
    if args.correlation:
        analyzer.plot_correlation_matrix(args.output_dir)
    if args.pca:
        analyzer.plot_pca_loadings(args.output_dir)
    if args.permutation:
        analyzer.plot_permutation_importance(args.output_dir)
    if args.shap:
        analyzer.plot_shap_summary(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and plot feature importances for a trained model")
    parser.add_argument("--model", required=True, help="Path to .pkl file (model, label_encoder, feature_names)")
    parser.add_argument("--test_csv", required=True, help="CSV with features and target")
    parser.add_argument("--target", required=True, help="Target column name (e.g., 'decade')")
    parser.add_argument("--drop_cols", nargs="+", required=True, help="Comma-separated columns to drop")
    parser.add_argument("--output_dir", default="feature_importance_plots", help="Where to save plots")
    parser.add_argument("--tree", action="store_true", help="Plot tree-based feature importances")
    parser.add_argument("--permutation", action="store_true", help="Plot permutation importances")
    parser.add_argument("--pca", action="store_true", help="Plot PCA-based loadings")
    parser.add_argument("--shap", action="store_true", help="Plot SHAP summary plot")
    parser.add_argument("--correlation", action="store_true", help="Plot correlation matrix")
    args = parser.parse_args()
    main(args)