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

        # Remove deprecated features from both dataset and model feature list
        deprecated_features = [
            'Special_Character_Ratio',
            'max_anachronism_gap_years',
            'anachronism_word_count',
            'anachronism_density'
        ]
        df = df.drop(columns=[col for col in deprecated_features if col in df.columns], errors='ignore')

        # Create mapping of which features to keep (track original indices)
        original_feature_names = self.feature_names.copy()
        self.feature_indices_to_keep = [i for i, f in enumerate(original_feature_names) if f not in deprecated_features]
        self.feature_names = [f for f in original_feature_names if f not in deprecated_features]

        if deprecated_features:
            print(f"[INFO] Excluded deprecated features: {deprecated_features}", flush=True)
            print(f"[INFO] Using {len(self.feature_names)} features for analysis (was {len(original_feature_names)})", flush=True)

        self.X = df.drop(columns=[self.target_col])[self.feature_names]
        self.y = self.label_encoder.transform(df[self.target_col])

    def plot_tree_feature_importance(self, output_dir):
        print("[INFO] Plotting tree-based feature importances...", flush=True)
        if not hasattr(self.model, "feature_importances_"):
            print("[WARNING] Model does not support feature_importances_", flush=True)
            return

        # Filter importances to match filtered features
        all_importances = self.model.feature_importances_
        importances = all_importances[self.feature_indices_to_keep] if hasattr(self, 'feature_indices_to_keep') else all_importances
        sorted_idx = np.argsort(importances)[::-1]

        tree_dir = os.path.join(output_dir, "tree_importance")
        os.makedirs(tree_dir, exist_ok=True)

        # Save all data as CSV
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances,
            'rank': np.argsort(np.argsort(importances)[::-1]) + 1
        }).sort_values('importance', ascending=False)
        importance_df.to_csv(os.path.join(tree_dir, "tree_feature_importances.csv"), index=False)

        # Show top 20 features for plot
        max_display = min(20, len(self.feature_names))
        top_idx = sorted_idx[:max_display]
        top_features = [self.feature_names[i] for i in top_idx]
        top_importances = importances[top_idx]

        plt.style.use('seaborn-v0_8')
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(top_features)), top_importances, alpha=0.8, color='#FF6B6B')
        plt.xticks(range(len(top_features)), top_features, rotation=45, ha='right')
        plt.title("Tree-Based Feature Importances (Top 20)", fontsize=14, fontweight='bold')
        plt.xlabel("Features", fontsize=12, fontweight='bold')
        plt.ylabel("Importance", fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, top_importances)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{importance:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(tree_dir, "tree_feature_importances.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("[INFO] Saved tree-based feature importances (both CSV and plot).", flush=True)

    def plot_permutation_importance(self, output_dir):
        print("[INFO] Calculating permutation importances...", flush=True)
        result = permutation_importance(self.model, self.X, self.y, n_repeats=10, random_state=42, n_jobs=-1)
        sorted_idx = result.importances_mean.argsort()[::-1]

        perm_dir = os.path.join(output_dir, "permutation_importance")
        os.makedirs(perm_dir, exist_ok=True)

        # Save all data as CSV
        perm_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std,
            'rank': np.argsort(np.argsort(result.importances_mean)[::-1]) + 1
        }).sort_values('importance_mean', ascending=False)
        perm_df.to_csv(os.path.join(perm_dir, "permutation_importances.csv"), index=False)

        # Show top 20 features for plot
        max_display = min(20, len(self.feature_names))
        top_idx = sorted_idx[:max_display]
        top_features = [self.feature_names[i] for i in top_idx]
        top_scores = result.importances_mean[top_idx]
        top_stds = result.importances_std[top_idx]

        plt.style.use('seaborn-v0_8')
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(top_features)), top_scores, yerr=top_stds,
                      alpha=0.8, color='#4ECDC4', capsize=5, error_kw={'elinewidth': 1})
        plt.xticks(range(len(top_features)), top_features, rotation=45, ha='right')
        plt.title("Permutation Importances with Standard Deviation (Top 20)", fontsize=14, fontweight='bold')
        plt.xlabel("Features", fontsize=12, fontweight='bold')
        plt.ylabel("Importance Score", fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (bar, score, std) in enumerate(zip(bars, top_scores, top_stds)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.001,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(perm_dir, "permutation_importances.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("[INFO] Saved permutation importances (both CSV and plot).", flush=True)

    def plot_pca_loadings(self, output_dir):
        print("[INFO] Performing PCA analysis...", flush=True)

        # Handle NaN values by dropping them
        X_clean = self.X.dropna()
        if len(X_clean) == 0:
            print("[WARNING] No data remaining after removing NaN values. Skipping PCA analysis.", flush=True)
            return

        print(f"[INFO] Using {len(X_clean)} samples after removing NaN values (originally {len(self.X)})", flush=True)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        pca = PCA(n_components=min(X_scaled.shape[0], X_scaled.shape[1]))
        pca.fit(X_scaled)

        loadings = pd.DataFrame(
            pca.components_.T,
            index=self.feature_names,
            columns=[f"PC{i+1}" for i in range(pca.n_components_)]
        )
        mean_abs_loading = loadings.abs().mean(axis=1).sort_values(ascending=False)

        pca_dir = os.path.join(output_dir, "pca_analysis")
        os.makedirs(pca_dir, exist_ok=True)

        # Save PCA data as CSV
        loadings.to_csv(os.path.join(pca_dir, "pca_loadings.csv"))

        # Save explained variance data
        variance_df = pd.DataFrame({
            'component': [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
        })
        variance_df.to_csv(os.path.join(pca_dir, "explained_variance.csv"), index=False)

        # Save mean absolute loadings
        mean_loading_df = pd.DataFrame({
            'feature': mean_abs_loading.index,
            'mean_abs_loading': mean_abs_loading.values,
            'rank': range(1, len(mean_abs_loading) + 1)
        })
        mean_loading_df.to_csv(os.path.join(pca_dir, "feature_contributions.csv"), index=False)

        # Plot 1: Mean absolute loadings (top 20)
        plt.style.use('seaborn-v0_8')
        plt.figure(figsize=(12, 8))
        top_20_loadings = mean_abs_loading.head(20)
        bars = plt.bar(range(len(top_20_loadings)), top_20_loadings.values, alpha=0.8, color='#95A5A6')
        plt.xticks(range(len(top_20_loadings)), top_20_loadings.index, rotation=45, ha='right')
        plt.title("PCA-Based Feature Contribution (Mean Absolute Loading - Top 20)", fontsize=14, fontweight='bold')
        plt.xlabel("Features", fontsize=12, fontweight='bold')
        plt.ylabel("Mean |Loading|", fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (bar, loading) in enumerate(zip(bars, top_20_loadings.values)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{loading:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(pca_dir, "pca_feature_contributions.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 2: Explained variance ratio
        plt.figure(figsize=(12, 6))
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)

        # Fix dimension mismatch: ensure x and y have same length
        n_components_to_plot = min(20, len(cumsum_var))
        x_range = range(1, n_components_to_plot + 1)
        y_values = cumsum_var[:n_components_to_plot]

        plt.plot(x_range, y_values, 'bo-', linewidth=2, markersize=6)
        plt.title("PCA Explained Variance Ratio (First 20 Components)", fontsize=14, fontweight='bold')
        plt.xlabel("Principal Component", fontsize=12, fontweight='bold')
        plt.ylabel("Cumulative Explained Variance", fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% Variance')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(pca_dir, "pca_explained_variance.png"), dpi=300, bbox_inches='tight')
        plt.close()

        if len(cumsum_var) >= 10:
            print(f"[INFO] Saved PCA analysis (both CSV and plots). Total variance explained by first 10 PCs: {cumsum_var[9]:.3f}", flush=True)
        else:
            print(f"[INFO] Saved PCA analysis (both CSV and plots). Total variance explained by all {len(cumsum_var)} PCs: {cumsum_var[-1]:.3f}", flush=True)
        
        
    def compute_shap_values_with_progress(self, explainer, X, chunk_size=1000):
        print(f"[INFO] Total samples: {X.shape[0]}", flush=True)

        # Use larger chunks for better performance
        shap_preview = explainer.shap_values(X.iloc[:1])
        if isinstance(shap_preview, list) or (isinstance(shap_preview, np.ndarray) and shap_preview.ndim == 3):
            print("[INFO] Detected multiclass model. Computing SHAP values per class...", flush=True)
            num_classes = len(shap_preview) if isinstance(shap_preview, list) else shap_preview.shape[2]
            shap_values_per_class = [[] for _ in range(num_classes)]

            # Process in larger chunks for better efficiency
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
        shap_dir = os.path.join(output_dir, "shap_analysis")
        os.makedirs(shap_dir, exist_ok=True)

        # Create more readable feature names
        readable_feature_names = []
        for name in self.feature_names:
            if name.startswith('Compression_Ratio'):
                readable_name = name.replace('_', ' ').replace('Order', 'Order').replace('Markov', 'Markov')
            elif name.startswith('NRC'):
                readable_name = name.replace('_', ' ')
            elif name.startswith('Entropy'):
                readable_name = name.replace('_', ' ')
            elif name == 'Avg_Word_Length':
                readable_name = 'Avg Word Length'
            elif name == 'Avg_Sentence_Length':
                readable_name = 'Avg Sentence Length'
            elif name == 'Lexical_Richness':
                readable_name = 'Lexical Richness'
            elif name == 'Punctuation_Density':
                readable_name = 'Punctuation Density'
            elif name == 'Syllable_Per_Word':
                readable_name = 'Syllables Per Word'
            elif name == 'Digit_Ratio':
                readable_name = 'Digit Ratio'
            elif name == 'Flesch_Readability':
                readable_name = 'Flesch Readability'
            elif name == 'Stopword_Ratio':
                readable_name = 'Stopword Ratio'
            elif name == 'Shannon_Entropy':
                readable_name = 'Shannon Entropy'
            elif len(name) <= 4:  # Short words like 'at', 'and', 'by', etc.
                readable_name = f"word: '{name}'"
            else:
                readable_name = name.replace('_', ' ')
            readable_feature_names.append(readable_name)

        try:
            # Print model info for verification
            model_type = type(self.model).__name__
            print(f"[INFO] Model type: {model_type}", flush=True)

            # Verify model is working by testing a prediction
            test_pred = self.model.predict(self.X.iloc[:1])
            print(f"[INFO] Model test prediction: {test_pred[0]} (confirming model is loaded correctly)", flush=True)

            # Improved tree model detection
            is_tree_model = (
                hasattr(self.model, 'tree_') or
                hasattr(self.model, 'estimators_') or
                'RandomForest' in model_type or
                'GradientBoosting' in model_type or
                'XGB' in model_type or
                'LightGBM' in model_type or
                'CatBoost' in model_type
            )

            if is_tree_model:
                explainer = shap.TreeExplainer(self.model)
                print(f"[INFO] Using TreeExplainer for {model_type} (optimized for tree models)", flush=True)
            else:
                # Use much smaller background for KernelExplainer speed
                background_size = min(25, len(self.X))  # Drastically reduce for speed
                background = shap.sample(self.X, background_size)

                # Try to use predict instead of predict_proba if it's too slow
                try:
                    explainer = shap.KernelExplainer(self.model.predict, background)
                    print(f"[INFO] Using KernelExplainer for {model_type} with predict() and {background_size} background samples", flush=True)
                except:
                    explainer = shap.KernelExplainer(self.model.predict_proba, background)
                    print(f"[INFO] Using KernelExplainer for {model_type} with predict_proba() and {background_size} background samples", flush=True)
        except Exception as e:
            print(f"[ERROR] Failed to create SHAP explainer: {e}", flush=True)
            return

        try:
            # Use all data as requested
            X_sample = self.X
            print(f"[INFO] Processing all {len(self.X)} samples for SHAP analysis", flush=True)

            print(f"[INFO] Calculating SHAP values for {len(X_sample)} samples...", flush=True)
            # Use different chunk sizes based on explainer type
            if is_tree_model:
                chunk_size = 2000  # Tree models can handle larger chunks efficiently
            else:
                chunk_size = 50    # Very small chunks for KernelExplainer due to extreme slowness

            shap_values = self.compute_shap_values_with_progress(explainer, X_sample, chunk_size=chunk_size)

            if isinstance(shap_values, list):
                print("[INFO] Detected multiclass SHAP values. Global summary skipped.", flush=True)
                # Save multiclass SHAP values as CSV
                for i, sv in enumerate(shap_values):
                    class_name = self.label_encoder.classes_[i] if hasattr(self.label_encoder, 'classes_') else f"class_{i}"
                    shap_df = pd.DataFrame(sv, columns=self.feature_names)
                    shap_df.to_csv(os.path.join(shap_dir, f"shap_values_class_{class_name}.csv"), index=False)

                    # Calculate mean absolute SHAP values for this class
                    mean_shap = pd.DataFrame({
                        'feature': self.feature_names,
                        'mean_abs_shap': np.abs(sv).mean(axis=0),
                        'mean_shap': sv.mean(axis=0),
                        'rank': np.argsort(np.abs(sv).mean(axis=0))[::-1] + 1
                    }).sort_values('mean_abs_shap', ascending=False)
                    mean_shap.to_csv(os.path.join(shap_dir, f"shap_importance_class_{class_name}.csv"), index=False)

            elif len(shap_values.shape) == 3:
                print("[INFO] Multiclass SHAP values detected. Skipping global summary plot and proceeding to per-class summaries.", flush=True)
                # Save 3D SHAP values as CSV
                for i in range(shap_values.shape[2]):
                    class_name = self.label_encoder.classes_[i] if hasattr(self.label_encoder, 'classes_') else f"class_{i}"
                    shap_df = pd.DataFrame(shap_values[:, :, i], columns=self.feature_names)
                    shap_df.to_csv(os.path.join(shap_dir, f"shap_values_class_{class_name}.csv"), index=False)

                    # Calculate mean absolute SHAP values for this class
                    mean_shap = pd.DataFrame({
                        'feature': self.feature_names,
                        'mean_abs_shap': np.abs(shap_values[:, :, i]).mean(axis=0),
                        'mean_shap': shap_values[:, :, i].mean(axis=0),
                        'rank': np.argsort(np.abs(shap_values[:, :, i]).mean(axis=0))[::-1] + 1
                    }).sort_values('mean_abs_shap', ascending=False)
                    mean_shap.to_csv(os.path.join(shap_dir, f"shap_importance_class_{class_name}.csv"), index=False)

            else:
                # Save single output SHAP values as CSV
                shap_df = pd.DataFrame(shap_values, columns=self.feature_names)
                shap_df.to_csv(os.path.join(shap_dir, "shap_values.csv"), index=False)

                # Calculate mean absolute SHAP values
                mean_shap = pd.DataFrame({
                    'feature': self.feature_names,
                    'mean_abs_shap': np.abs(shap_values).mean(axis=0),
                    'mean_shap': shap_values.mean(axis=0),
                    'rank': np.argsort(np.abs(shap_values).mean(axis=0))[::-1] + 1
                }).sort_values('mean_abs_shap', ascending=False)
                mean_shap.to_csv(os.path.join(shap_dir, "shap_importance.csv"), index=False)

                # Summary plot (beeswarm)
                plt.style.use('seaborn-v0_8')
                plt.figure(figsize=(14, 10))
                shap.summary_plot(shap_values, features=X_sample, feature_names=readable_feature_names,
                                max_display=len(self.feature_names), show=False)
                plt.title("SHAP Summary Plot (All Features)", fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(shap_dir, "shap_summary_plot.png"), dpi=300, bbox_inches='tight')
                plt.close()

                # Bar plot (feature importance) - show all features
                plt.figure(figsize=(12, 10))
                shap.summary_plot(shap_values, features=X_sample, feature_names=readable_feature_names,
                                plot_type="bar", max_display=len(self.feature_names), show=False)
                plt.title("SHAP Feature Importance (All Features)", fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(shap_dir, "shap_importance_bar.png"), dpi=300, bbox_inches='tight')
                plt.close()
                print(f"[INFO] Saved SHAP summary and importance plots (both CSV and PNG) with {len(self.feature_names)} features.", flush=True)
            
            if isinstance(shap_values, list) and hasattr(self.label_encoder, 'classes_'):
                for i, cls in tqdm(enumerate(self.label_encoder.classes_), total=len(self.label_encoder.classes_), desc="SHAP class-wise plots"):
                    sv = shap_values[i]

                    # Convert class number to century name
                    century_name = f"{cls}th Century" if cls != 21 else "21st Century"

                    print(f"[DEBUG] SHAP class {cls} ({century_name}): shape {sv.shape}, X_sample shape {X_sample.shape}", flush=True)

                    if sv.shape[1] != X_sample.shape[1]:
                        print(f"[WARNING] Skipping class {cls}: feature mismatch (SHAP {sv.shape[1]} vs X_sample {X_sample.shape[1]})", flush=True)
                        continue

                    try:
                        # Summary plot (beeswarm) for class
                        plt.style.use('seaborn-v0_8')
                        plt.figure(figsize=(14, 10))
                        shap.summary_plot(sv, features=X_sample, feature_names=readable_feature_names,
                                        max_display=len(self.feature_names), show=False)
                        plt.title(f"SHAP Summary for {century_name}", fontsize=14, fontweight='bold')
                        plt.tight_layout()
                        plt.savefig(os.path.join(shap_dir, f"shap_summary_century_{cls}.png"), dpi=300, bbox_inches='tight')
                        plt.close()

                        # Bar plot (importance) for class
                        plt.figure(figsize=(12, 10))
                        shap.summary_plot(sv, features=X_sample, feature_names=readable_feature_names,
                                        plot_type="bar", max_display=len(self.feature_names), show=False)
                        plt.title(f"SHAP Feature Importance for {century_name}", fontsize=14, fontweight='bold')
                        plt.tight_layout()
                        plt.savefig(os.path.join(shap_dir, f"shap_importance_century_{cls}.png"), dpi=300, bbox_inches='tight')
                        plt.close()

                        print(f"[INFO] Saved SHAP plots for {century_name}.", flush=True)
                    except Exception as e:
                        print(f"[ERROR] Failed to plot SHAP for {century_name}: {e}", flush=True)


        except Exception as e:
            print(f"[ERROR] Failed to generate SHAP plot: {e}", flush=True)



    def plot_correlation_matrix(self, output_dir):
        print("[INFO] Generating correlation matrix...", flush=True)
        df = pd.DataFrame(self.X, columns=self.feature_names)
        corr = df.corr()

        corr_dir = os.path.join(output_dir, "correlation_analysis")
        os.makedirs(corr_dir, exist_ok=True)

        # Save correlation data as CSV
        print("[INFO] Saving correlation matrix as CSV for analysis...", flush=True)
        corr.to_csv(os.path.join(corr_dir, "correlation_matrix.csv"))

        # Generate heatmap plot for visual use
        plt.style.use('seaborn-v0_8')
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr, dtype=bool))  # Show only lower triangle
        sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, annot=False,
                   square=True, fmt=".2f", cbar=True, cbar_kws={"shrink": 0.8})
        plt.title("Feature Correlation Matrix (Lower Triangle)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(corr_dir, "correlation_matrix.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 2: Top absolute correlations
        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                corr_pairs.append((corr.columns[i], corr.columns[j], abs(corr.iloc[i, j]), corr.iloc[i, j]))

        corr_pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs = corr_pairs[:20]

        # Save correlation pairs as CSV
        if top_pairs:
            corr_pairs_df = pd.DataFrame([
                {
                    'feature_1': pair[0],
                    'feature_2': pair[1],
                    'correlation': pair[3],
                    'abs_correlation': pair[2],
                    'rank': i + 1
                }
                for i, pair in enumerate(top_pairs)
            ])
            corr_pairs_df.to_csv(os.path.join(corr_dir, "top_correlations.csv"), index=False)

            features = [f"{pair[0]} - {pair[1]}" for pair in top_pairs]
            correlations = [pair[3] for pair in top_pairs]
            abs_correlations = [pair[2] for pair in top_pairs]

            plt.figure(figsize=(12, 8))
            colors = ['#FF6B6B' if corr > 0 else '#4ECDC4' for corr in correlations]
            bars = plt.barh(range(len(features)), correlations, color=colors, alpha=0.8)
            plt.yticks(range(len(features)), features)
            plt.xlabel("Correlation Coefficient", fontsize=12, fontweight='bold')
            plt.title("Top 20 Feature Correlations", fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='x')

            # Add value labels
            for i, (bar, corr) in enumerate(zip(bars, correlations)):
                plt.text(corr + 0.01 if corr > 0 else corr - 0.01, bar.get_y() + bar.get_height()/2,
                        f'{corr:.3f}', ha='left' if corr > 0 else 'right', va='center', fontsize=8)

            plt.tight_layout()
            plt.savefig(os.path.join(corr_dir, "top_correlations.png"), dpi=300, bbox_inches='tight')
            plt.close()

            print(f"[INFO] Saved correlation analysis (both CSV and plots). Highest correlation: {top_pairs[0][2]:.3f}", flush=True)
        else:
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