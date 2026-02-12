# Feature-Benchmark

Benchmarking pipeline for text age prediction using tree-based models (CatBoost, Random Forest, XGBoost, SVM, KNN, Gaussian NB), with feature extraction, evaluation on selectable datasets, and support for SHAP, PCA, and feature importance analysis.

## Setup

```bash
# Activate virtual environment
source ../../virtual-venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
Feature-Benchmark/
├── Feature_extraction/       # Text feature extraction pipeline
│   ├── feature_extractor.py  # Main TextFeatureExtractor class
│   ├── clean_csv.py          # Data cleaning utilities
│   ├── target_words.json     # Words for distance feature calculation
│   └── run_feature_extraction.sh
│
├── Base_model/               # Multi-class classification models
│   ├── model_trainer.py      # Training (single + multi-seed)
│   └── model_tester.py       # Testing (single + multi-seed + aggregation)
│
├── Binary_model/             # Binary classification (old vs new texts)
│   └── binary_model_tester.py
│
├── Plotting/                 # Visualization and analysis
│   ├── feature_plotter.py              # Feature distribution plots + combination
│   ├── feature_importance_analyzer.py  # SHAP, PCA, importance + comparative plots
│   ├── model_comparison_plotter.py     # Model performance comparison
│   └── plot_binary_model_results.py    # Binary model result visualization
│
├── Extracted_features/       # Processed feature CSV files (gitignored)
├── Saved_models/             # Trained model files (gitignored)
├── Saved_models_binary/      # Binary classification models (gitignored)
├── Results/                  # All analysis outputs (gitignored)
│   ├── feature_analysis/     # Feature distribution analysis
│   ├── feature_importance/   # SHAP and importance analysis
│   ├── model_analysis/       # Model comparison results
│   ├── saved_models_results/ # Training/testing metrics
│   ├── comparison_plots/     # Cross-model comparisons
│   ├── comparative_2x3_plots/# Comparative feature analysis
│   └── multi_seed/           # Multi-seed experiment results
│
├── run_all_feature_models.sh      # Master training/testing pipeline
├── run_multi_seed_experiments.sh  # Multi-seed experiments
├── run_feature_analysis.sh        # Feature distribution analysis
├── run_feature_importance.sh      # Feature importance + comparative
├── run_model_analysis.sh          # Model comparison plots
└── requirements.txt
```

## Workflows

### 1. Feature Extraction

Extract linguistic, compression, and structural features from raw text data.

```bash
sbatch Feature_extraction/run_feature_extraction.sh
```

### 2. Model Training and Testing

Train all models across all feature subsets (compression, lexical_structure, readability, distance, neologism, final_model, optimal) for both decades and centuries classification.

```bash
# Train all feature subsets with all models
sbatch run_all_feature_models.sh

# Train only a specific feature subset
sbatch run_all_feature_models.sh -f compression
sbatch run_all_feature_models.sh -f optimal
```

The script trains models and then runs both base and binary model testing automatically.

#### Python CLI (direct usage)

```bash
# Train models
python Base_model/model_trainer.py train \
    --train_file Extracted_features/train_features_cleaned_*.csv \
    --val_file Extracted_features/valid_features_cleaned_*.csv \
    --target decade \
    --models catboost xgboost random_forest \
    --n_estimators 1000 \
    --drop_cols "file_name,year,century" \
    --output_dir Saved_models/compression/decades

# Test models
python Base_model/model_tester.py test \
    --test_file Extracted_features/test_features_cleaned_*.csv \
    --target decade \
    --models Saved_models/compression/decades/catboost_model.pkl \
    --drop_cols "file_name,year,century" \
    --output_dir Results/saved_models_results/compression/decades
```

### 3. Multi-Seed Experiments

Train and test models with multiple random seeds for statistical robustness.

```bash
sbatch run_multi_seed_experiments.sh
```

#### Python CLI (direct usage)

```bash
# Multi-seed training
python Base_model/model_trainer.py multi-seed \
    --train_file Extracted_features/train_features_cleaned_*.csv \
    --val_file Extracted_features/valid_features_cleaned_*.csv \
    --feature_subset compression \
    --model catboost \
    --time_scale centuries \
    --target_col century \
    --n_seeds 10 \
    --use_centuries \
    --drop_cols file_name year decade

# Multi-seed testing
python Base_model/model_tester.py multi-seed \
    --test_file Extracted_features/test_features_cleaned_*.csv \
    --feature_subset compression \
    --model catboost \
    --time_scale centuries \
    --target_col century \
    --dataset_name test \
    --use_centuries \
    --drop_cols file_name year decade

# Aggregate results across all seeds
python Base_model/model_tester.py aggregate --base_dir Results/multi_seed
```

### 4. Feature Distribution Analysis

Generate per-feature distribution plots (line, box, violin) for each dataset, then combine them into comparison grids.

```bash
# Run both individual and combined plots
sbatch run_feature_analysis.sh --target decade --mode both

# Individual plots only
sbatch run_feature_analysis.sh --target century --mode individual
```

#### Python CLI (direct usage)

```bash
# Generate individual plots
python Plotting/feature_plotter.py plot \
    --csvs train.csv val.csv test.csv \
    --labels train val test \
    --target decade \
    --output_dir Results/feature_analysis/individual

# Combine into comparison grids
python Plotting/feature_plotter.py combine \
    --train_dir Results/feature_analysis/individual/train \
    --val_dir Results/feature_analysis/individual/validation \
    --test_dir Results/feature_analysis/individual/test \
    --gutenberg_dir Results/feature_analysis/individual/gutenberg \
    --output_dir Results/feature_analysis/combined
```

### 5. Feature Importance Analysis

Run SHAP, PCA, permutation importance, tree-based importance, and correlation analysis. Optionally includes comparative 2x3 plots.

```bash
# Standard feature importance analysis
sbatch run_feature_importance.sh

# With comparative 2x3 plots
sbatch run_feature_importance.sh --comparative

# Disable SHAP (faster)
sbatch run_feature_importance.sh --no-shap
```

#### Python CLI (direct usage)

```bash
# Single-model analysis
python Plotting/feature_importance_analyzer.py analyze \
    --model Saved_models/compression/centuries/catboost_model.pkl \
    --test_csv Extracted_features/test_features_cleaned_*.csv \
    --target century \
    --drop_cols "text,file_name,id" \
    --output_dir Results/feature_importance/compression_centuries \
    --tree --permutation --pca --shap --correlation --combined_1x3

# Comparative 2x3 plots (decades vs centuries per feature subset)
python Plotting/feature_importance_analyzer.py comparative \
    --model_dir Saved_models \
    --test_csv Extracted_features/test_features_cleaned_*.csv \
    --drop_cols "text,file_name,id" \
    --output_dir Results/comparative_2x3_plots
```

### 6. Model Comparison Plots

Generate cross-model performance comparisons (heatmaps, bar charts, dataset comparisons).

```bash
sbatch run_model_analysis.sh --mode all
```

#### Python CLI (direct usage)

```bash
python Plotting/model_comparison_plotter.py \
    --base_dir . \
    --output_dir Results/model_analysis
```

## Feature Subsets

| Subset | Features | Description |
|--------|----------|-------------|
| `compression` | 7 | Markov compression ratios, NRC, Shannon entropy |
| `lexical_structure` | 6 | Word length, lexical richness, sentence structure |
| `readability` | 2 | Flesch readability, stopword ratio |
| `distance` | 17 | Word spacing patterns (at, and, by, the, ...) |
| `neologism` | 11 | Temporal vocabulary indicators |
| `final_model` | All | All features combined |
| `optimal` | 9 | Best-performing feature selection |

## Supported Models

- **CatBoost** - Gradient boosting with categorical feature support
- **XGBoost** - Extreme gradient boosting
- **Random Forest** - Ensemble of decision trees
- **LightGBM** - Light gradient boosting
- **SVM** - Support vector machine (with probability)
- **Gaussian NB** - Gaussian naive bayes
- **KNN** - K-nearest neighbors

## SLURM Environment

All shell scripts are SLURM batch scripts. They can be submitted with `sbatch` on HPC clusters or run directly with `bash` for local execution.

```bash
# HPC submission
sbatch run_all_feature_models.sh

# Local execution
bash run_all_feature_models.sh
```

Required environment variables (set automatically by SLURM scripts):

```bash
export PYTHONPATH=$(pwd)
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
```
