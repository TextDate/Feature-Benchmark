#!/bin/bash
#SBATCH --job-name=Tree_models                # create a short name for your job
#SBATCH --output="Tree_models_decades/tree_models-%j.out"         # %j will be replaced by the slurm jobID
#SBATCH --error="Tree_models_decades/tree_models-%j.err"         # %j will be replaced by the slurm jobID
#SBATCH --nodes=1                         # node count
#SBATCH --ntasks=1                        # total number of tasks across all nodes
#SBATCH --cpus-per-task=32              # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:0                      # number of gpus per node
#SBATCH --mem=200GB                          # Total amount of RAM requested
#SBATCH --partition=cpu                   # The queue where to submit the job

# cd "TextDate/Feature-Benchmark"

export PYTHONPATH=$(pwd)

export OPENBLAS_NUM_THREADS=2

echo "Welcome to the TextDate Feature Benchmarking Script, running Base Models for decades classification."

echo "Running Model Trainer script"
  
echo ""
python Base_model/model_trainer.py \
  --train_file Extracted_features/cleaned_train_features.csv \
  --val_file cleaned_validation_features.csv \
  --target decade \
  --models random_forest xgboost catboost \
  --n_estimators 1000 \
  --drop_cols "file_name,year,century,special_character_ratio" \
  --output_dir Saved_models/decades \

echo "Running Model Tester script for Test Data"

echo ""
python Base_model/model_tester.py \
    --test_file Extracted_features/cleaned_test_features.csv \
    --target decade \
    --models Saved_models/decades/random_forest_model.pkl Saved_models/decades/xgboost_model.pkl Saved_models/decades/catboost_model.pkl \
    --drop_cols "file_name,year,century,special_character_ratio" \
    --output_dir Saved_models/decades \

echo "Running Model Tester script for Gutenberg Data"

echo ""
python Base_model/model_tester.py \
    --test_file Extracted_features/cleaned_gutenberg_features.csv \
    --target decade \
    --models Saved_models/decades/random_forest_model.pkl Saved_models/decades/xgboost_model.pkl Saved_models/decades/catboost_model.pkl \
    --drop_cols "file_name,year,century,special_character_ratio" \
    --output_dir Saved_models/decades \