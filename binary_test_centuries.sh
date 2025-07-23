#!/bin/bash
#SBATCH --job-name=Tree_models                # create a short name for your job
#SBATCH --output="Tree_models_centuries/tree_models-%j.out"         # %j will be replaced by the slurm jobID
#SBATCH --error="Tree_models_centuries/tree_models-%j.err"         # %j will be replaced by the slurm jobID
#SBATCH --nodes=1                         # node count
#SBATCH --ntasks=1                        # total number of tasks across all nodes
#SBATCH --cpus-per-task=32              # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:0                      # number of gpus per node
#SBATCH --mem=200GB                          # Total amount of RAM requested
#SBATCH --partition=cpu                   # The queue where to submit the job

# cd "TextDate/Feature-Benchmark"

export PYTHONPATH=$(pwd)

export OPENBLAS_NUM_THREADS=2

echo "Welcome to the TextDate Feature Benchmarking Script, running Binary Models for centuries classification."


echo "Running Model Trainer script for centuries"

echo ""
python binary_model_tester.py \
    --test_file Extracted_features/cleaned_validation_features.cs.csv \
    --target century \
    --models Saved_models/centuries/random_forest_model.pkl Saved_models/centuries/xgboost_model.pkl Saved_models/centuries/catboost_model.pkl random\
    --drop_cols "file_name,year,century,special_character_ratio" \
    --output_file binary_results_validation.csv \
    --output_dir Saved_models_binary/centuries \

echo ""
python binary_model_tester.py \
    --test_file Extracted_features/cleaned_test_features.csv \
    --target century \
    --models Saved_models/centuries/random_forest_model.pkl Saved_models/centuries/xgboost_model.pkl Saved_models/centuries/catboost_model.pkl random\
    --drop_cols "file_name,year,century,special_character_ratio" \
    --output_file binary_results_test.csv \
    --output_dir Saved_models_binary/centuries \

echo ""
python binary_model_tester.py \
    --test_file Extracted_features/cleaned_gutenberg_features.csv \
    --target century \
    --models Saved_models/centuries/random_forest_model.pkl Saved_models/centuries/xgboost_model.pkl Saved_models/centuries/catboost_model.pkl random\
    --drop_cols "file_name,year,century,special_character_ratio" \
    --output_file binary_results_gutenberg.csv \
    --output_dir Saved_models_binary/centuries \