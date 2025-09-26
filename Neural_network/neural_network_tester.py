import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    root_mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)


def load_test_data(test_file, target_col, drop_cols, label_encoder, feature_names, use_centuries=False):
    """Load and prepare test data for evaluation"""
    df = pd.read_csv(test_file)

    # Select target column
    y_test = df["century"] if use_centuries else df[target_col]

    # Drop specified columns and target columns
    columns_to_drop = [target_col, "century", "decade"] + drop_cols
    X_test = df.drop(columns_to_drop, axis=1, errors='ignore')

    # Ensure features match training data
    # Add missing columns with zeros
    for col in feature_names:
        if col not in X_test.columns:
            X_test[col] = 0

    # Keep only the features used during training
    X_test = X_test[feature_names]

    # Encode target labels
    try:
        y_test = label_encoder.transform(y_test.values.ravel())
    except ValueError as e:
        print(f"Warning: Some labels in test data not seen during training: {e}")
        # Filter out unseen labels
        known_labels = set(label_encoder.classes_)
        mask = y_test.isin(known_labels)
        X_test = X_test[mask]
        y_test = y_test[mask]
        y_test = label_encoder.transform(y_test.values.ravel())

    return X_test, y_test


def evaluate_model(model, X_test, y_test, label_encoder, device):
    """Evaluate model on test data and return metrics"""
    model.eval()

    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
        logits = model(X_test_tensor)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()

    # Decode predictions and true labels
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    y_test_decoded = label_encoder.inverse_transform(y_test)

    # Calculate metrics
    results = {
        "Accuracy": accuracy_score(y_test_decoded, y_pred_decoded),
        "F1 Macro": f1_score(y_test_decoded, y_pred_decoded, average="macro", zero_division=0),
        "F1 Weighted": f1_score(y_test_decoded, y_pred_decoded, average="weighted", zero_division=0),
        "Recall Macro": recall_score(y_test_decoded, y_pred_decoded, average="macro", zero_division=0),
        "Recall Weighted": recall_score(y_test_decoded, y_pred_decoded, average="weighted", zero_division=0),
        "Precision Macro": precision_score(y_test_decoded, y_pred_decoded, average="macro", zero_division=0),
        "Precision Weighted": precision_score(y_test_decoded, y_pred_decoded, average="weighted", zero_division=0),
        "RMSE": root_mean_squared_error(y_test_decoded, y_pred_decoded),
        "MAE": mean_absolute_error(y_test_decoded, y_pred_decoded),
        "R2": r2_score(y_test_decoded, y_pred_decoded)
    }

    return results, y_test_decoded, y_pred_decoded


def test_neural_network(args):
    """Main function to test neural network model"""
    print(f"Loading model from: {args.model_path}")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device)

    label_encoder = checkpoint['label_encoder']
    feature_names = checkpoint['input_features']

    # Load test data
    print(f"Loading test data from: {args.test_file}")
    X_test, y_test = load_test_data(
        args.test_file, args.target, args.drop_cols,
        label_encoder, feature_names, args.use_centuries
    )

    print(f"Test data shape: {X_test.shape}")
    print(f"Number of test samples: {len(y_test)}")

    # Initialize model
    n_classes = len(label_encoder.classes_)
    input_dim = len(feature_names)
    model = FeedforwardNN(input_dim, n_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate model
    results, y_true, y_pred = evaluate_model(model, X_test, y_test, label_encoder, device)

    # Print results
    print("\n" + "="*50)
    print(f"Neural Network Test Results - {os.path.basename(args.test_file)}")
    print("="*50)
    for metric, value in results.items():
        print(f"{metric:20s}: {value:.4f}")
    print("="*50)

    # Save detailed results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        # Save metrics to CSV
        results_df = pd.DataFrame([results])
        results_df['dataset'] = os.path.basename(args.test_file)
        results_df['model'] = 'neural_network'
        results_df['target'] = args.target

        output_file = os.path.join(args.output_dir, args.output_file)
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            results_df = pd.concat([existing_df, results_df], ignore_index=True)

        results_df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")

        # Save detailed classification report
        report_file = os.path.join(args.output_dir, f"classification_report_{os.path.basename(args.test_file).split('.')[0]}.txt")
        with open(report_file, 'w') as f:
            f.write(f"Classification Report for Neural Network - {args.target}\n")
            f.write(f"Dataset: {args.test_file}\n")
            f.write("="*60 + "\n")
            f.write(classification_report(y_true, y_pred))
            f.write("\n" + "="*60 + "\n")
            f.write("Confusion Matrix:\n")
            f.write(str(confusion_matrix(y_true, y_pred)))

        print(f"Classification report saved to: {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test neural network model on test datasets")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model file")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument("--drop_cols", nargs='+', required=True, help="Columns to drop")
    parser.add_argument("--output_dir", type=str, help="Directory to save results")
    parser.add_argument("--output_file", type=str, default="nn_test_results.csv", help="Output file name")
    parser.add_argument("--use_centuries", action="store_true", help="Use century column as target")

    args = parser.parse_args()
    test_neural_network(args)