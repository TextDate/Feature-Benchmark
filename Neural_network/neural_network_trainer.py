import argparse
import os
import pandas as pd
import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    root_mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


def load_data(train_file, val_file, target_col, drop_cols, use_smote, use_tomek, use_centuries, exclude_decades=None, exclude_centuries=None):
    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)

    if exclude_decades:
        df_train = df_train[~df_train["decade"].isin(exclude_decades)]
        df_val = df_val[~df_val["decade"].isin(exclude_decades)]
    if exclude_centuries:
        df_train = df_train[~df_train["century"].isin(exclude_centuries)]
        df_val = df_val[~df_val["century"].isin(exclude_centuries)]

    y_train = df_train["century"] if use_centuries else df_train[target_col]
    y_val = df_val["century"] if use_centuries else df_val[target_col]
    X_train = df_train.drop([target_col, "century", "decade"] + drop_cols, axis=1)
    X_val = df_val.drop([target_col, "century", "decade"] + drop_cols, axis=1)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train.values.ravel())
    y_val = label_encoder.transform(y_val.values.ravel())

    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    if use_tomek:
        tomek = TomekLinks()
        X_train, y_train = tomek.fit_resample(X_train, y_train)

    return X_train, y_train, X_val, y_val, label_encoder


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


def train_and_save_nn(args):
    X_train, y_train, X_val, y_val, label_encoder = load_data(
        args.train_file, args.val_file, args.target,
        args.drop_cols.split(","), args.use_smote, args.use_tomek,
        args.use_centuries, args.exclude_decades, args.exclude_centuries,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    n_classes = len(np.unique(y_train))

    print(f"\n Shape of training data: {X_train.shape}, Number of classes: {n_classes}", flush=True)
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    
    print(f"Shape of validation data: {X_val.shape}, Number of classes: {n_classes}", flush=True)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    print("Training data and validation data loaded successfully.", flush=True)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print("Data loaded into DataLoader successfully.", flush=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    model = FeedforwardNN(X_train.shape[1], n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0] * n_classes).to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"\n=== Training Neural Network for {args.epochs} epochs ===", flush=True)
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        logits = model(X_val_tensor.to(device))
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_true = y_val
        y_pred_decoded = label_encoder.inverse_transform(y_pred)
        y_val_decoded = label_encoder.inverse_transform(y_true)

    results = {
        "Accuracy": accuracy_score(y_val_decoded, y_pred_decoded),
        "F1 Macro": f1_score(y_val_decoded, y_pred_decoded, average="macro", zero_division=0),
        "F1 Weighted": f1_score(y_val_decoded, y_pred_decoded, average="weighted", zero_division=0),
        "Recall Macro": recall_score(y_val_decoded, y_pred_decoded, average="macro", zero_division=0),
        "Recall Weighted": recall_score(y_val_decoded, y_pred_decoded, average="weighted", zero_division=0),
        "Precision Macro": precision_score(y_val_decoded, y_pred_decoded, average="macro", zero_division=0),
        "Precision Weighted": precision_score(y_val_decoded, y_pred_decoded, average="weighted", zero_division=0),
        "RMSE": root_mean_squared_error(y_val_decoded, y_pred_decoded),
        "MAE": mean_absolute_error(y_val_decoded, y_pred_decoded),
        "R2": r2_score(y_val_decoded, y_pred_decoded)
    }
    print("----- Validation Results -----", flush=True)
    for k, v in results.items():
        print(f"{k}: {v:.4f}", flush=True)
    print("------------------------------", flush=True)

    output_path = os.path.join(args.output_dir, "feedforward_nn_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder': label_encoder,
        'input_features': X_train.columns.tolist()
    }, output_path)
    print(f"Model saved to: {output_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save a feedforward neural network model.")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.3)
    parser.add_argument("--drop_cols", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_smote", action="store_true")
    parser.add_argument("--use_tomek", action="store_true")
    parser.add_argument("--exclude_decades", type=int, nargs='*')
    parser.add_argument("--use_centuries", action="store_true")
    parser.add_argument("--exclude_centuries", type=int, nargs='*')
    args = parser.parse_args()

    train_and_save_nn(args)
