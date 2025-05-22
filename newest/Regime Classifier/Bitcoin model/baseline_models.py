import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

from config import CONFIG
from data import load_folder_as_tensor, SlidingWindowDataset

SAVE_DIR = os.path.join("Data", "Predictions")
os.makedirs(SAVE_DIR, exist_ok=True)

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, target_length):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length
        self.target_length = target_length

    def __len__(self):
        return len(self.data) - self.seq_length - self.target_length + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length:idx + self.seq_length + self.target_length]
        return x, y

def load_training_data():
    """Load the same training data as used by the diffusion model"""
    train_tensor = load_folder_as_tensor(CONFIG["train_data_path"])
    print(f"Loaded training data shape: {train_tensor.shape}")
    return train_tensor

def load_testing_data():
    """Load the same testing data as used by the diffusion model"""
    test_tensor = load_folder_as_tensor(CONFIG["test_data_path"])
    print(f"Loaded testing data shape: {test_tensor.shape}")
    return test_tensor

def prepare_sliding_windows(data_tensor, history_len, predict_len):
    """Create sliding windows in the same way as the diffusion model"""
    dataset = SlidingWindowDataset(
        data_tensor=data_tensor,
        history_len=history_len,
        predict_len=predict_len,
        mask_prob=0.0  # No masking for baseline models
    )
    return dataset

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_len):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_len = output_len
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_len)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x.unsqueeze(-1))
        return self.fc(lstm_out[:, -1, :])

class TransformerForecaster(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_len):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, output_len)
        
    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1))
        x = self.transformer(x)
        return self.output_proj(x[:, -1, :])

def train_deep_model(model, train_dataset, val_dataset, epochs=100, batch_size=32, device="cuda"):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    model = model.to(device)
    
    best_val_loss = float('inf')
    best_model = None
    patience = 10
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_hist, x_future in train_loader:
            x_hist, x_future = x_hist.to(device), x_future.to(device)
            optimizer.zero_grad()
            pred = model(x_hist)
            loss = criterion(pred, x_future)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_hist, x_future in val_loader:
                x_hist, x_future = x_hist.to(device), x_future.to(device)
                pred = model(x_hist)
                val_loss += criterion(pred, x_future).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss/len(val_loader):.6f}")
    
    return best_model

def prepare_ml_features(dataset):
    """Convert dataset into features suitable for ML models"""
    X, y = [], []
    for i in range(len(dataset)):
        hist, future = dataset[i]
        X.append(hist.numpy())
        y.append(future.numpy())
    return np.array(X), np.array(y)

def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

def run_baseline_models(history_len, predict_len):
    print("Loading data...")
    train_tensor = load_training_data()
    test_tensor = load_testing_data()
    
    # Prepare training and validation datasets
    train_dataset = prepare_sliding_windows(train_tensor, history_len, predict_len)
    test_dataset = prepare_sliding_windows(test_tensor, history_len, predict_len)
    
    # Split training data into train and validation
    train_size = int(0.8 * len(train_dataset))
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, 
        [train_size, len(train_dataset) - train_size]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Statistical Models
    statistical_models = {
        "ARIMA": lambda hist: ARIMA(hist, order=(5, 1, 0)).fit(),
        "SARIMA": lambda hist: SARIMAX(hist, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False),
        "ExpSmoothing": lambda hist: ExponentialSmoothing(hist, trend="add", seasonal="add", seasonal_periods=7).fit()
    }

    # ML Models
    ml_models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "SVR": SVR(kernel='rbf')
    }

    # Deep Learning Models
    dl_models = {
        "LSTM": LSTMForecaster(input_dim=1, hidden_dim=64, num_layers=2, output_len=predict_len),
        "Transformer": TransformerForecaster(input_dim=1, d_model=64, nhead=4, num_layers=2, output_len=predict_len)
    }

    # Train ML models
    print("\nTraining ML models...")
    X_train, y_train = prepare_ml_features(train_subset)
    trained_ml_models = {}
    for name, model in ml_models.items():
        print(f"Training {name}...")
        model.fit(X_train.reshape(X_train.shape[0], -1), y_train.reshape(y_train.shape[0], -1))
        trained_ml_models[name] = model

    # Train deep learning models
    print("\nTraining deep learning models...")
    trained_dl_models = {}
    for name, model in dl_models.items():
        print(f"Training {name}...")
        trained_dl_models[name] = train_deep_model(
            model, train_subset, val_subset, 
            epochs=100, batch_size=32, device=device
        )

    # Evaluate on test set
    all_metrics = []
    per_sample_metrics = []

    for idx in range(len(test_dataset)):
        x_hist, x_true = test_dataset[idx]
        x_hist = x_hist.numpy()
        x_true = x_true.numpy()
        
        row = {"Sample": idx}
        
        # Plot setup
        fig, ax = plt.subplots(figsize=(12, 6))
        hist_line = np.arange(len(x_hist))
        pred_line = np.arange(len(x_hist), len(x_hist) + len(x_true))
        
        ax.plot(hist_line, x_hist, label="History", color="black")
        ax.plot(pred_line, x_true, label="True", linestyle="--", color="green")

        # Statistical Models
        for model_name, model_fn in statistical_models.items():
            try:
                model = model_fn(x_hist)
                pred = model.forecast(steps=predict_len)
                metrics = evaluate_forecast(x_true, pred)

                for k, v in metrics.items():
                    row[f"{model_name}_{k}"] = v

                per_sample_metrics.append({
                    "Model": model_name,
                    "Sample": idx,
                    **metrics
                })

                ax.plot(pred_line, pred, label=f"{model_name}")
            except Exception as e:
                print(f"❌ {model_name} failed on Sample {idx}: {e}")

        # ML Models
        for model_name, model in trained_ml_models.items():
            try:
                pred = model.predict(x_hist.reshape(1, -1))
                pred = pred.reshape(-1)
                metrics = evaluate_forecast(x_true, pred)

                for k, v in metrics.items():
                    row[f"{model_name}_{k}"] = v

                per_sample_metrics.append({
                    "Model": model_name,
                    "Sample": idx,
                    **metrics
                })

                ax.plot(pred_line, pred, label=f"{model_name}")
            except Exception as e:
                print(f"❌ {model_name} failed on Sample {idx}: {e}")

        # Deep Learning Models
        x_hist_tensor = torch.FloatTensor(x_hist).to(device)
        for model_name, model in trained_dl_models.items():
            try:
                model.eval()
                with torch.no_grad():
                    pred = model(x_hist_tensor).cpu().numpy()
                metrics = evaluate_forecast(x_true, pred)

                for k, v in metrics.items():
                    row[f"{model_name}_{k}"] = v

                per_sample_metrics.append({
                    "Model": model_name,
                    "Sample": idx,
                    **metrics
                })

                ax.plot(pred_line, pred, label=f"{model_name}")
            except Exception as e:
                print(f"❌ {model_name} failed on Sample {idx}: {e}")

        ax.set_title(f"Sample {idx} - All Models Comparison")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"baseline_sample_{idx}.png"), bbox_inches='tight')
        plt.close()
        all_metrics.append(row)

    if not per_sample_metrics:
        print("❌ No valid samples found for baseline evaluation.")
        return

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(SAVE_DIR, "baseline_metrics.csv"), index=False)
    print(f"✅ Saved baseline metrics to: {os.path.join(SAVE_DIR, 'baseline_metrics.csv')}")

    try:
        diffusion_df = pd.read_csv(os.path.join(SAVE_DIR, "prediction_metrics.csv"))
        diffusion_summary = diffusion_df[["Sample", "MAE", "MSE", "RMSE"]].copy()
        diffusion_summary["Model"] = "Diffusion"
        diffusion_summary["R2"] = np.nan
        per_sample_metrics += diffusion_summary.to_dict(orient="records")
    except Exception as e:
        print(f"❌ Failed to combine with diffusion metrics: {e}")

    all_df = pd.DataFrame(per_sample_metrics)
    summary_mean = all_df.groupby("Model").mean(numeric_only=True)
    summary_std = all_df.groupby("Model").std(numeric_only=True)

    # Create detailed summary with mean ± std
    highlight = pd.DataFrame(index=summary_mean.index, columns=summary_mean.columns)
    for col in ["MAE", "MSE", "RMSE", "R2"]:
        for idx in summary_mean.index:
            mean_val = summary_mean.loc[idx, col]
            std_val = summary_std.loc[idx, col]
            highlight.loc[idx, col] = f"{mean_val:.4f} ± {std_val:.4f}"

    # Highlight best models
    for col in ["MAE", "MSE", "RMSE"]:
        if summary_mean[col].notna().any():
            best_model = summary_mean[col].idxmin()
            highlight.loc[best_model, col] += " *"

    if summary_mean["R2"].notna().any():
        best_r2 = summary_mean["R2"].idxmax()
        highlight.loc[best_r2, "R2"] += " *"

    print("\nModel Performance Summary (mean ± std):")
    print(highlight)

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    metrics = ["MAE", "MSE", "RMSE", "R2"]
    
    for i, metric in enumerate(metrics):
        summary_mean[metric].plot(kind="bar", yerr=summary_std[metric], ax=axes[i], capsize=5)
        axes[i].set_title(f"{metric} by Model")
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "model_comparison.png"))
    plt.close()

    # Save detailed summary table
    fig, ax = plt.subplots(figsize=(12, 2 + 0.5 * len(highlight)))
    ax.axis("off")
    table_data = [[model] + list(row) for model, row in highlight.iterrows()]
    columns = ["Model"] + list(highlight.columns)
    table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
    table.scale(1.2, 1.5)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "model_comparison_table.png"), bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    run_baseline_models(
        history_len=CONFIG["history_len"],
        predict_len=CONFIG["predict_len"]
    )
