import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import CONFIG

SAVE_DIR = os.path.join("Data", "Predictions")
os.makedirs(SAVE_DIR, exist_ok=True)

def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

def run_baseline_models(history_len, predict_len):
    df = pd.read_csv(CONFIG["test_csv_path"])
    full_series = df.values.flatten()

    models = {
        "ARIMA": lambda hist: ARIMA(hist, order=(5, 1, 0)).fit(),
        "SARIMA": lambda hist: SARIMAX(hist, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)).fit(disp=False),
        "ExpSmoothing": lambda hist: ExponentialSmoothing(hist, trend="add", seasonal=None).fit()
    }

    all_metrics = []
    per_sample_metrics = []
    initial_price = 100.0

    # Load selected diffusion sampling start indices
    index_path = os.path.join(r'C:\Users\thoma\Desktop\Diffusion\Bitcoin model\Data\Predictions\selected_window_columns.csv')
    col_indices = pd.read_csv(index_path)["column_idx"].to_numpy()

    for idx, start in enumerate(col_indices):
        row = {"Sample": idx}
        end_hist = start + history_len
        end_pred = end_hist + predict_len

        if end_pred > len(full_series):
            print(f"⚠️ Sample {idx} (start={start}) too short for prediction. Skipping.")
            continue

        x_hist = full_series[start:end_hist]
        x_true = full_series[end_hist:end_pred]

        hist_price = initial_price * np.exp(np.cumsum(x_hist))
        true_price = hist_price[-1] * np.exp(np.cumsum(x_true))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(len(hist_price)), hist_price, label="History", color="black")
        ax.plot(range(len(hist_price), len(hist_price) + len(true_price)), true_price, label="True", linestyle="--", color="green")

        for model_name, model_fn in models.items():
            try:
                model = model_fn(x_hist)
                pred_returns = model.forecast(steps=predict_len)
                pred_prices = hist_price[-1] * np.exp(np.cumsum(pred_returns))
                metrics = evaluate_forecast(true_price, pred_prices)

                for k, v in metrics.items():
                    row[f"{model_name}_{k}"] = v

                per_sample_metrics.append({
                    "Model": model_name,
                    "Sample": idx,
                    **metrics
                })

                ax.plot(range(len(hist_price), len(hist_price) + len(pred_prices)), pred_prices, label=model_name)
            except Exception as e:
                print(f"❌ {model_name} failed on Sample {idx}: {e}")

        ax.set_title(f"Sample {idx}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"baseline_sample_{idx}.png"))
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

    highlight = summary_mean.copy().astype("object")
    for col in ["MAE", "MSE", "RMSE"]:
        if summary_mean[col].notna().any():
            best_model = summary_mean[col].idxmin()
            highlight.loc[best_model, col] = f"{summary_mean.loc[best_model, col]:.4f} *"
        else:
            highlight[col] = "N/A"

    if summary_mean["R2"].notna().any():
        best_r2 = summary_mean["R2"].idxmax()
        highlight.loc[best_r2, "R2"] = f"{summary_mean.loc[best_r2, 'R2']:.4f} *"
    else:
        highlight["R2"] = "N/A"

    print("\nBest models per metric:")
    print(highlight)

    summary_mean.plot(kind="bar", figsize=(10, 6))
    plt.title("Average Forecasting Metrics by Model (Price Space)")
    plt.ylabel("Score (lower is better except R²)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "model_comparison.png"))

    fig, ax = plt.subplots(figsize=(10, 2 + 0.5 * len(highlight)))
    ax.axis("off")
    table_data = [[model] + list(row) for model, row in highlight.iterrows()]
    columns = ["Model"] + list(highlight.columns)
    table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
    table.scale(1.2, 1.5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "model_comparison_table.png"))
    plt.close()

if __name__ == "__main__":
    run_baseline_models(
        history_len=CONFIG["history_len"],
        predict_len=CONFIG["predict_len"]
    )
