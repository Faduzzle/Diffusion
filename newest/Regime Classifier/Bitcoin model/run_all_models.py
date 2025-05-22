import os
import time
import torch
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime

from config import CONFIG
from trainer import train_model_from_config
from baseline_models import run_baseline_models
from predictandsave import generate_predictions
from data import WaveletSlidingWindowDataset, load_folder_as_tensor

class ModelRunner:
    def __init__(self):
        self.start_time = None
        self.config = CONFIG
        self.checkpoint_dir = CONFIG["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Create a timestamp for this run
        # Format: YYYYMMDD_HHMMSS (e.g., 20240315_144530 for March 15, 2024, 14:45:30)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.checkpoint_dir, f"run_log_{self.timestamp}.txt")
        
        # Wavelet configuration
        self.wavelet_config = {
            "use_wavelets": CONFIG.get("use_wavelets", True),
            "wavelet": CONFIG.get("wavelet", "db4"),
            "level": CONFIG.get("wavelet_level", 3)
        }
        
    def log(self, message):
        """Log a message to both console and file"""
        print(message)
        with open(self.log_file, "a", encoding='utf-8') as f:
            f.write(f"{message}\n")

    def run_all(self):
        """Run all models with progress tracking"""
        self.start_time = time.time()
        total_steps = 4  # Diffusion training, prediction, baselines, evaluation
        
        with tqdm(total=total_steps, desc="Overall Progress", position=0) as pbar:
            try:
                # Step 1: Train Diffusion Model
                self.log("\n🚀 Training Diffusion Model...")
                pbar.set_description("Training Diffusion Model")
                train_model_from_config()
                pbar.update(1)
                
                # Step 2: Generate Predictions
                self.log("\n📊 Generating Diffusion Model Predictions...")
                pbar.set_description("Generating Predictions")
                
                # Load model and data
                device = torch.device(self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
                checkpoint = torch.load(self.config["checkpoint_path"], map_location=device)
                
                # Create and load model
                model = ScoreTransformerNet(
                    input_dim=test_tensor.shape[-1] * (self.wavelet_config["level"] + 1) if self.wavelet_config["use_wavelets"] else test_tensor.shape[-1],
                    history_len=self.config["history_len"],
                    predict_len=self.config["predict_len"],
                    model_dim=self.config.get("model_dim", 256)
                ).to(device)
                
                # Load state dict (prefer EMA model if available)
                if "ema_score_net_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["ema_score_net_state_dict"])
                    self.log("Using EMA model weights")
                else:
                    model.load_state_dict(checkpoint["score_net_state_dict"])
                    self.log("Using regular model weights")
                
                model.eval()  # Set to evaluation mode
                
                test_tensor = load_folder_as_tensor(self.config["test_data_path"])
                
                # Create dataset with wavelet transform if enabled
                if self.wavelet_config["use_wavelets"]:
                    dataset = WaveletSlidingWindowDataset(
                        data_tensor=test_tensor,
                        history_len=self.config["history_len"],
                        predict_len=self.config["predict_len"],
                        wavelet=self.wavelet_config["wavelet"],
                        level=self.wavelet_config["level"]
                    )
                    self.log("🌊 Using wavelet decomposition for predictions")
                else:
                    dataset = SlidingWindowDataset(
                        data_tensor=test_tensor,
                        history_len=self.config["history_len"],
                        predict_len=self.config["predict_len"]
                    )
                
                # Generate predictions with wavelet analysis
                generate_predictions(
                    model=model,
                    dataset=dataset,
                    device=device,
                    config=self.config,
                    n_samples=self.config.get("num_prediction_samples", 100)
                )
                pbar.update(1)
                
                # Step 3: Run Baseline Models
                self.log("\n🔍 Training and Evaluating Baseline Models...")
                pbar.set_description("Running Baselines")
                run_baseline_models(
                    history_len=self.config["history_len"],
                    predict_len=self.config["predict_len"]
                )
                pbar.update(1)
                
                # Step 4: Final Evaluation and Comparison
                self.log("\n📈 Performing Final Evaluation...")
                pbar.set_description("Final Evaluation")
                self.compare_all_models()
                pbar.update(1)
                
                duration = time.time() - self.start_time
                self.log(f"\n✅ All models completed in {duration/60:.2f} minutes!")
                
            except Exception as e:
                self.log(f"\n❌ Error during execution: {str(e)}")
                raise e

    def compare_all_models(self):
        """Compare all model results and generate final comparison"""
        try:
            # Load metrics
            predictions_dir = os.path.join("Data", "Predictions")
            diffusion_metrics = os.path.join(predictions_dir, "prediction_metrics.csv")
            baseline_metrics = os.path.join(predictions_dir, "baseline_metrics.csv")
            wavelet_metrics = os.path.join(predictions_dir, "aggregate_metrics.npz")
            
            if not os.path.exists(diffusion_metrics) or not os.path.exists(baseline_metrics):
                self.log("❌ Missing metrics files for comparison")
                return
            
            # Create comprehensive comparison plots and tables
            self.log("📊 Generating final comparison visualizations...")
            self.generate_comparison_plots(diffusion_metrics, baseline_metrics, wavelet_metrics)
            
        except Exception as e:
            self.log(f"❌ Error in final comparison: {str(e)}")

    def generate_comparison_plots(self, diffusion_path, baseline_path, wavelet_path=None):
        """Generate comprehensive comparison visualizations"""
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Load metrics
        diff_df = pd.read_csv(diffusion_path)
        base_df = pd.read_csv(baseline_path)
        
        # Create comparison directory
        comparison_dir = os.path.join("Data", "Comparisons", self.timestamp)
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Plot settings
        plt.style.use('seaborn')
        metrics = ['MAE', 'MSE', 'RMSE']
        
        # 1. Box plots for each metric
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for i, metric in enumerate(metrics):
            diff_data = diff_df[metric]
            base_data = {model: base_df[f"{model}_{metric}"] 
                        for model in ['ARIMA', 'SARIMA', 'ExpSmoothing', 'RandomForest', 
                                    'XGBoost', 'LightGBM', 'LSTM', 'Transformer']}
            base_data['Diffusion'] = diff_data
            
            data = []
            labels = []
            for model, values in base_data.items():
                data.extend(values)
                labels.extend([model] * len(values))
            
            sns.boxplot(x=labels, y=data, ax=axes[i])
            axes[i].set_title(f'{metric} Distribution')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, 'metric_distributions.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # 2. Create summary statistics
        summary = {
            'Diffusion': {metric: {
                'mean': diff_df[metric].mean(),
                'std': diff_df[metric].std(),
                'median': diff_df[metric].median(),
            } for metric in metrics},
        }
        
        for model in ['ARIMA', 'SARIMA', 'ExpSmoothing', 'RandomForest', 
                     'XGBoost', 'LightGBM', 'LSTM', 'Transformer']:
            summary[model] = {metric: {
                'mean': base_df[f"{model}_{metric}"].mean(),
                'std': base_df[f"{model}_{metric}"].std(),
                'median': base_df[f"{model}_{metric}"].median(),
            } for metric in metrics}
        
        # Save summary statistics
        with open(os.path.join(comparison_dir, 'summary_statistics.txt'), 'w') as f:
            f.write("Model Performance Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for model, metrics_dict in summary.items():
                f.write(f"{model}:\n")
                for metric, stats in metrics_dict.items():
                    f.write(f"  {metric}:\n")
                    for stat_name, value in stats.items():
                        f.write(f"    {stat_name}: {value:.4f}\n")
                f.write("\n")
            
            # Add wavelet analysis summary if available
            if wavelet_path and os.path.exists(wavelet_path):
                f.write("\nWavelet Analysis Summary\n")
                f.write("=" * 50 + "\n\n")
                wavelet_metrics = np.load(wavelet_path)
                for metric_name, values in wavelet_metrics.items():
                    f.write(f"{metric_name}:\n")
                    for i, val in enumerate(values):
                        f.write(f"  Level {i}: {val:.4f}\n")
                    f.write("\n")
        
        self.log(f"✅ Comparison results saved to: {comparison_dir}")

def main():
    """Main entry point"""
    print("🚀 Starting comprehensive model evaluation...")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    runner = ModelRunner()
    runner.run_all()

if __name__ == "__main__":
    main() 