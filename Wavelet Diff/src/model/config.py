# src/model/config.py

import os
import torch

PROJECT_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_ROOT         = os.path.join(PROJECT_ROOT, "data")
WAVELETS_TRAIN_DIR = os.path.join(DATA_ROOT, "wavelets", "train wavelet")
WAVELETS_TEST_DIR  = os.path.join(DATA_ROOT, "wavelets", "test wavelet")
CHECKPOINT_DIR     = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
PREDICTION_OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")

os.makedirs(WAVELETS_TRAIN_DIR, exist_ok=True)
os.makedirs(WAVELETS_TEST_DIR,  exist_ok=True)
os.makedirs(CHECKPOINT_DIR,     exist_ok=True)
os.makedirs(PREDICTION_OUT_DIR, exist_ok=True)

CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "history_len": 50,
    "predict_len": 20,
    "wavelet": "db4",
    "wavelet_level": 4,  # 4 levels → 5 bands

    # Paths
    "train_data_path": os.path.join(DATA_ROOT, "wavelets", "train wavelet"),
    "test_data_path":  os.path.join(DATA_ROOT, "wavelets", "test wavelet"),
    "wavelets_path":  os.path.join(DATA_ROOT, "wavelets", "train wavelet"),
    "checkpoint_dir": CHECKPOINT_DIR,
    "checkpoint_path": os.path.join(CHECKPOINT_DIR, "score_transformer_ep300.pth"),
    "prediction_output_dir": PREDICTION_OUT_DIR,

    # Training hyperparams
    "model_dim": 256,
    "num_heads": 8,
    "num_layers": 4,
    "mlp_ratio": 4.0,
    "drop_rate": 0.1,
    "attn_drop_rate": 0.1,
    "samples_per_epoch": 400,
    "batch_size": 32,
    "pin_memory": True,
    "num_workers": 4,
    "prefetch_factor": 2,
    "grad_accum_steps": 1,
    "n_epochs": 500,
    "lr": 1e-4,
    "ema_decay": 0.999,
    "checkpoint_freq": 10,
    "save_name": "score_transformer",
    "mask_ratio": 0.1,
    "cond_drop_prob": 0.2,

    # Diffusion / sampling
    "num_diffusion_timesteps": 500,
    "classifier_free_guidance_weight": .25,
    "regular_samples": 50,
    "high_samples": 200,
    "num_regular_windows": 3,
    "include_high_sample": True,
    "input_dim": 1,

    # Normalization files (still from train‐wavelet)
    #   train_wavelet/wavelet_means.pt  (shape [5,feat_dim])
    #   train_wavelet/wavelet_stds.pt   (shape [5,feat_dim])
}
