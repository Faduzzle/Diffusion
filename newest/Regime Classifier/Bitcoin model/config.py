import torch

CONFIG = {
    # === Time Window Settings ===
    "history_len": 50,
    "predict_len": 20,
    "window_stride": 5,  # How far to move the window each time (smaller = more overlap)

    # === Model Settings ===
    "model_dim": 128,

    # === Training Settings ===
    "n_epochs": 80,
    "samples_per_epoch": 300,
    "batch_size": 64,
    "lr": 1e-3,
    "ema_decay": 0.999,
    "checkpoint_freq": 35,
    "checkpoint_dir": "checkpoints",
    "save_name": "diffusion_model",

    # === Data Paths ===
    "train_data_path": r"C:\Users\thoma\Desktop\Diffusion\Bitcoin model\training data",
    "test_data_path": r"C:\Users\thoma\Desktop\Diffusion\Bitcoin model\Testing Data",

    # === Data Augmentation ===
    "mask_prob": 0.05,

    # === Device ===
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # === Inference Settings ===
    "checkpoint_path": r"C:\Users\thoma\Desktop\Diffusion\Bitcoin model\checkpoints\diffusion_model.pth",
    "num_diffusion_timesteps": 500,
    "num_paths": 800,

    # === Classifier-Free Guidance and Masking ===
    "cond_drop_prob": 0.1,  # Probability of dropping conditional information
    "mask_ratio": 0.15,     # Ratio of tokens to mask during training
    "classifier_free_guidance_weight": 3.0,
    "min_mask_ratio": 0.05, # Minimum masking ratio
    "max_mask_ratio": 0.25, # Maximum masking ratio
    "dynamic_masking": True, # Whether to use dynamic masking schedule

    # Wavelet Analysis Settings
    "use_wavelets": True,  # Enable/disable wavelet decomposition
    "wavelet": "db4",      # Wavelet type (db4 = Daubechies-4)
    "wavelet_level": 3,    # Number of wavelet decomposition levels
    
    # Prediction Settings
    "num_prediction_samples": 100,  # Number of prediction paths to generate
}
