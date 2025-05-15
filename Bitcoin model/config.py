import torch

CONFIG = {
    # === Time Window Settings ===
    "history_len": 50,
    "predict_len": 20,

    # === Model Settings ===
    "model_dim": 256,

    # === Training Settings ===
    "n_epochs": 500,
    "samples_per_epoch": 700,
    "batch_size": 64,
    "lr": 1e-3,
    "ema_decay": 0.999,
    "checkpoint_freq": 50,
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

    # === Classifier-Free Guidance ===
    "cond_drop_prob": 0.1,
    "classifier_free_guidance_weight": 3.0,
}
