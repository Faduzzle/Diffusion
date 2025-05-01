import torch

CONFIG = {
    # ====== Paths ======
    "train_data_folder": "./training_data",  # Folder with raw CSVs (for preprocessing)
    "processed_dataset_path": "./processed_data/train_dataset.pt",  # Preprocessed .pt dataset
    "checkpoint_dir": "./checkpoints",  # Folder where checkpoints will be saved

    # ====== Training Settings ======
    "save_name": "multivariate_model_v1",  # Checkpoint base name during training
    "batch_size": 128,
    "n_epochs": 1000,
    "learning_rate": 1e-4,
    "checkpoint_freq": 50,  # Save checkpoint every N epochs

    # ====== Model Settings ======
    "history_len": 50,
    "predict_len": 20,
    "model_dim": 256,
    "num_heads": 4,
    "num_layers": 4,
    "num_diffusion_timesteps": 500,

    # ====== Device ======
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # ====== Sampling Settings ======
    "checkpoint_path": "./checkpoints/multivariate_model_v1_epoch1000.pth",  # Path to trained model checkpoint
    "predictions_per_window": 1000,  # Number of samples per window during sampling
}
