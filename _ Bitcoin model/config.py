import torch

CONFIG = {
    # === Data settings ===
    "train_csv_path": "Data Files/ROLLED_train_data_bitcoin.csv",
    "test_csv_path": "Data Files/ROLLED_test_data_bitcoin.csv",
    "use_csv": True,
    "use_synthetic": False,
    "history_len": 50,
    "predict_len": 20,
    "input_dim": 1,
    "n_samples": 1000,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # === Training ===
    "n_epochs": 200,
    "batch_size": 64,
    "lr": 1e-3,
    "checkpoint_freq": 40,
    "checkpoint_dir": "checkpoints",
    "save_name": "bitcoin_csv_trained",

    # === Inference ===
    "checkpoint_path": "checkpoints/bitcoin_csv_trained.pth",
    "num_diffusion_timesteps": 500,
    "num_paths": 3000
}
