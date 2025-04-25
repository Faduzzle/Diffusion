# ✅ Shared Configuration for Training and Prediction

CONFIG = {
    # --- Model I/O ---
    "checkpoint_dir": "checkpoints",
    "save_name": "bitcoin_data_trained",  # used during training
    "checkpoint_path": "checkpoints/bitcoin_data_trained.pth",  # used during prediction

    # --- Data ---
    "tsf_path": "Data Files/bitcoin_data.tsf",
    "history_len": 50,
    "predict_len": 50,
    "input_dim": 1,
    "n_samples": 1000,

    # --- Training ---
    "n_epochs": 500,
    "batch_size": 64,
    "lr": 1e-3,
    "checkpoint_freq": 100,

    # --- Sampling ---
    "num_diffusion_timesteps": 500,
    "num_paths": 1000,

    # --- Toggle ---
    "use_tsf": True
}
