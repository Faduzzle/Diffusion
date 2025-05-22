import os
import time
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

from config import CONFIG
from model import ScoreTransformerNet
from sde import VPSDE
from data import SlidingWindowDataset, load_folder_as_tensor, WaveletSlidingWindowDataset

def update_ema(ema_model, model, decay):
    with torch.no_grad():
        model_params = dict(model.named_parameters())
        for name, ema_param in ema_model.named_parameters():
            ema_param.data.mul_(decay).add_(model_params[name].data, alpha=1 - decay)

def get_dynamic_mask_ratio(epoch, n_epochs, min_ratio, max_ratio):
    # Cosine schedule for mask ratio
    progress = epoch / n_epochs
    cosine_factor = 0.5 * (1 + np.cos(progress * np.pi))
    return min_ratio + (max_ratio - min_ratio) * cosine_factor

def train(model, sde, dataset, history_len, predict_len, n_epochs=1000, batch_size=64, lr=1e-3,
          save_dir='checkpoints', checkpoint_freq=100, device="cuda", save_name="latest", 
          ema_decay=0.999, cond_drop_prob=0.1, mask_ratio=0.15, dynamic_masking=True,
          min_mask_ratio=0.05, max_mask_ratio=0.25, cfg_weight=3.0, samples_per_epoch=500):

    os.makedirs(save_dir, exist_ok=True)
    
    # Optimization setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    scaler = GradScaler()  # For mixed precision training
    
    # EMA model setup
    ema_model = copy.deepcopy(model)
    ema_model.eval()

    # DataLoader setup for efficient data loading
    loader_kwargs = {
        'batch_size': batch_size,
        'pin_memory': True,
        'num_workers': 4,
        'persistent_workers': True
    }
    
    loss_history = []
    best_loss = float('inf')
    
    # Configure progress bars
    epoch_bar = tqdm(range(n_epochs), desc="Epochs", position=0)
    batch_bar = tqdm(total=samples_per_epoch // batch_size, desc="Training", position=1, leave=False)
    
    start_time = time.time()
    torch.backends.cudnn.benchmark = True

    try:
        for epoch in epoch_bar:
            model.train()
            total_loss = 0.0
            batch_bar.reset()

            # Create new random subset for this epoch
            subset_indices = torch.randint(0, len(dataset), (samples_per_epoch,)).tolist()
            loader = DataLoader(
                dataset,
                sampler=torch.utils.data.SubsetRandomSampler(subset_indices),
                **loader_kwargs
            )

            # Calculate dynamic mask ratio if enabled
            current_mask_ratio = get_dynamic_mask_ratio(epoch, n_epochs, min_mask_ratio, max_mask_ratio) if dynamic_masking else mask_ratio

            for hist, future in loader:
                # Move data to device efficiently
                hist = hist.to(device, non_blocking=True)
                future = future.to(device, non_blocking=True)
                
                # Generate timesteps and noise
                t = (torch.rand(hist.size(0), device=device) * 0.998 + 0.001).unsqueeze(1)
                t_expand = t.expand(-1, future.size(1))

                with autocast():  # Mixed precision
                    mean, std = sde.p(future, t_expand)
                    noise = torch.randn_like(std)
                    x_t = mean + std * noise

                    # Forward pass
                    score_pred = model(x_t, hist, t, 
                                    cond_drop_prob=cond_drop_prob,
                                    mask_ratio=current_mask_ratio,
                                    cfg_weight=cfg_weight)
                    
                    loss = torch.mean((std * score_pred + noise) ** 2)

                # Optimized backward pass
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                # Update EMA model
                update_ema(ema_model, model, ema_decay)

                total_loss += loss.item()
                batch_bar.update(1)
                batch_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.6f}",
                    mask=f"{current_mask_ratio:.3f}"
                )

            # End of epoch processing
            avg_loss = total_loss / len(loader)
            loss_history.append(avg_loss)
            scheduler.step()
            
            # Update progress bars
            epoch_bar.set_postfix(
                avg_loss=f"{avg_loss:.4f}",
                best=f"{best_loss:.4f}",
                mask=f"{current_mask_ratio:.3f}"
            )
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    "score_net_state_dict": model.state_dict(),
                    "ema_score_net_state_dict": ema_model.state_dict(),
                    "epoch": epoch,
                    "loss_history": loss_history,
                    "best_loss": best_loss,
                    "config": {
                        "cond_drop_prob": cond_drop_prob,
                        "mask_ratio": mask_ratio,
                        "dynamic_masking": dynamic_masking,
                        "min_mask_ratio": min_mask_ratio,
                        "max_mask_ratio": max_mask_ratio,
                        "cfg_weight": cfg_weight
                    }
                }, os.path.join(save_dir, f"{save_name}_best.pth"))
                
            # Regular checkpoint saving
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save({
                    "score_net_state_dict": model.state_dict(),
                    "ema_score_net_state_dict": ema_model.state_dict(),
                    "epoch": epoch,
                    "loss_history": loss_history,
                    "config": {
                        "cond_drop_prob": cond_drop_prob,
                        "mask_ratio": mask_ratio,
                        "dynamic_masking": dynamic_masking,
                        "min_mask_ratio": min_mask_ratio,
                        "max_mask_ratio": max_mask_ratio,
                        "cfg_weight": cfg_weight
                    }
                }, os.path.join(save_dir, f"{save_name}.pth"))

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    finally:
        batch_bar.close()
        
    duration = time.time() - start_time
    print(f"\n✅ Training completed in {duration:.2f} sec ({duration/60:.2f} min)")
    print(f"Best loss: {best_loss:.4f}")

    # Plot training metrics
    plt.figure(figsize=(12, 5))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss history
    ax1.plot(loss_history)
    ax1.set_title("Training Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    
    # Plot masking ratio if dynamic
    if dynamic_masking:
        mask_ratios = [get_dynamic_mask_ratio(e, n_epochs, min_mask_ratio, max_mask_ratio) 
                      for e in range(len(loss_history))]
        ax2.plot(mask_ratios)
        ax2.set_title("Dynamic Mask Ratio Schedule")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Mask Ratio")
        ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{save_name}_training_plots.png"))
    plt.close()

def train_model_from_config():
    device = CONFIG.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    history_len = CONFIG["history_len"]
    predict_len = CONFIG["predict_len"]
    samples_per_epoch = CONFIG.get("samples_per_epoch", 500)  # Number of windows to sample per epoch

    # Load and preprocess data
    train_tensor = load_folder_as_tensor(CONFIG["train_data_path"])
    print("🧪 Loaded train_tensor shape:", train_tensor.shape)

    # Create appropriate dataset based on config
    if CONFIG.get("use_wavelets", False):
        dataset = WaveletSlidingWindowDataset(
            data_tensor=train_tensor,
            history_len=history_len,
            predict_len=predict_len,
            mask_prob=CONFIG.get("mask_prob", 0.1),
            wavelet=CONFIG.get("wavelet", "db4"),
            level=CONFIG.get("wavelet_level", 3)
        )
        input_dim = train_tensor.shape[-1] * (CONFIG.get("wavelet_level", 3) + 1)
        print("🌊 Using wavelet decomposition with input dimension:", input_dim)
    else:
        dataset = SlidingWindowDataset(
            data_tensor=train_tensor,
            history_len=history_len,
            predict_len=predict_len,
            mask_prob=CONFIG.get("mask_prob", 0.1)
        )
        input_dim = train_tensor.shape[-1]
    
    # Print dataset info
    print(f"📊 Total possible windows: {len(dataset)}")
    print(f"📊 Sampling {samples_per_epoch} windows per epoch")

    # Create model
    model = ScoreTransformerNet(
        input_dim=input_dim,
        history_len=history_len,
        predict_len=predict_len,
        model_dim=CONFIG.get("model_dim", 256)
    ).to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model Parameters: {total_params:,} total, {trainable_params:,} trainable")

    sde = VPSDE()
    train(model, sde, dataset,
          history_len=history_len,
          predict_len=predict_len,
          n_epochs=CONFIG["n_epochs"],
          batch_size=CONFIG["batch_size"],
          lr=CONFIG["lr"],
          save_dir=CONFIG["checkpoint_dir"],
          checkpoint_freq=CONFIG["checkpoint_freq"],
          save_name=CONFIG["save_name"],
          device=device,
          cond_drop_prob=CONFIG.get("cond_drop_prob", 0.1),
          mask_ratio=CONFIG.get("mask_ratio", 0.15),
          dynamic_masking=CONFIG.get("dynamic_masking", True),
          min_mask_ratio=CONFIG.get("min_mask_ratio", 0.05),
          max_mask_ratio=CONFIG.get("max_mask_ratio", 0.25),
          cfg_weight=CONFIG.get("classifier_free_guidance_weight", 3.0),
          samples_per_epoch=samples_per_epoch)

if __name__ == "__main__":
    train_model_from_config()
