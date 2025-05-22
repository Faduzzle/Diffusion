# Wavelet Decomposition Integration Guide for Diffusion Transformer Forecasting

## 1. Introduction
This document outlines a step-by-step procedure to augment your existing diffusion-transformer forecasting pipeline with wavelet decomposition (MODWT/IMODWT). By predicting in the wavelet domain, you enable your model to learn frequency-specific patterns and then reconstruct a full-resolution time series.

## 2. Prerequisites
- Python ≥ 3.7
- PyTorch
- PyWavelets (`pip install PyWavelets`)
- Your existing codebase: `data.py`, `trainer.py`, `predictandsave.py`, `model.py`, etc.

## 3. Wavelet Utilities Module
Create a new file `wavelet_utils.py`:

```python
import numpy as np
import pywt

def modwt_decompose(x: np.ndarray, wavelet: str, level: int) -> np.ndarray:
    """
    Performs a stationary (maximal overlap) wavelet decomposition.

    Returns an array shape (level+1, L): [cA_level, cD_level, ..., cD1].
    """
    coeffs = pywt.modwt(x, wavelet, level=level)
    return np.stack(coeffs, axis=0)

def imodwt_reconstruct(coeffs: np.ndarray, wavelet: str) -> np.ndarray:
    """
    Reconstructs a signal from its MODWT coefficients array of shape (level+1, L).
    """
    coeffs_list = [coeffs[0]] + [coeffs[i] for i in range(1, coeffs.shape[0])]
    return pywt.imodwt(coeffs_list, wavelet)
```

## 4. Dataset Wrapper with Wavelet Decomposition
In `data.py`, subclass your sliding-window dataset:

```python
from wavelet_utils import modwt_decompose
import numpy as np
import torch

class WaveletSlidingWindowDataset(SlidingWindowDataset):
    def __init__(self, data_tensor, history_len, predict_len,
                 mask_prob=0.01, wavelet="db4", level=3):
        super().__init__(data_tensor, history_len, predict_len, mask_prob)
        self.wavelet = wavelet
        self.level = level
        # new feature dimension: original_dim * (level + 1)
        self.coeff_dim = data_tensor.shape[1] * (level + 1)

    def __getitem__(self, idx):
        hist, fut = super().__getitem__(idx)  # shape [T, D]
        hist_w = self._decompose_window(hist)
        fut_w  = self._decompose_window(fut)
        return hist_w, fut_w

    def _decompose_window(self, win: torch.Tensor) -> torch.Tensor:
        arr = win.numpy()  # [T, D]
        out = []
        for d in range(arr.shape[1]):
            coeffs = modwt_decompose(arr[:, d], self.wavelet, self.level)  # (level+1, T)
            out.append(coeffs.T)  # (T, level+1)
        # Concatenate channels → (T, D*(level+1))
        return torch.from_numpy(np.concatenate(out, axis=1).astype(np.float32))
```

## 5. Training Script Modifications
In `trainer.py`, swap in the new dataset and adjust model input:

```diff
- from data import SlidingWindowDataset
+ from data import WaveletSlidingWindowDataset

- dataset = SlidingWindowDataset(
+ dataset = WaveletSlidingWindowDataset(
     data_tensor=train_tensor,
     history_len=history_len,
     predict_len=predict_len,
     mask_prob=CONFIG.get("mask_prob", 0.1),
+    wavelet="db4",
+    level=3
 )

- model = ScoreTransformerNet(
-    input_dim=train_tensor.shape[-1],
+ model = ScoreTransformerNet(
+    input_dim=dataset.coeff_dim,
     history_len=history_len,
     predict_len=predict_len,
     model_dim=CONFIG.get("model_dim", 256)
 ).to(device)
```

## 6. Inference & Post-Processing
In `predictandsave.py`, after sampling, invert wavelets:

```python
from wavelet_utils import imodwt_reconstruct
import numpy as np

# x_pred_paths shape: (n_windows, paths_per_window, predict_len, coeff_dim)
D = original_series_dim
lev = 3
for w, window_preds in enumerate(x_pred_paths):
    rec_paths = []
    for path in window_preds:  # [predict_len, coeff_dim]
        arr = path.cpu().numpy()  # [L, D*(lev+1)]
        series_rec = []
        for d in range(D):
            c = arr[:, d*(lev+1):(d+1)*(lev+1)].T  # (lev+1, L)
            rec = imodwt_reconstruct(c, wavelet="db4")  # [L]
            series_rec.append(rec)
        rec_paths.append(np.stack(series_rec, axis=1))  # [L, D]
    rec_paths = np.stack(rec_paths, axis=0)
    # Convert increments → prices as before
```

## 7. Summary of Steps
1. **Install PyWavelets**: `pip install PyWavelets`.  
2. **Add** `wavelet_utils.py` with `modwt_decompose` and `imodwt_reconstruct`.  
3. **Create** `WaveletSlidingWindowDataset` in `data.py`.  
4. **Update** `trainer.py` and `predictandsave.py` to use the new dataset and adjust `input_dim`.  
5. **After sampling**, reconstruct time-domain paths via `imodwt_reconstruct`.  

Implementing these changes will allow your model to learn and predict in the wavelet domain, capturing multi-scale dynamics before reconstructing a high-resolution forecast.
