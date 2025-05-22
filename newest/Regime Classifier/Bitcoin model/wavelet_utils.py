import numpy as np
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

def modwt_decompose(x: np.ndarray, wavelet: str, level: int) -> np.ndarray:
    """
    Implements Maximal Overlap Discrete Wavelet Transform (MODWT).
    This is a non-decimated version of DWT that maintains time resolution at each level.
    
    Args:
        x: Input signal of shape (N,)
        wavelet: Wavelet type (e.g., 'db4')
        level: Number of decomposition levels
    
    Returns:
        Array of shape (level+1, N) containing [cA_level, cD_level, ..., cD1]
    """
    # Get wavelet filters
    wav = pywt.Wavelet(wavelet)
    filters = wav.filter_bank
    h = filters[0]  # Low-pass
    g = filters[1]  # High-pass
    
    # Normalize filters for MODWT
    h_tilde = h / np.sqrt(2)
    g_tilde = g / np.sqrt(2)
    
    # Initialize output
    N = len(x)
    output = np.zeros((level + 1, N))
    
    # Current signal to decompose
    v_j = x.copy()
    
    for j in range(level):
        # Calculate step size
        step = 2**j
        
        # Periodic padding for convolution
        v_j_padded = np.pad(v_j, (len(h) - 1, 0), mode='wrap')
        
        # Apply filters
        w_jp1 = np.zeros(N)  # Details
        v_jp1 = np.zeros(N)  # Approximation
        
        for k in range(N):
            # Circular convolution
            idx = np.arange(len(h)) * step
            idx = idx + k
            idx = idx % N
            
            # High-pass (details)
            w_jp1[k] = np.sum(g_tilde * v_j_padded[idx])
            # Low-pass (approximation)
            v_jp1[k] = np.sum(h_tilde * v_j_padded[idx])
        
        # Store details
        output[j + 1] = w_jp1
        # Update for next level
        v_j = v_jp1
    
    # Store final approximation
    output[0] = v_j
    
    return output

def imodwt_reconstruct(coeffs: np.ndarray, wavelet: str) -> np.ndarray:
    """
    Implements Inverse Maximal Overlap Discrete Wavelet Transform (IMODWT).
    
    Args:
        coeffs: Array of shape (level+1, N) containing [cA_level, cD_level, ..., cD1]
        wavelet: Wavelet type (e.g., 'db4')
    
    Returns:
        Reconstructed signal of shape (N,)
    """
    level = coeffs.shape[0] - 1
    N = coeffs.shape[1]
    
    # Get wavelet filters
    wav = pywt.Wavelet(wavelet)
    filters = wav.filter_bank
    h = filters[0]  # Low-pass
    g = filters[1]  # High-pass
    
    # Normalize filters for IMODWT
    h_tilde = h / np.sqrt(2)
    g_tilde = g / np.sqrt(2)
    
    # Start with final approximation
    v_j = coeffs[0].copy()
    
    # Reconstruct from each level
    for j in range(level - 1, -1, -1):
        # Calculate step size
        step = 2**j
        
        # Get details at current level
        w_jp1 = coeffs[j + 1]
        
        # Periodic padding
        v_j_padded = np.pad(v_j, (len(h) - 1, 0), mode='wrap')
        w_jp1_padded = np.pad(w_jp1, (len(g) - 1, 0), mode='wrap')
        
        # Reconstruction
        v_j_new = np.zeros(N)
        
        for k in range(N):
            # Circular convolution
            idx = np.arange(len(h)) * step
            idx = idx + k
            idx = idx % N
            
            # Combine low and high frequencies
            v_j_new[k] = (np.sum(h_tilde * v_j_padded[idx]) + 
                         np.sum(g_tilde * w_jp1_padded[idx]))
        
        v_j = v_j_new
    
    return v_j

def plot_wavelet_decomposition(signal: np.ndarray, coeffs: np.ndarray, 
                             wavelet: str, level: int, 
                             title: str = "Wavelet Decomposition") -> plt.Figure:
    """
    Creates a visualization of the wavelet decomposition with the original signal
    and all wavelet coefficients.
    """
    n_plots = level + 2  # original signal + approximation + details
    fig, axes = plt.subplots(n_plots, 1, figsize=(15, 3*n_plots))
    fig.suptitle(title, fontsize=16)
    
    # Plot original signal
    axes[0].plot(signal, 'b', label='Original Signal')
    axes[0].set_title('Original Signal')
    axes[0].grid(True)
    
    # Plot approximation coefficient
    axes[1].plot(coeffs[0], 'r', label=f'Approximation (Level {level})')
    axes[1].set_title(f'Approximation Coefficient (Level {level})')
    axes[1].grid(True)
    
    # Plot detail coefficients
    for i in range(level):
        axes[i+2].plot(coeffs[i+1], 'g', label=f'Detail Level {level-i}')
        axes[i+2].set_title(f'Detail Coefficient Level {level-i}')
        axes[i+2].grid(True)
    
    plt.tight_layout()
    return fig

def plot_wavelet_prediction_comparison(original: np.ndarray, 
                                     predicted: np.ndarray,
                                     coeffs_orig: np.ndarray,
                                     coeffs_pred: np.ndarray,
                                     level: int,
                                     title: str = "Prediction Comparison") -> plt.Figure:
    """
    Creates a comparison visualization between original and predicted signals
    in both time and wavelet domains.
    """
    n_plots = level + 2
    fig, axes = plt.subplots(n_plots, 2, figsize=(20, 3*n_plots))
    fig.suptitle(title, fontsize=16)
    
    # Time domain comparison
    axes[0, 0].plot(original, 'b', label='Original')
    axes[0, 0].plot(predicted, 'r--', label='Predicted')
    axes[0, 0].set_title('Time Domain Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Error plot
    error = np.abs(original - predicted)
    axes[0, 1].plot(error, 'r', label='Absolute Error')
    axes[0, 1].fill_between(range(len(error)), 0, error, alpha=0.3)
    axes[0, 1].set_title('Prediction Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Wavelet domain comparisons
    for i in range(level + 1):
        axes[i+1, 0].plot(coeffs_orig[i], 'b', label='Original', alpha=0.7)
        axes[i+1, 0].plot(coeffs_pred[i], 'r--', label='Predicted', alpha=0.7)
        axes[i+1, 0].set_title(f'{"Approximation" if i==0 else f"Detail Level {level-i+1}"}')
        axes[i+1, 0].legend()
        axes[i+1, 0].grid(True)
        
        # Coefficient error
        coeff_error = np.abs(coeffs_orig[i] - coeffs_pred[i])
        axes[i+1, 1].plot(coeff_error, 'r', label='Coefficient Error')
        axes[i+1, 1].fill_between(range(len(coeff_error)), 0, coeff_error, alpha=0.3)
        axes[i+1, 1].set_title(f'Coefficient Error')
        axes[i+1, 1].legend()
        axes[i+1, 1].grid(True)
    
    plt.tight_layout()
    return fig

def plot_wavelet_spectrum(coeffs: np.ndarray, 
                         sampling_rate: float = 1.0,
                         title: str = "Wavelet Power Spectrum") -> plt.Figure:
    """
    Creates a visualization of the wavelet power spectrum across scales.
    """
    level = coeffs.shape[0] - 1
    
    # Compute power at each level
    power = [np.mean(np.square(c)) for c in coeffs]
    
    # Create frequency bands
    freq_bands = [f"A{level}"] + [f"D{i}" for i in range(level, 0, -1)]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(title, fontsize=16)
    
    # Bar plot of power distribution
    ax1.bar(freq_bands, power)
    ax1.set_title("Power Distribution Across Scales")
    ax1.set_xlabel("Wavelet Coefficient")
    ax1.set_ylabel("Power")
    ax1.grid(True)
    
    # Heatmap of coefficient magnitudes over time
    time_steps = coeffs.shape[1]
    magnitude = np.abs(coeffs)
    sns.heatmap(magnitude, ax=ax2, cmap='viridis',
                xticklabels=time_steps//10,
                yticklabels=freq_bands)
    ax2.set_title("Coefficient Magnitude Over Time")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Scale")
    
    plt.tight_layout()
    return fig

def analyze_wavelet_prediction(original: np.ndarray,
                             predicted: np.ndarray,
                             wavelet: str = "db4",
                             level: int = 3,
                             save_dir: str = None) -> dict:
    """
    Performs comprehensive wavelet analysis of original and predicted signals,
    generating visualizations and computing metrics.
    
    Returns:
        dict: Dictionary containing wavelet-domain metrics
    """
    # Compute wavelet decompositions
    coeffs_orig = modwt_decompose(original, wavelet, level)
    coeffs_pred = modwt_decompose(predicted, wavelet, level)
    
    # Generate visualizations
    figs = {
        'decomp_orig': plot_wavelet_decomposition(
            original, coeffs_orig, wavelet, level, 
            "Original Signal Decomposition"
        ),
        'decomp_pred': plot_wavelet_decomposition(
            predicted, coeffs_pred, wavelet, level,
            "Predicted Signal Decomposition"
        ),
        'comparison': plot_wavelet_prediction_comparison(
            original, predicted, coeffs_orig, coeffs_pred, level
        ),
        'spectrum_orig': plot_wavelet_spectrum(
            coeffs_orig, title="Original Signal Wavelet Spectrum"
        ),
        'spectrum_pred': plot_wavelet_spectrum(
            coeffs_pred, title="Predicted Signal Wavelet Spectrum"
        )
    }
    
    # Save figures if directory provided
    if save_dir:
        for name, fig in figs.items():
            fig.savefig(f"{save_dir}/wavelet_{name}.png", 
                       bbox_inches='tight', dpi=300)
            plt.close(fig)
    
    # Compute metrics
    metrics = {
        'mse_by_level': [
            np.mean(np.square(co - cp)) 
            for co, cp in zip(coeffs_orig, coeffs_pred)
        ],
        'correlation_by_level': [
            np.corrcoef(co, cp)[0,1]
            for co, cp in zip(coeffs_orig, coeffs_pred)
        ],
        'power_ratio': [
            np.mean(np.square(cp)) / np.mean(np.square(co))
            for co, cp in zip(coeffs_orig, coeffs_pred)
        ]
    }
    
    return metrics 