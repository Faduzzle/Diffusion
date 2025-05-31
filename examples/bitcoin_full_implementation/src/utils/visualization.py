"""Visualization utilities for time series predictions."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import seaborn as sns
from pathlib import Path


def plot_predictions(
    predictions: Dict[str, np.ndarray],
    history: Optional[np.ndarray] = None,
    ground_truth: Optional[np.ndarray] = None,
    title: str = "Bitcoin Price Predictions",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Union[str, Path]] = None
):
    """
    Plot prediction samples with confidence intervals.
    
    Args:
        predictions: Dictionary with 'samples', 'mean', 'quantiles'
        history: Historical data to plot
        ground_truth: True future values
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Time indices
    history_len = history.shape[0] if history is not None else 0
    predict_len = predictions['mean'].shape[0]
    
    history_time = np.arange(-history_len, 0) if history_len > 0 else np.array([])
    future_time = np.arange(predict_len)
    
    # Plot history
    if history is not None:
        ax.plot(history_time, history, 'k-', label='History', linewidth=2)
    
    # Plot predictions
    ax.plot(future_time, predictions['mean'], 'b-', label='Mean Prediction', linewidth=2)
    
    # Plot confidence intervals
    if 'quantiles' in predictions:
        ax.fill_between(
            future_time,
            predictions['quantiles'][0.05],
            predictions['quantiles'][0.95],
            alpha=0.2, color='blue', label='90% CI'
        )
        ax.fill_between(
            future_time,
            predictions['quantiles'][0.25],
            predictions['quantiles'][0.75],
            alpha=0.3, color='blue', label='50% CI'
        )
    
    # Plot sample trajectories
    if 'samples' in predictions and predictions['samples'].shape[0] > 1:
        n_samples_to_plot = min(5, predictions['samples'].shape[0])
        for i in range(n_samples_to_plot):
            ax.plot(
                future_time,
                predictions['samples'][i],
                'b--', alpha=0.3, linewidth=0.5
            )
    
    # Plot ground truth
    if ground_truth is not None:
        ax.plot(future_time, ground_truth, 'r-', label='Ground Truth', linewidth=2)
    
    # Formatting
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    title: str = "Model Performance Comparison",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Union[str, Path]] = None
):
    """
    Plot comparison of metrics across models.
    
    Args:
        metrics_dict: Dictionary mapping model names to metrics
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    # Extract metrics and models
    models = list(metrics_dict.keys())
    metrics = list(next(iter(metrics_dict.values())).keys())
    
    # Create subplot for each metric
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = [metrics_dict[model].get(metric, 0) for model in models]
        
        bars = ax.bar(models, values)
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylabel('Value')
        
        # Color bars based on whether lower is better
        lower_is_better = metric in ['mse', 'mae', 'rmse', 'crps', 'pinball', 'interval_score']
        colors = ['red' if v == min(values) else 'blue' for v in values] if lower_is_better else \
                ['green' if v == max(values) else 'blue' for v in values]
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
    
    # Remove empty subplots
    for i in range(n_metrics, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_rolling_metrics(
    rolling_metrics: Dict[str, List[float]],
    window_labels: Optional[List[str]] = None,
    title: str = "Rolling Window Metrics",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Union[str, Path]] = None
):
    """
    Plot metrics over rolling windows.
    
    Args:
        rolling_metrics: Dictionary mapping metric names to lists of values
        window_labels: Labels for each window
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    n_metrics = len(rolling_metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for i, (metric, values) in enumerate(rolling_metrics.items()):
        ax = axes[i]
        
        x = np.arange(len(values))
        ax.plot(x, values, 'o-', linewidth=2, markersize=6)
        
        # Add trend line
        if len(values) > 3:
            z = np.polyfit(x, values, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), 'r--', alpha=0.5, label=f'Trend: {z[0]:.3f}')
        
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel('Window')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        if window_labels:
            ax.set_xticks(x)
            ax.set_xticklabels(window_labels, rotation=45)
    
    # Remove empty subplots
    for i in range(len(rolling_metrics), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_prediction_distribution(
    samples: np.ndarray,
    time_steps: List[int] = None,
    title: str = "Prediction Distribution",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Union[str, Path]] = None
):
    """
    Plot distribution of predictions at specific time steps.
    
    Args:
        samples: Prediction samples (num_samples, time_steps)
        time_steps: Specific time steps to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    if time_steps is None:
        time_steps = [0, samples.shape[1] // 2, samples.shape[1] - 1]
    
    n_steps = len(time_steps)
    fig, axes = plt.subplots(1, n_steps, figsize=figsize)
    axes = axes if n_steps > 1 else [axes]
    
    for i, t in enumerate(time_steps):
        ax = axes[i]
        
        # Plot histogram
        ax.hist(samples[:, t], bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
        
        # Plot KDE
        from scipy import stats
        kde = stats.gaussian_kde(samples[:, t])
        x_range = np.linspace(samples[:, t].min(), samples[:, t].max(), 100)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        # Add statistics
        mean = samples[:, t].mean()
        std = samples[:, t].std()
        ax.axvline(mean, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean:.3f}')
        ax.axvline(mean - std, color='orange', linestyle=':', linewidth=1)
        ax.axvline(mean + std, color='orange', linestyle=':', linewidth=1, label=f'±1 SD')
        
        ax.set_title(f'Time Step {t+1}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_attention_weights(
    attention_weights: np.ndarray,
    history_len: int,
    predict_len: int,
    title: str = "Attention Weights",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None
):
    """
    Visualize attention weights from the model.
    
    Args:
        attention_weights: Attention weight matrix
        history_len: Length of history sequence
        predict_len: Length of prediction sequence
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        attention_weights,
        cmap='Blues',
        cbar=True,
        square=True,
        xticklabels=False,
        yticklabels=False,
        ax=ax
    )
    
    # Add labels
    ax.set_xlabel('Key (History)')
    ax.set_ylabel('Query (Future)')
    ax.set_title(title)
    
    # Add separating lines
    ax.axhline(y=0, color='black', linewidth=2)
    ax.axhline(y=predict_len, color='black', linewidth=2)
    ax.axvline(x=0, color='black', linewidth=2)
    ax.axvline(x=history_len, color='black', linewidth=2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()