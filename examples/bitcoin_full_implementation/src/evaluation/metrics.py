"""
Evaluation metrics for time series predictions.

Implements various metrics for assessing the quality of
probabilistic time series predictions.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats


def compute_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Mean Squared Error.
    
    Args:
        predictions: Predicted values (can be mean of samples)
        targets: True values
        
    Returns:
        MSE value
    """
    return np.mean((predictions - targets) ** 2)


def compute_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Mean Absolute Error.
    
    Args:
        predictions: Predicted values
        targets: True values
        
    Returns:
        MAE value
    """
    return np.mean(np.abs(predictions - targets))


def compute_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Root Mean Squared Error.
    
    Args:
        predictions: Predicted values
        targets: True values
        
    Returns:
        RMSE value
    """
    return np.sqrt(compute_mse(predictions, targets))


def compute_mape(predictions: np.ndarray, targets: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.
    
    Args:
        predictions: Predicted values
        targets: True values
        epsilon: Small value to avoid division by zero
        
    Returns:
        MAPE value (as percentage)
    """
    return 100 * np.mean(np.abs((targets - predictions) / (targets + epsilon)))


def compute_directional_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Directional accuracy (correct sign prediction).
    
    Args:
        predictions: Predicted values
        targets: True values
        
    Returns:
        Fraction of correct directional predictions
    """
    # Compute changes
    if len(predictions.shape) == 1:
        pred_direction = np.sign(predictions)
        true_direction = np.sign(targets)
    else:
        # For sequences, compute direction of change
        pred_direction = np.sign(np.diff(predictions, axis=-1))
        true_direction = np.sign(np.diff(targets, axis=-1))
    
    return np.mean(pred_direction == true_direction)


def compute_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Sharpe ratio for returns.
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Annualized Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    
    if excess_returns.std() == 0:
        return 0.0
    
    # Annualized Sharpe ratio
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def compute_quantile_coverage(
    samples: np.ndarray,
    targets: np.ndarray,
    quantiles: List[float] = [0.05, 0.25, 0.75, 0.95]
) -> Dict[float, float]:
    """
    Compute quantile coverage (calibration).
    
    Args:
        samples: Prediction samples (num_samples, ...)
        targets: True values
        quantiles: Quantiles to evaluate
        
    Returns:
        Dictionary mapping quantile to coverage
    """
    coverage = {}
    
    for q in quantiles:
        quantile_val = np.quantile(samples, q, axis=0)
        empirical_coverage = np.mean(targets <= quantile_val)
        coverage[q] = empirical_coverage
    
    return coverage


def compute_pinball_loss(
    samples: np.ndarray,
    targets: np.ndarray,
    quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95]
) -> Dict[float, float]:
    """
    Pinball (quantile) loss for probabilistic predictions.
    
    Args:
        samples: Prediction samples
        targets: True values
        quantiles: Quantiles to evaluate
        
    Returns:
        Dictionary mapping quantile to pinball loss
    """
    losses = {}
    
    for q in quantiles:
        quantile_pred = np.quantile(samples, q, axis=0)
        error = targets - quantile_pred
        
        # Pinball loss
        loss = np.where(error >= 0, q * error, (q - 1) * error)
        losses[q] = np.mean(loss)
    
    return losses


def compute_crps(samples: np.ndarray, targets: np.ndarray) -> float:
    """
    Continuous Ranked Probability Score.
    
    Lower is better. Measures the quality of probabilistic predictions.
    
    Args:
        samples: Prediction samples (num_samples, ...)
        targets: True values
        
    Returns:
        CRPS value
    """
    # Flatten for computation
    samples_flat = samples.reshape(samples.shape[0], -1)
    targets_flat = targets.flatten()
    
    crps_values = []
    
    for i in range(targets_flat.shape[0]):
        # Empirical CDF
        sample_col = samples_flat[:, i]
        crps = np.mean(np.abs(sample_col[:, np.newaxis] - sample_col[np.newaxis, :])) / 2
        crps += np.mean(np.abs(sample_col - targets_flat[i]))
        crps_values.append(crps)
    
    return np.mean(crps_values)


def compute_interval_score(
    lower: np.ndarray,
    upper: np.ndarray,
    targets: np.ndarray,
    alpha: float = 0.1
) -> float:
    """
    Interval score for prediction intervals.
    
    Args:
        lower: Lower bound of prediction interval
        upper: Upper bound of prediction interval
        targets: True values
        alpha: Significance level (e.g., 0.1 for 90% interval)
        
    Returns:
        Interval score
    """
    # Width of interval
    width = upper - lower
    
    # Penalties for being outside interval
    lower_penalty = (2 / alpha) * np.maximum(lower - targets, 0)
    upper_penalty = (2 / alpha) * np.maximum(targets - upper, 0)
    
    return np.mean(width + lower_penalty + upper_penalty)


def compute_log_likelihood(
    samples: np.ndarray,
    targets: np.ndarray,
    bandwidth: Optional[float] = None
) -> float:
    """
    Log likelihood of targets under empirical distribution of samples.
    
    Args:
        samples: Prediction samples
        targets: True values
        bandwidth: KDE bandwidth (auto if None)
        
    Returns:
        Average log likelihood
    """
    # Flatten for KDE
    samples_flat = samples.reshape(samples.shape[0], -1)
    targets_flat = targets.flatten()
    
    log_likelihoods = []
    
    for i in range(targets_flat.shape[0]):
        # Fit KDE to samples
        kde = stats.gaussian_kde(samples_flat[:, i], bw_method=bandwidth)
        
        # Evaluate at target
        log_prob = np.log(kde(targets_flat[i])[0] + 1e-10)
        log_likelihoods.append(log_prob)
    
    return np.mean(log_likelihoods)


class MetricCalculator:
    """Helper class to compute all metrics."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Configuration for metrics
        """
        self.config = config or {}
        self.metrics_to_compute = self.config.get('metrics', [
            'mse', 'mae', 'rmse', 'directional', 'sharpe', 'crps'
        ])
    
    def compute_all_metrics(
        self,
        predictions: Union[np.ndarray, Dict[str, np.ndarray]],
        targets: np.ndarray,
        return_type: str = 'dict'
    ) -> Union[Dict[str, float], np.ndarray]:
        """
        Compute all configured metrics.
        
        Args:
            predictions: Either samples array or dict with 'samples' and 'mean'
            targets: True values
            return_type: 'dict' for dictionary, 'array' for numpy array
            
        Returns:
            Dictionary or array of metric values
        """
        results = {}
        
        # Extract samples and mean
        if isinstance(predictions, dict):
            samples = predictions.get('samples')
            mean_pred = predictions.get('mean')
        else:
            samples = predictions
            mean_pred = np.mean(samples, axis=0) if samples.ndim > targets.ndim else samples
        
        # Point metrics (using mean prediction)
        if 'mse' in self.metrics_to_compute:
            results['mse'] = compute_mse(mean_pred, targets)
        
        if 'mae' in self.metrics_to_compute:
            results['mae'] = compute_mae(mean_pred, targets)
        
        if 'rmse' in self.metrics_to_compute:
            results['rmse'] = compute_rmse(mean_pred, targets)
        
        if 'mape' in self.metrics_to_compute:
            results['mape'] = compute_mape(mean_pred, targets)
        
        if 'directional' in self.metrics_to_compute:
            results['directional_accuracy'] = compute_directional_accuracy(mean_pred, targets)
        
        # Return-based metrics
        if 'sharpe' in self.metrics_to_compute:
            # Convert to returns if needed
            if mean_pred.ndim == 1:
                returns = mean_pred
            else:
                returns = mean_pred.flatten()
            results['sharpe_ratio'] = compute_sharpe_ratio(returns)
        
        # Probabilistic metrics (using samples)
        if samples is not None and samples.ndim > targets.ndim:
            if 'crps' in self.metrics_to_compute:
                results['crps'] = compute_crps(samples, targets)
            
            if 'quantile' in self.metrics_to_compute:
                coverage = compute_quantile_coverage(samples, targets)
                for q, cov in coverage.items():
                    results[f'coverage_{q}'] = cov
            
            if 'pinball' in self.metrics_to_compute:
                pinball = compute_pinball_loss(samples, targets)
                for q, loss in pinball.items():
                    results[f'pinball_{q}'] = loss
            
            if 'interval' in self.metrics_to_compute:
                # 90% prediction interval
                lower = np.quantile(samples, 0.05, axis=0)
                upper = np.quantile(samples, 0.95, axis=0)
                results['interval_score_90'] = compute_interval_score(
                    lower, upper, targets, alpha=0.1
                )
        
        if return_type == 'array':
            return np.array(list(results.values()))
        
        return results
    
    def compute_rolling_metrics(
        self,
        predictions: List[Dict[str, np.ndarray]],
        targets: np.ndarray,
        window_size: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Compute metrics over rolling windows.
        
        Args:
            predictions: List of prediction dictionaries
            targets: True values for all windows
            window_size: Size of each window (inferred if None)
            
        Returns:
            Dictionary with list of metrics for each window
        """
        rolling_metrics = {metric: [] for metric in self.metrics_to_compute}
        
        current_idx = 0
        for pred in predictions:
            if window_size is None:
                window_size = pred['mean'].shape[-1]
            
            window_targets = targets[current_idx:current_idx + window_size]
            
            # Compute metrics for this window
            metrics = self.compute_all_metrics(pred, window_targets)
            
            # Append to rolling results
            for key, value in metrics.items():
                if key in rolling_metrics:
                    rolling_metrics[key].append(value)
            
            current_idx += window_size
        
        return rolling_metrics