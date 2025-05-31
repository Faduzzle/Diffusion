"""
Time Series Diffusion Framework

A modular framework for experimenting with diffusion models on time series data.
"""

__version__ = "0.1.0"

from .src.core.registry import Registry
from .src.core.base import (
    BaseComponent,
    BasePreprocessor,
    BaseArchitecture,
    BaseNoise,
    BaseSDE,
    BaseObjective,
    BaseSampler,
    BaseEvaluator
)

# Convenience imports
from .src.experiments.runner import ExperimentRunner
from .src.experiments.predictor import DiffusionPredictor

__all__ = [
    "Registry",
    "BaseComponent",
    "BasePreprocessor", 
    "BaseArchitecture",
    "BaseNoise",
    "BaseSDE",
    "BaseObjective",
    "BaseSampler",
    "BaseEvaluator",
    "ExperimentRunner",
    "DiffusionPredictor"
]