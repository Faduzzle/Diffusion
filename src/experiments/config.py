"""
Experiment configuration system.

Handles loading, validation, and management of experiment configurations.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
from dataclasses import dataclass, field
import copy
from datetime import datetime

from ..core import Registry


@dataclass
class ExperimentConfig:
    """Structured experiment configuration."""
    
    # Experiment metadata
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Component configurations
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Data configuration
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Training configuration
    training: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation configuration
    evaluation: Dict[str, Any] = field(default_factory=dict)
    
    # Output configuration
    output: Dict[str, Any] = field(default_factory=dict)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set defaults and validate."""
        # Set creation timestamp
        if 'created_at' not in self.metadata:
            self.metadata['created_at'] = datetime.now().isoformat()
        
        # Set default output directory
        if 'output_dir' not in self.output:
            self.output['output_dir'] = f"experiments/{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Validate component configurations
        self._validate_components()
    
    def _validate_components(self):
        """Validate that all required components are specified."""
        required_components = ['preprocessing', 'architecture', 'sde', 'objective', 'sampler']
        
        for comp_type in required_components:
            if comp_type not in self.components:
                raise ValueError(f"Missing required component: {comp_type}")
            
            if 'type' not in self.components[comp_type]:
                raise ValueError(f"Component {comp_type} must specify 'type'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'tags': self.tags,
            'components': self.components,
            'data': self.data,
            'training': self.training,
            'evaluation': self.evaluation,
            'output': self.output,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: Union[str, Path]):
        """Save configuration to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            with open(path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from file."""
        path = Path(path)
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        else:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    def create_components(self) -> Dict[str, Any]:
        """Create all components from configuration."""
        components = {}
        
        for comp_type, comp_config in self.components.items():
            components[comp_type] = Registry.create_from_config(comp_type, comp_config)
        
        return components
    
    def merge(self, other: Union['ExperimentConfig', Dict[str, Any]]) -> 'ExperimentConfig':
        """Merge with another configuration."""
        if isinstance(other, ExperimentConfig):
            other_dict = other.to_dict()
        else:
            other_dict = other
        
        merged_dict = deep_merge(self.to_dict(), other_dict)
        return ExperimentConfig.from_dict(merged_dict)


class ConfigBuilder:
    """Builder for creating experiment configurations."""
    
    def __init__(self, name: str):
        self.config = {
            'name': name,
            'description': '',
            'tags': [],
            'components': {},
            'data': {},
            'training': {},
            'evaluation': {},
            'output': {},
            'metadata': {}
        }
    
    def description(self, desc: str) -> 'ConfigBuilder':
        """Set description."""
        self.config['description'] = desc
        return self
    
    def tags(self, *tags: str) -> 'ConfigBuilder':
        """Add tags."""
        self.config['tags'].extend(tags)
        return self
    
    def component(self, comp_type: str, comp_name: str, **params) -> 'ConfigBuilder':
        """Add a component."""
        self.config['components'][comp_type] = {
            'type': comp_name,
            'params': params
        }
        return self
    
    def data(self, **data_config) -> 'ConfigBuilder':
        """Set data configuration."""
        self.config['data'].update(data_config)
        return self
    
    def training(self, **training_config) -> 'ConfigBuilder':
        """Set training configuration."""
        self.config['training'].update(training_config)
        return self
    
    def evaluation(self, **eval_config) -> 'ConfigBuilder':
        """Set evaluation configuration."""
        self.config['evaluation'].update(eval_config)
        return self
    
    def output(self, **output_config) -> 'ConfigBuilder':
        """Set output configuration."""
        self.config['output'].update(output_config)
        return self
    
    def build(self) -> ExperimentConfig:
        """Build the configuration."""
        return ExperimentConfig.from_dict(self.config)


class ConfigTemplate:
    """Pre-defined configuration templates."""
    
    @staticmethod
    def minimal(name: str) -> ExperimentConfig:
        """Minimal configuration for quick experiments."""
        return (ConfigBuilder(name)
                .description("Minimal experiment configuration")
                .component("preprocessing", "standard")
                .component("architecture", "transformer", model_dim=128, num_layers=4)
                .component("sde", "vpsde")
                .component("objective", "score_matching")
                .component("sampler", "euler")
                .data(history_len=100, predict_len=10)
                .training(batch_size=32, num_epochs=50, learning_rate=1e-4)
                .evaluation(num_samples=100, num_steps=1000)
                .build())
    
    @staticmethod
    def bitcoin(name: str) -> ExperimentConfig:
        """Configuration optimized for Bitcoin data."""
        return (ConfigBuilder(name)
                .description("Bitcoin price diffusion experiment")
                .tags("bitcoin", "financial", "crypto")
                .component("preprocessing", "log_returns", normalize=True)
                .component("architecture", "transformer", 
                          model_dim=256, num_heads=8, num_layers=6,
                          history_len=252, predict_len=21)
                .component("sde", "vpsde", beta_min=0.1, beta_max=20.0)
                .component("objective", "score_matching", loss_weight_type="uniform")
                .component("sampler", "euler")
                .data(
                    dataset="bitcoin",
                    history_len=252,  # 1 year
                    predict_len=21,   # 1 month
                    train_ratio=0.8,
                    val_ratio=0.1
                )
                .training(
                    batch_size=32,
                    num_epochs=100,
                    learning_rate=1e-4,
                    gradient_clip=1.0,
                    use_ema=True,
                    ema_decay=0.999
                )
                .evaluation(
                    num_samples=100,
                    num_steps=1000,
                    guidance_scale=2.0,
                    metrics=["mse", "mae", "sharpe", "directional"]
                )
                .build())
    
    @staticmethod
    def multivariate(name: str, num_series: int = 5) -> ExperimentConfig:
        """Configuration for multivariate time series."""
        return (ConfigBuilder(name)
                .description(f"Multivariate time series with {num_series} series")
                .tags("multivariate")
                .component("preprocessing", "standard")
                .component("architecture", "transformer",
                          input_dim=num_series,
                          model_dim=256,
                          num_heads=8,
                          num_layers=6)
                .component("sde", "vpsde")
                .component("objective", "score_matching")
                .component("sampler", "euler")
                .data(
                    num_series=num_series,
                    history_len=100,
                    predict_len=20
                )
                .training(
                    batch_size=16,
                    num_epochs=100,
                    learning_rate=1e-4
                )
                .evaluation(
                    num_samples=50,
                    num_steps=1000
                )
                .build())


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    
    return result


def validate_config(config: Union[ExperimentConfig, Dict[str, Any]]) -> List[str]:
    """
    Validate a configuration and return list of issues.
    
    Returns:
        List of validation error messages (empty if valid)
    """
    if isinstance(config, dict):
        config = ExperimentConfig.from_dict(config)
    
    issues = []
    
    # Check component availability
    for comp_type, comp_config in config.components.items():
        comp_name = comp_config.get('type')
        if comp_name:
            available = Registry.list(comp_type).get(comp_type, [])
            if comp_name not in available:
                issues.append(f"Unknown {comp_type}: {comp_name}. Available: {available}")
    
    # Check data configuration
    if 'history_len' not in config.data:
        issues.append("Missing required data.history_len")
    if 'predict_len' not in config.data:
        issues.append("Missing required data.predict_len")
    
    # Check training configuration
    if 'batch_size' not in config.training:
        issues.append("Missing required training.batch_size")
    if 'num_epochs' not in config.training:
        issues.append("Missing required training.num_epochs")
    
    return issues