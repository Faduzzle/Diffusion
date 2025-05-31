"""Configuration management utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any, Union
import json


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve relative paths
    if 'checkpoint_dir' in config.get('training', {}):
        config['training']['checkpoint_dir'] = str(
            config_path.parent / config['training']['checkpoint_dir']
        )
    
    return config


def save_config(config: Dict[str, Any], save_path: Union[str, Path]):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations, with override taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    import copy
    
    result = copy.deepcopy(base_config)
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def config_to_args_string(config: Dict[str, Any], prefix: str = '') -> str:
    """
    Convert configuration to command line arguments string.
    
    Args:
        config: Configuration dictionary
        prefix: Prefix for nested keys
        
    Returns:
        Command line arguments string
    """
    args = []
    
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            args.append(config_to_args_string(value, full_key))
        elif isinstance(value, list):
            args.append(f"--{full_key}={','.join(map(str, value))}")
        elif isinstance(value, bool):
            if value:
                args.append(f"--{full_key}")
        else:
            args.append(f"--{full_key}={value}")
    
    return ' '.join(filter(None, args))


class ConfigValidator:
    """Validate configuration against schema."""
    
    @staticmethod
    def validate_model_config(config: Dict[str, Any]):
        """Validate model configuration."""
        required_keys = ['type', 'input_dim', 'model_dim', 'history_len', 'predict_len']
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required model config key: {key}")
        
        if config['history_len'] <= 0:
            raise ValueError("history_len must be positive")
        
        if config['predict_len'] <= 0:
            raise ValueError("predict_len must be positive")
    
    @staticmethod
    def validate_training_config(config: Dict[str, Any]):
        """Validate training configuration."""
        if config.get('batch_size', 1) <= 0:
            raise ValueError("batch_size must be positive")
        
        if config.get('learning_rate', 1e-4) <= 0:
            raise ValueError("learning_rate must be positive")
        
        if config.get('num_epochs', 1) <= 0:
            raise ValueError("num_epochs must be positive")
    
    @staticmethod
    def validate_full_config(config: Dict[str, Any]):
        """Validate full configuration."""
        if 'model' in config:
            ConfigValidator.validate_model_config(config['model'])
        
        if 'training' in config:
            ConfigValidator.validate_training_config(config['training'])
        
        if 'sde' not in config:
            raise ValueError("Missing required config section: sde")
        
        if 'data' not in config:
            raise ValueError("Missing required config section: data")