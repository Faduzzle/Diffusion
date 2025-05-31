"""
Experiment runner for time series diffusion models.

Handles the complete experiment lifecycle: data loading, model training,
evaluation, and result logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import json
import time
from datetime import datetime
import numpy as np
from tqdm import tqdm

from .config import ExperimentConfig
from ..core import Registry
from ..data.dataset import create_dataset
from ..training.trainer import DiffusionTrainer
from ..evaluation.evaluator import DiffusionEvaluator
from ..utils.logging import setup_logging, get_logger


class ExperimentRunner:
    """Manages end-to-end experiment execution."""
    
    def __init__(self, config: Union[ExperimentConfig, Dict[str, Any], str, Path]):
        """
        Initialize runner with configuration.
        
        Args:
            config: ExperimentConfig object, config dict, or path to config file
        """
        # Load configuration
        if isinstance(config, (str, Path)):
            self.config = ExperimentConfig.load(config)
        elif isinstance(config, dict):
            self.config = ExperimentConfig.from_dict(config)
        else:
            self.config = config
        
        # Setup output directory
        self.output_dir = Path(self.config.output.get('output_dir', f'experiments/{self.config.name}'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        self.config.save(self.output_dir / 'config.yaml')
        
        # Setup logging
        self.logger = setup_logging(
            self.output_dir / 'experiment.log',
            name=f'experiment.{self.config.name}'
        )
        
        # Initialize components
        self.components = None
        self.model = None
        self.trainer = None
        self.evaluator = None
        
        # Results storage
        self.results = {
            'config': self.config.to_dict(),
            'metrics': {},
            'timings': {},
            'artifacts': {}
        }
    
    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> 'ExperimentRunner':
        """Create runner from config file."""
        return cls(config_path)
    
    def setup(self):
        """Setup all components for the experiment."""
        self.logger.info(f"Setting up experiment: {self.config.name}")
        
        # Create components
        self.logger.info("Creating components...")
        self.components = self.config.create_components()
        
        # Log component info
        for comp_type, component in self.components.items():
            self.logger.info(f"  {comp_type}: {component.__class__.__name__}")
        
        # Setup model
        self._setup_model()
        
        # Setup data
        self._setup_data()
        
        # Setup trainer
        self._setup_trainer()
        
        # Setup evaluator
        self._setup_evaluator()
        
        self.logger.info("Setup complete")
    
    def _setup_model(self):
        """Setup the diffusion model."""
        # Get architecture
        architecture = self.components['architecture']
        
        # Wrap with diffusion-specific layers if needed
        self.model = architecture
        
        # Log model info
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters: {num_params:,}")
    
    def _setup_data(self):
        """Setup datasets and dataloaders."""
        self.logger.info("Loading data...")
        
        # Create datasets
        train_dataset, val_dataset, test_dataset = create_dataset(
            self.config.data,
            preprocessor=self.components['preprocessing']
        )
        
        # Create dataloaders
        batch_size = self.config.training.get('batch_size', 32)
        num_workers = self.config.training.get('num_workers', 4)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ) if val_dataset is not None else None
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ) if test_dataset is not None else None
        
        self.logger.info(f"  Train samples: {len(train_dataset)}")
        if val_dataset:
            self.logger.info(f"  Val samples: {len(val_dataset)}")
        if test_dataset:
            self.logger.info(f"  Test samples: {len(test_dataset)}")
    
    def _setup_trainer(self):
        """Setup the trainer."""
        self.trainer = DiffusionTrainer(
            model=self.model,
            sde=self.components['sde'],
            objective=self.components['objective'],
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=self.config.training,
            output_dir=self.output_dir / 'checkpoints'
        )
    
    def _setup_evaluator(self):
        """Setup the evaluator."""
        self.evaluator = DiffusionEvaluator(
            model=self.model,
            sde=self.components['sde'],
            sampler=self.components['sampler'],
            preprocessor=self.components['preprocessing'],
            config=self.config.evaluation,
            output_dir=self.output_dir / 'evaluation'
        )
    
    def train(self) -> Dict[str, Any]:
        """Run training phase."""
        self.logger.info("Starting training...")
        start_time = time.time()
        
        # Train model
        num_epochs = self.config.training.get('num_epochs', 100)
        history = self.trainer.train(num_epochs)
        
        # Record results
        train_time = time.time() - start_time
        self.results['timings']['train'] = train_time
        self.results['metrics']['train'] = history
        
        self.logger.info(f"Training completed in {train_time:.2f} seconds")
        
        # Save training curves
        self._save_training_curves(history)
        
        return history
    
    def evaluate(self, use_test: bool = True) -> Dict[str, Any]:
        """Run evaluation phase."""
        self.logger.info("Starting evaluation...")
        start_time = time.time()
        
        # Choose dataset
        eval_loader = self.test_loader if use_test and self.test_loader else self.val_loader
        
        if eval_loader is None:
            self.logger.warning("No evaluation data available")
            return {}
        
        # Evaluate model
        eval_results = self.evaluator.evaluate(
            eval_loader,
            num_samples=self.config.evaluation.get('num_samples', 100)
        )
        
        # Record results
        eval_time = time.time() - start_time
        self.results['timings']['evaluate'] = eval_time
        self.results['metrics']['evaluation'] = eval_results
        
        self.logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
        
        # Log key metrics
        for metric_name, metric_value in eval_results.get('aggregate_metrics', {}).items():
            self.logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        return eval_results
    
    def run(self) -> Dict[str, Any]:
        """Run complete experiment."""
        self.logger.info("="*50)
        self.logger.info(f"Running experiment: {self.config.name}")
        self.logger.info(f"Description: {self.config.description}")
        self.logger.info("="*50)
        
        total_start = time.time()
        
        try:
            # Setup
            setup_start = time.time()
            self.setup()
            self.results['timings']['setup'] = time.time() - setup_start
            
            # Training
            if self.config.training.get('skip_training', False):
                self.logger.info("Skipping training (loading checkpoint)")
                self._load_checkpoint()
            else:
                self.train()
            
            # Evaluation
            if not self.config.evaluation.get('skip_evaluation', False):
                self.evaluate()
            
            # Generate samples
            if self.config.evaluation.get('generate_samples', True):
                self._generate_samples()
            
            # Total time
            self.results['timings']['total'] = time.time() - total_start
            
            # Save results
            self._save_results()
            
            self.logger.info("="*50)
            self.logger.info("Experiment completed successfully!")
            self.logger.info(f"Results saved to: {self.output_dir}")
            self.logger.info("="*50)
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}", exc_info=True)
            self.results['error'] = str(e)
            self._save_results()
            raise
        
        return self.results
    
    def _generate_samples(self):
        """Generate sample predictions."""
        self.logger.info("Generating sample predictions...")
        
        # Get a few test samples
        if self.test_loader:
            batch = next(iter(self.test_loader))
        elif self.val_loader:
            batch = next(iter(self.val_loader))
        else:
            self.logger.warning("No data available for sample generation")
            return
        
        # Generate predictions
        num_vis_samples = min(5, batch['history'].shape[0])
        
        predictions = self.evaluator.generate_predictions(
            history=batch['history'][:num_vis_samples],
            num_samples=self.config.evaluation.get('num_samples', 100)
        )
        
        # Save visualizations
        vis_dir = self.output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        from ..utils.visualization import plot_predictions
        
        for i in range(num_vis_samples):
            plot_predictions(
                history=batch['history'][i],
                future=batch['future'][i] if 'future' in batch else None,
                predictions=predictions[i],
                save_path=vis_dir / f'sample_{i}.png'
            )
        
        self.logger.info(f"Saved {num_vis_samples} sample visualizations")
    
    def _save_training_curves(self, history: Dict[str, Any]):
        """Save training curves."""
        from ..utils.visualization import plot_training_curves
        
        plot_training_curves(
            history,
            save_path=self.output_dir / 'training_curves.png'
        )
    
    def _save_results(self):
        """Save all experiment results."""
        # Save as JSON
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary
        summary = {
            'name': self.config.name,
            'description': self.config.description,
            'completed_at': datetime.now().isoformat(),
            'total_time': self.results['timings'].get('total', 0),
            'key_metrics': {}
        }
        
        # Extract key metrics
        if 'evaluation' in self.results['metrics']:
            eval_metrics = self.results['metrics']['evaluation'].get('aggregate_metrics', {})
            for metric in ['mse', 'mae', 'crps', 'sharpe_ratio']:
                if metric in eval_metrics:
                    summary['key_metrics'][metric] = eval_metrics[metric]
        
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _load_checkpoint(self):
        """Load model from checkpoint."""
        checkpoint_path = self.config.training.get('checkpoint_path')
        if not checkpoint_path:
            # Try to find latest checkpoint
            checkpoint_dir = self.output_dir / 'checkpoints'
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob('*.pt'))
                if checkpoints:
                    checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        if checkpoint_path:
            self.logger.info(f"Loading checkpoint: {checkpoint_path}")
            self.trainer.load_checkpoint(checkpoint_path)
        else:
            raise ValueError("No checkpoint found to load")


class ExperimentComparator:
    """Compare multiple experiments."""
    
    def __init__(self, experiment_dirs: List[Union[str, Path]]):
        """
        Initialize comparator with experiment directories.
        
        Args:
            experiment_dirs: List of experiment output directories
        """
        self.experiment_dirs = [Path(d) for d in experiment_dirs]
        self.experiments = []
        
        # Load experiment results
        for exp_dir in self.experiment_dirs:
            if (exp_dir / 'results.json').exists():
                with open(exp_dir / 'results.json', 'r') as f:
                    results = json.load(f)
                    results['dir'] = exp_dir
                    self.experiments.append(results)
    
    def compare_metrics(self, metrics: List[str]) -> Dict[str, Dict[str, float]]:
        """Compare specific metrics across experiments."""
        comparison = {}
        
        for exp in self.experiments:
            exp_name = exp['config']['name']
            comparison[exp_name] = {}
            
            # Extract metrics
            eval_metrics = exp.get('metrics', {}).get('evaluation', {}).get('aggregate_metrics', {})
            
            for metric in metrics:
                if metric in eval_metrics:
                    comparison[exp_name][metric] = eval_metrics[metric]
        
        return comparison
    
    def generate_report(self, save_path: Union[str, Path]):
        """Generate comparison report."""
        from ..utils.visualization import create_comparison_table
        
        # Get all available metrics
        all_metrics = set()
        for exp in self.experiments:
            eval_metrics = exp.get('metrics', {}).get('evaluation', {}).get('aggregate_metrics', {})
            all_metrics.update(eval_metrics.keys())
        
        # Compare all metrics
        comparison = self.compare_metrics(list(all_metrics))
        
        # Create report
        report = {
            'comparison_date': datetime.now().isoformat(),
            'num_experiments': len(self.experiments),
            'experiments': [exp['config']['name'] for exp in self.experiments],
            'metrics': comparison,
            'best_performers': {}
        }
        
        # Find best performers for each metric
        for metric in all_metrics:
            best_value = None
            best_exp = None
            
            for exp_name, exp_metrics in comparison.items():
                if metric in exp_metrics:
                    value = exp_metrics[metric]
                    # Assume lower is better for error metrics
                    if best_value is None or value < best_value:
                        best_value = value
                        best_exp = exp_name
            
            if best_exp:
                report['best_performers'][metric] = {
                    'experiment': best_exp,
                    'value': best_value
                }
        
        # Save report
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create visualization
        create_comparison_table(
            comparison,
            save_path=save_path.with_suffix('.png')
        )
        
        return report