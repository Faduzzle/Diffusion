"""
Central registry system for pluggable components.

This module provides a registry pattern that allows components to be dynamically
registered and retrieved by type and name. This enables easy swapping of components
and addition of new implementations without modifying core code.
"""

from typing import Dict, Type, Any, List, Optional, Callable
import inspect
from functools import wraps
import importlib
import pkgutil
from pathlib import Path


class ComponentRegistry:
    """Registry for managing pluggable components."""
    
    def __init__(self):
        # Nested dict: component_type -> component_name -> component_class
        self._registry: Dict[str, Dict[str, Type]] = {}
        # Store component metadata
        self._metadata: Dict[str, Dict[str, Dict[str, Any]]] = {}
        # Component type validation functions
        self._validators: Dict[str, Callable] = {}
    
    def register(
        self,
        component_type: str,
        component_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """
        Decorator to register a component.
        
        Args:
            component_type: Type of component (e.g., 'architecture', 'preprocessing')
            component_name: Unique name for this component
            metadata: Optional metadata about the component
            
        Returns:
            Decorator function
            
        Example:
            @Registry.register("architecture", "transformer")
            class TransformerArchitecture(BaseArchitecture):
                pass
        """
        def decorator(cls: Type) -> Type:
            # Initialize nested dicts if needed
            if component_type not in self._registry:
                self._registry[component_type] = {}
                self._metadata[component_type] = {}
            
            # Check for duplicate registration
            if component_name in self._registry[component_type]:
                raise ValueError(
                    f"Component '{component_name}' already registered for type '{component_type}'"
                )
            
            # Register the component
            self._registry[component_type][component_name] = cls
            
            # Store metadata
            self._metadata[component_type][component_name] = metadata or {}
            # Add automatic metadata
            self._metadata[component_type][component_name].update({
                'class_name': cls.__name__,
                'module': cls.__module__,
                'docstring': cls.__doc__,
                'init_params': self._extract_init_params(cls)
            })
            
            # Validate if validator exists
            if component_type in self._validators:
                self._validators[component_type](cls)
            
            return cls
        
        return decorator
    
    def register_validator(self, component_type: str) -> Callable:
        """
        Register a validator function for a component type.
        
        The validator will be called whenever a new component is registered.
        It should raise an exception if the component is invalid.
        """
        def decorator(validator_func: Callable) -> Callable:
            self._validators[component_type] = validator_func
            return validator_func
        return decorator
    
    def get(
        self,
        component_type: str,
        component_name: str,
        **kwargs
    ) -> Any:
        """
        Get an instance of a registered component.
        
        Args:
            component_type: Type of component
            component_name: Name of component
            **kwargs: Arguments passed to component constructor
            
        Returns:
            Instantiated component
            
        Raises:
            KeyError: If component not found
        """
        if component_type not in self._registry:
            raise KeyError(f"Unknown component type: {component_type}")
        
        if component_name not in self._registry[component_type]:
            available = list(self._registry[component_type].keys())
            raise KeyError(
                f"Unknown {component_type} component: {component_name}. "
                f"Available: {available}"
            )
        
        cls = self._registry[component_type][component_name]
        return cls(**kwargs)
    
    def get_class(
        self,
        component_type: str,
        component_name: str
    ) -> Type:
        """Get the class for a registered component without instantiating it."""
        if component_type not in self._registry:
            raise KeyError(f"Unknown component type: {component_type}")
        
        if component_name not in self._registry[component_type]:
            raise KeyError(f"Unknown {component_type} component: {component_name}")
        
        return self._registry[component_type][component_name]
    
    def list(self, component_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List all registered components.
        
        Args:
            component_type: If specified, only list components of this type
            
        Returns:
            Dictionary mapping component types to lists of component names
        """
        if component_type is not None:
            if component_type not in self._registry:
                return {}
            return {component_type: list(self._registry[component_type].keys())}
        
        return {
            comp_type: list(components.keys())
            for comp_type, components in self._registry.items()
        }
    
    def get_metadata(
        self,
        component_type: str,
        component_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get metadata for a component or all components of a type."""
        if component_type not in self._metadata:
            return {}
        
        if component_name is not None:
            return self._metadata[component_type].get(component_name, {})
        
        return self._metadata[component_type]
    
    def inspect(
        self,
        component_type: str,
        component_name: str
    ) -> Dict[str, Any]:
        """
        Get detailed information about a component.
        
        Returns:
            Dictionary with component details including metadata,
            init parameters, methods, etc.
        """
        cls = self.get_class(component_type, component_name)
        metadata = self.get_metadata(component_type, component_name)
        
        # Get all public methods
        methods = [
            name for name, obj in inspect.getmembers(cls)
            if (inspect.ismethod(obj) or inspect.isfunction(obj)) 
            and not name.startswith('_')
        ]
        
        # Get parent classes
        parents = [base.__name__ for base in cls.__bases__]
        
        return {
            'class': cls,
            'metadata': metadata,
            'methods': methods,
            'parents': parents,
            'module': cls.__module__,
            'file': inspect.getfile(cls),
        }
    
    def _extract_init_params(self, cls: Type) -> Dict[str, Any]:
        """Extract __init__ parameters from a class."""
        try:
            sig = inspect.signature(cls.__init__)
            params = {}
            
            for name, param in sig.parameters.items():
                if name in ('self', 'cls'):
                    continue
                
                param_info = {
                    'default': param.default if param.default != param.empty else None,
                    'annotation': str(param.annotation) if param.annotation != param.empty else None,
                    'kind': str(param.kind)
                }
                params[name] = param_info
            
            return params
        except Exception:
            return {}
    
    def auto_discover(self, package_path: str):
        """
        Automatically discover and load all components in a package.
        
        Args:
            package_path: Path to package containing components
        """
        # Import the package
        package = importlib.import_module(package_path)
        
        # Walk through all submodules
        for importer, modname, ispkg in pkgutil.walk_packages(
            package.__path__,
            prefix=package.__name__ + "."
        ):
            # Skip __pycache__ and test modules
            if '__pycache__' in modname or 'test' in modname:
                continue
            
            try:
                # Import the module (which will trigger @register decorators)
                importlib.import_module(modname)
            except Exception as e:
                print(f"Warning: Failed to import {modname}: {e}")
    
    def create_from_config(
        self,
        component_type: str,
        config: Dict[str, Any]
    ) -> Any:
        """
        Create a component from a configuration dictionary.
        
        Args:
            component_type: Type of component
            config: Configuration with 'type' key and optional 'params'
            
        Returns:
            Instantiated component
        """
        if 'type' not in config:
            raise ValueError(f"Config must contain 'type' key")
        
        component_name = config['type']
        params = config.get('params', {})
        
        return self.get(component_type, component_name, **params)


# Global registry instance
Registry = ComponentRegistry()


# Convenience decorators for common component types
def architecture(name: str, **metadata):
    """Register an architecture component."""
    return Registry.register("architecture", name, metadata)


def preprocessing(name: str, **metadata):
    """Register a preprocessing component."""
    return Registry.register("preprocessing", name, metadata)


def noise(name: str, **metadata):
    """Register a noise component."""
    return Registry.register("noise", name, metadata)


def sde(name: str, **metadata):
    """Register an SDE component."""
    return Registry.register("sde", name, metadata)


def objective(name: str, **metadata):
    """Register a training objective component."""
    return Registry.register("objective", name, metadata)


def sampler(name: str, **metadata):
    """Register a sampler component."""
    return Registry.register("sampler", name, metadata)


# Component validators
@Registry.register_validator("architecture")
def validate_architecture(cls: Type):
    """Validate that architecture has required methods."""
    required_methods = ['forward']
    for method in required_methods:
        if not hasattr(cls, method):
            raise TypeError(f"Architecture {cls.__name__} must implement {method} method")


@Registry.register_validator("preprocessing")
def validate_preprocessing(cls: Type):
    """Validate that preprocessor has required methods."""
    required_methods = ['transform', 'inverse_transform']
    for method in required_methods:
        if not hasattr(cls, method):
            raise TypeError(f"Preprocessor {cls.__name__} must implement {method} method")


@Registry.register_validator("sde")
def validate_sde(cls: Type):
    """Validate that SDE has required methods."""
    required_methods = ['f', 'g', 'transition_kernel', 'prior_sampling']
    for method in required_methods:
        if not hasattr(cls, method):
            raise TypeError(f"SDE {cls.__name__} must implement {method} method")