"""Configuration management utilities."""

from typing import Any, Dict, Optional
from pathlib import Path
import yaml
import json
from omegaconf import DictConfig, OmegaConf


class ConfigManager:
    """Configuration management for experiments."""
    
    @staticmethod
    def load_config(config_path: str) -> DictConfig:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
            
        Returns:
            Configuration as OmegaConf DictConfig
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        return OmegaConf.create(config_dict)
    
    @staticmethod
    def save_config(config: DictConfig, save_path: str) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration to save
            save_path: Path to save configuration
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_path.suffix.lower() in ['.yaml', '.yml']:
            OmegaConf.save(config, save_path)
        elif save_path.suffix.lower() == '.json':
            with open(save_path, 'w') as f:
                json.dump(OmegaConf.to_container(config), f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {save_path.suffix}")
    
    @staticmethod
    def merge_configs(*configs: DictConfig) -> DictConfig:
        """Merge multiple configurations.
        
        Args:
            configs: Configurations to merge
            
        Returns:
            Merged configuration
        """
        if not configs:
            return OmegaConf.create({})
        
        merged = configs[0]
        for config in configs[1:]:
            merged = OmegaConf.merge(merged, config)
        
        return merged
    
    @staticmethod
    def override_config(config: DictConfig, overrides: Dict[str, Any]) -> DictConfig:
        """Override configuration values.
        
        Args:
            config: Base configuration
            overrides: Values to override
            
        Returns:
            Updated configuration
        """
        override_conf = OmegaConf.create(overrides)
        return OmegaConf.merge(config, override_conf)
