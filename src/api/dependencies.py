"""
FastAPI dependencies for dependency injection
"""

from functools import lru_cache
from typing import Dict, Any
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from src.core.protocols import RoutingStrategy, LLMProvider
from src.core.router_strategy import (
    HybridRoutingStrategy,
    SemanticRoutingStrategy,
    KeywordRoutingStrategy
)
from src.core.llm_client import OllamaClient
from src.forensics.logger import DataLogger


class AppState:
    """Global application state container"""
    def __init__(self):
        self.config: DictConfig = None
        self.router: RoutingStrategy = None
        self.llm_client: LLMProvider = None
        self.logger: DataLogger = None
        self.experts_config: Dict[str, Dict[str, Any]] = {}


# Singleton app state
app_state = AppState()


def load_config() -> DictConfig:
    """
    Load Hydra configuration from conf/ directory.
    This should be called once at application startup.
    """
    # Clear any existing Hydra instance
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # Initialize Hydra with the config path
    with initialize(version_base=None, config_path="../../conf"):
        cfg = compose(config_name="config")

    return cfg


def initialize_router(config: DictConfig) -> RoutingStrategy:
    """
    Initialize the routing strategy based on configuration.

    Args:
        config: Hydra configuration object

    Returns:
        Initialized routing strategy instance
    """
    router_name = config.router.name

    if router_name == "hybrid":
        threshold = config.router.parameters.semantic_threshold
        return HybridRoutingStrategy(semantic_threshold=threshold)
    elif router_name == "semantic":
        model = config.router.parameters.semantic_model
        threshold = config.router.parameters.semantic_threshold
        return SemanticRoutingStrategy(model_name=model, threshold=threshold)
    elif router_name == "keyword":
        # Load patterns from config if available
        patterns = None
        if hasattr(config.router, 'parameters') and hasattr(config.router.parameters, 'patterns'):
            patterns = OmegaConf.to_container(config.router.parameters.patterns, resolve=True)
        return KeywordRoutingStrategy(patterns=patterns)
    else:
        raise ValueError(f"Unknown router type: {router_name}")


def initialize_llm_client(config: DictConfig) -> LLMProvider:
    """
    Initialize the LLM client (Ollama).

    Args:
        config: Hydra configuration object

    Returns:
        Initialized LLM provider instance
    """
    # For now, we only support Ollama
    # In the future, this could be extended to support other providers
    return OllamaClient(base_url="http://localhost:11434", timeout=30)


def initialize_logger(config: DictConfig) -> DataLogger:
    """
    Initialize the data logger for telemetry.

    Args:
        config: Hydra configuration object

    Returns:
        Initialized data logger instance
    """
    log_path = config.forensics.log_path
    return DataLogger(log_file=log_path)


def load_experts_config(config: DictConfig) -> Dict[str, Dict[str, Any]]:
    """
    Load expert configurations from Hydra config.

    Args:
        config: Hydra configuration object

    Returns:
        Dictionary mapping expert IDs to their configurations
    """
    experts = {}

    # Load professor config
    if hasattr(config, 'professor'):
        experts['professor'] = OmegaConf.to_container(config.professor, resolve=True)

    # Load zoomer config
    if hasattr(config, 'zoomer'):
        experts['zoomer'] = OmegaConf.to_container(config.zoomer, resolve=True)

    return experts


def initialize_app_state():
    """
    Initialize all application dependencies.
    This should be called once at application startup.
    """
    # Load configuration
    config = load_config()
    app_state.config = config

    # Initialize components
    app_state.router = initialize_router(config)
    app_state.llm_client = initialize_llm_client(config)
    app_state.logger = initialize_logger(config)
    app_state.experts_config = load_experts_config(config)


# FastAPI dependency functions
def get_router() -> RoutingStrategy:
    """FastAPI dependency to get the router instance"""
    if app_state.router is None:
        raise RuntimeError("Application not initialized. Call initialize_app_state() first.")
    return app_state.router


def get_llm_client() -> LLMProvider:
    """FastAPI dependency to get the LLM client instance"""
    if app_state.llm_client is None:
        raise RuntimeError("Application not initialized. Call initialize_app_state() first.")
    return app_state.llm_client


def get_logger() -> DataLogger:
    """FastAPI dependency to get the logger instance"""
    if app_state.logger is None:
        raise RuntimeError("Application not initialized. Call initialize_app_state() first.")
    return app_state.logger


def get_config() -> DictConfig:
    """FastAPI dependency to get the config instance"""
    if app_state.config is None:
        raise RuntimeError("Application not initialized. Call initialize_app_state() first.")
    return app_state.config


def get_experts_config() -> Dict[str, Dict[str, Any]]:
    """FastAPI dependency to get the experts configuration"""
    if not app_state.experts_config:
        raise RuntimeError("Application not initialized. Call initialize_app_state() first.")
    return app_state.experts_config
