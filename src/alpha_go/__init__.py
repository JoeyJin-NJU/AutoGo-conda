"""AlphaGo research codebase using GNU Go as the game engine."""
from alpha_go.engine import BLACK, EMPTY, WHITE, GTPEngine
from alpha_go.model import (
    MODEL_CONFIGS,
    MuPGoResNet,
    MuPModelConfig,
    create_mup_model,
    count_parameters,
    get_model_info,
)

__all__ = [
    "BLACK",
    "EMPTY",
    "WHITE",
    "GTPEngine",
    # muP model exports
    "MuPGoResNet",
    "MuPModelConfig",
    "MODEL_CONFIGS",
    "create_mup_model",
    "count_parameters",
    "get_model_info",
]
