from . import library  # noqa: F401  (registers transitions)
from .base import REGISTRY, Ctx, Transition, Window

__all__ = ["REGISTRY", "Ctx", "Transition", "Window", "get"]


def get(name: str) -> Transition:
    if name not in REGISTRY:
        raise KeyError(f"unknown transition '{name}'. Available: {', '.join(REGISTRY)}")
    return REGISTRY[name]()
