"""Model backends: capture + sparse decode behind a common interface."""

from .base import Capture, ModelBackend, ModelDims

__all__ = ["Capture", "ModelBackend", "ModelDims"]
