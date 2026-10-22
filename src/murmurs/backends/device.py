"""Choosing the torch device for model backends from ``compute.device``."""

from __future__ import annotations

from murmurs.config import Device
from murmurs.errors import BackendSetupError


def resolve_device(setting: Device) -> tuple[str, bool]:
    """The torch device for ``setting``, and whether ``auto`` fell back to the CPU.

    ``auto`` picks CUDA when available, else the CPU. ``cuda`` with no CUDA device
    raises ``BackendSetupError``. Imports ``torch`` only when called.
    """
    import torch

    if setting is Device.CPU:
        return "cpu", False
    if torch.cuda.is_available():
        return "cuda", False
    if setting is Device.CUDA:
        raise BackendSetupError("compute.device is 'cuda' but no CUDA device is available")
    return "cpu", True
