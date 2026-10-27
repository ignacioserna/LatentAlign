"""Spherical linear interpolation along a latent direction (Sec. 4.2, Supp. B).

``slerp(e, v, t)`` moves the unit embedding ``e`` towards the group direction
``v`` (t > 0, attribute addition) or away from it (t < 0, attribute removal).
"""
from __future__ import annotations

import numpy as np


def slerp(u: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
    dot = float(np.clip(u @ v, -1.0, 1.0))
    theta = np.arccos(dot)
    if theta < 1e-6:
        w = (1 - t) * u + t * v
        return w / np.linalg.norm(w)
    return (np.sin((1 - t) * theta) * u + np.sin(t * theta) * v) / np.sin(theta)


def slerp_torch(u, v, t):
    """Torch version used with InfiniteYou (works on CUDA tensors)."""
    import torch

    dot = torch.clamp(u @ v, -1.0, 1.0)
    theta = torch.acos(dot)
    s = (torch.sin((1 - t) * theta) * u + torch.sin(t * theta) * v) / torch.sin(theta)
    lerp = (1 - t) * u + t * v
    lerp = lerp / lerp.norm(dim=-1, keepdim=True)
    return torch.where(theta < 1e-6, lerp, s)
