"""Full Defense path: L2 gradient clipping + Gaussian noise injection
(differential-privacy-style sanitization) for flagged updates."""
from typing import Dict, Tuple

import numpy as np

import config


class FullDefense:
    def __init__(self, clip_threshold: float = config.CLIP_THRESHOLD, noise_scale: float = config.NOISE_SCALE):
        self.clip_threshold = clip_threshold
        self.noise_scale = noise_scale

    def apply(self, update: np.ndarray) -> Tuple[np.ndarray, Dict]:
        original_norm = float(np.linalg.norm(update))
        clipped = original_norm > self.clip_threshold

        if clipped:
            sanitized = update * (self.clip_threshold / original_norm)
        else:
            sanitized = update.copy()

        clipped_norm = float(np.linalg.norm(sanitized))

        noise = np.random.normal(0, self.noise_scale, size=sanitized.shape)
        sanitized = sanitized + noise

        meta = {
            "clipped": clipped,
            "original_norm": original_norm,
            "clipped_norm": clipped_norm,
            "noise_added": float(np.linalg.norm(noise)),
        }
        return sanitized, meta
