import numpy as np
from typing import Optional, Tuple


class NegativeSampler:
    """Samples negative word indices in constant time using a pre-computed, smoothed unigram lookup table"""
    def __init__(self, freqs: np.ndarray, power: float = 0.75, table_size: int = 10_000_000, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()
        self.table_size = table_size
        self.unigram_table = self._build_table(freqs, power)

    def _build_table(self, freqs: np.ndarray, power: float) -> np.ndarray:
        """Builds the unigram distribution and resulting lookup table"""
        adjusted = freqs ** power
        dist = adjusted / adjusted.sum()
        
        counts = np.round(dist * self.table_size).astype(int)
        table = np.repeat(np.arange(len(dist)), counts)
        self.rng.shuffle(table)
        return table

    def sample(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Samples negative indices in constant time"""
        random_indices = self.rng.integers(0, len(self.unigram_table), size=shape)
        return self.unigram_table[random_indices]
        