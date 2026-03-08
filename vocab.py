import numpy as np
from typing import Sequence, Mapping
from dataclasses import dataclass, field


@dataclass
class Vocabulary:
    """
    Stores a mapping between unique tokens and their integer indices

    Attributes:
        tokens: An ordered sequence of unique strings in the vocabulary
        token_to_idx: A mapping from each token string to its unique integer index
    """
    tokens: Sequence[str]
    token_to_idx: Mapping[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    def idx(self, token: str) -> int:
        """Retrieves the integer index for a given token"""
        return self.token_to_idx[token]
    

@dataclass
class Parameters:
    """
    Container for Word2Vec weight matrices
    Attributes:
        W_center: The input embedding matrix of shape (vocab_size, embedding_dim)
        W_context: The output embedding matrix of shape (vocab_size, embedding_dim)
    """
    W_center: np.ndarray
    W_context: np.ndarray
