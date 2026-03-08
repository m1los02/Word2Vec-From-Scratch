import numpy as np
from typing import Tuple, Dict, Any

from utils import sigmoid
from vocab import Parameters
from sampler import NegativeSampler


def init_params(vocab_size: int, dim: int, rng: np.random.Generator) -> Parameters:
    """Initializes Word2Vec matrices using uniform distribution"""
    bound = 1.0 / np.sqrt(dim)
    W_center = rng.uniform(-bound, bound, size=(vocab_size, dim))
    W_context = rng.uniform(-bound, bound, size=(vocab_size, dim))
    return Parameters(W_center=W_center, W_context=W_context)


class SkipGramNeg:
    """Skip-Gram model with Negative Sampling (batch version)"""
    
    def __init__(self, params: Parameters, K: int, sampler: NegativeSampler) -> None:
        self.params = params
        self.K = K
        self.sampler = sampler

    def forward(self, center_indices: np.ndarray, context_indices: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Executes the batch forward pass, returning loss and cache"""
        B = center_indices.shape[0]

        v_c = self.params.W_center[center_indices]
        pos_vectors = self.params.W_context[context_indices]
        
        # sample negatives
        neg_indices = self.sampler.sample(shape=(B, self.K))
        neg_vectors = self.params.W_context[neg_indices]

        # scores
        pos_scores = np.sum(pos_vectors * v_c, axis=1)
        pos_sig = sigmoid(pos_scores)
        
        #neg_scores = np.sum(neg_vectors * v_c[:, None, :], axis=2)
        neg_scores = np.einsum('bd,bkd->bk', v_c, neg_vectors)
        neg_sig = sigmoid(-neg_scores)

        # loss
        pos_loss = -np.log(pos_sig + 1e-12)
        neg_loss = -np.sum(np.log(neg_sig + 1e-12), axis=1)
        loss = np.mean(pos_loss + neg_loss)

        # cacheing
        cache = {
            "center_indices": center_indices,
            "context_indices": context_indices,
            "v_c": v_c,
            "pos_vectors": pos_vectors,
            "neg_vectors": neg_vectors,
            "pos_sig": pos_sig,
            "neg_scores": neg_scores,
            "neg_indices": neg_indices,
        }
        return loss, cache

    def backward(self, cache: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Computes gradients based on the forward cache"""
        neg_sig_pos = sigmoid(cache["neg_scores"])

        grad_center = (cache["pos_sig"] - 1.0)[:, None] * cache["pos_vectors"]
        #grad_center += np.sum(neg_sig_pos[:, :, None] * neg_vectors, axis=1)
        grad_center += np.einsum('bk,bkd->bd', neg_sig_pos, cache["neg_vectors"])

        grad_context_pos = (cache["pos_sig"] - 1.0)[:, None] * cache["v_c"]
        grad_context_neg = neg_sig_pos[:, :, None] * cache["v_c"][:, None, :]

        return {
            "grad_center": grad_center,
            "grad_context_pos": grad_context_pos,
            "grad_context_neg": grad_context_neg,
            "center_indices": cache["center_indices"],
            "context_indices": cache["context_indices"],
            "neg_indices": cache["neg_indices"],
        }
