from typing import List, Tuple, Iterator
import numpy as np

from model import SkipGramNeg
from vocab import Vocabulary
from utils import get_lr


def generate_pairs(tokens: List[str], vocab: Vocabulary, window: int) -> Iterator[Tuple[int, int]]:
    """Yields (center, context) word pairs using a sliding window"""
    lookup = vocab.token_to_idx
    
    for i, tok in enumerate(tokens):
        if tok not in lookup:
            continue
        center_idx = lookup[tok]
        
        start = max(0, i - window)
        end = min(len(tokens), i + window + 1)
        
        for j in range(start, end):
            if i == j:
                continue
            ctx_tok = tokens[j]
            if ctx_tok not in lookup:
                continue
            yield (center_idx, lookup[ctx_tok])


def batch_iterator(pairs: Iterator[Tuple[int, int]], batch_size: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Groups pairs into mini-batches"""
    centers = np.empty(batch_size, dtype=np.int32)
    contexts = np.empty(batch_size, dtype=np.int32)
    
    idx = 0
    for center_idx, context_idx in pairs:
        centers[idx] = center_idx
        contexts[idx] = context_idx
        idx += 1

        if idx == batch_size:
            yield centers, contexts
            centers = np.empty(batch_size, dtype=np.int32)
            contexts = np.empty(batch_size, dtype=np.int32)
            idx = 0

    if idx > 0:
        yield centers[:idx], contexts[:idx]


def train_epoch(
    model: SkipGramNeg, 
    pairs: Iterator[Tuple[int, int]], 
    initial_lr: float, 
    batch_size: int, 
    log_every: int,
    current_global_step: int,
    total_global_steps: int,
    schedule: str
) -> Tuple[float, List[float], int]:
    """Executes one epoch of mini-batch SGD."""
    total_loss = 0.0
    step_losses: List[float] = []
    total_steps = 0

    for centers, contexts in batch_iterator(pairs, batch_size):
        lr = get_lr(initial_lr, current_global_step, total_global_steps, schedule)

        loss, cache = model.forward(centers, contexts)
        grads = model.backward(cache)

        # param update
        np.add.at(model.params.W_center, grads["center_indices"], -lr * grads["grad_center"])
        np.add.at(model.params.W_context, grads["context_indices"], -lr * grads["grad_context_pos"])
        
        flat_neg_idx = grads["neg_indices"].reshape(-1)
        flat_grad_neg = grads["grad_context_neg"].reshape(-1, grads["grad_context_neg"].shape[-1])
        np.add.at(model.params.W_context, flat_neg_idx, -lr * flat_grad_neg)

        total_loss += loss
        total_steps += 1
        current_global_step += 1

        if log_every and total_steps % log_every == 0:
            avg_loss = total_loss / total_steps
            step_losses.append(avg_loss)
            print(f"Batch {total_steps} | LR: {lr:.5f} | Avg Loss: {avg_loss:.4f} | Current Loss: {loss:.4f}")

    return total_loss / max(1, total_steps), step_losses, current_global_step
