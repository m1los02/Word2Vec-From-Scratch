import numpy as np
import matplotlib.pyplot as plt
from typing import List
import os

from vocab import Vocabulary


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Calculates sigmoid with clipping to prevent overflow"""
    z = np.clip(z, -15, 15)
    return 1.0 / (1.0 + np.exp(-z))


def get_lr(
    initial_lr: float, 
    current_step: int, 
    total_steps: int, 
    schedule: str = "linear", 
    min_lr_fraction: float = 0.001
) -> float:
    """Calculates the learning rate for the current step"""
    progress = min(1.0, current_step / max(1, total_steps))
    min_lr = initial_lr * min_lr_fraction

    if schedule == "cosine":
        return min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + np.cos(np.pi * progress))
    elif schedule == "linear":
        return max(min_lr, initial_lr * (1.0 - progress))
    else:
        return initial_lr


def get_exact_total_pairs(tokens: List[str], vocab: Vocabulary, window: int) -> int:
    """Calculates the exact number of valid (center, context) pairs using"""
    lookup = vocab.token_to_idx
    valid_mask = np.array([tok in lookup for tok in tokens], dtype=np.int32)
    
    kernel = np.ones(window * 2 + 1, dtype=np.int32)
    kernel[window] = 0 
    
    context_counts = np.convolve(valid_mask, kernel, mode='same')
    exact_pairs = np.sum(valid_mask * context_counts)
    
    return int(exact_pairs)


def plot_training_results(
    step_losses: List[float], 
    epoch_losses: List[float], 
    dim: int, 
    lr: float, 
    vocab_len: int,
    save_dir: str
) -> None:
    """Generates and saves loss curves to the specified directory"""
    plt.style.use('ggplot') 
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].plot(step_losses, color='skyblue', alpha=0.4, label='Raw Step Loss')
    if len(step_losses) > 10:
        smooth_loss = np.convolve(step_losses, np.ones(10)/10, mode='valid')
        axes[0].plot(smooth_loss, color='navy', linewidth=1.5, label='Moving Avg (10)')
    
    axes[0].set_title('Training Progress (Steps)', fontweight='bold')
    axes[0].set_xlabel('Log Steps')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(range(1, len(epoch_losses) + 1), epoch_losses, color='crimson', marker='o', linewidth=2)
    axes[1].set_title('Training Convergence (Epochs)', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Average Loss')
    axes[1].set_xticks(range(1, len(epoch_losses) + 1))
    
    plt.suptitle(f'Skip-Gram: dim={dim}, lr={lr}, vocab={vocab_len}', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs(save_dir, exist_ok=True)
    filename = f"loss.png"
    full_path = os.path.join(save_dir, filename)
    
    plt.savefig(full_path, dpi=300)
    print(f"Plots saved to {full_path}")
    plt.show()
