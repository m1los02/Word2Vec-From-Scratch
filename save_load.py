import os
import numpy as np

from vocab import Vocabulary, Parameters


def save_model(path, model, vocab):

    save_dir = os.path.dirname(path)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(
        path,
        W_center=model.params.W_center,
        W_context=model.params.W_context,
        vocab=np.array(vocab.tokens)
    )
    print(f"Model saved to {path}")


def load_model(path):
    data = np.load(path, allow_pickle=True)
    vocab = Vocabulary(tokens=list(data["vocab"]))
    params = Parameters(data["W_center"], data["W_context"])
    print(f"Loaded model from {path}")
    return vocab, params
