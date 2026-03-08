# Word2Vec-From-Scratch

Pure-NumPy implementation of the Word2Vec Skip-Gram model with Negative Sampling.

This project was built from the ground up to handle massive text corpora efficiently without relying on heavy deep learning frameworks like PyTorch or TensorFlow. It includes custom data streaming, O(1) negative sampling, and dynamic learning rate schedules, validated against the standard Google Analogy Test Set.

## Key Features:
- O(1) Negative Sampling
- Memory-Efficient Data Streaming - uses Python generators to yield sliding-window context pairs on the fly
- Step-Based LR Scheduling - implements both linear decay and Cosine Annealing, updated per batch rather than per epoch.

## Running code

Dependecies:
``` bash
pip install numpy matplotlib
```

Run the main script via the command line. You can configure the dimensions, window size, negative samples, and learning rate schedule
``` bash
python main.py \
    --data_path "text8" \
    --max_tokens 18000000 \
    --vocab_size 300000 \
    --dim 300 \
    --window 5 \
    --epochs 5 \
    --lr 0.02 \
    --schedule linear \
    --neg_k 5 \
    --eval_analogies \
    --plot \
    --save_path "models/skipgram.npz"
```

## Experiments & Findings
During development, I ran several experiments on the text8 dataset to measure how learning rate schedules and negative sampling counts impact both accuracy (measured via the Google Analogy Test Set) and computational efficiency.

### Experiment 1:
Setup: lr = 0.01 (Fixed), K = 10 negative samples.

| Epochs | Analogy Accuracy |
|-------|------------------|
| 2     | 11.47%           |
| 5     | 11.97%           |
| 10    | 10.81%           |
| 15    | 8.75%            |

Observation: With a fixed learning rate, the model peaks around epoch 5 and then actively degrades. Because the model takes aggressively large steps late in training, it endlessly overshoots the optimal local minima, causing the embedding quality to collapse.

### Experiment 2
Setup: lr = 0.01 (Linear Decay Schedule), Epochs = 10.

| Negative Samples (K) | Analogy Accuracy | Avg Time per Epoch |
|----------------------|------------------|--------------------|
| 5                    | 15.62%           | ~1470s             |
| 10                   | 15.65%           | ~2470s             |

Observation: 1. Introducing a linear step-based decay schedule completely fixed the degradation seen in Experiment 1, pushing accuracy to ~15.6%.
2. Comparing K=5 against K=10 revealed massive diminishing returns. Doubling the number of negative samples increased the training time per epoch by nearly 70% (an extra 1,000 seconds), but yielded a negligible 0.03% improvement in analogy accuracy. Moving forward, K=5 is the optimal default for computational efficiency.

## Repository Structure
* `main.py` - CLI entry point and high-level training orchestration.
* `model.py` - The SkipGramNeg model architecture and forward/backward passes.
* `sampler.py` - Unigram table construction and O(1) sampling logic.
* `train.py` - Batch iteration, exact pair counting, and SGD loop.
* `utils.py` - Math utilities, sigmoid clipping, and dynamic LR schedulers.
* `read_data.py` - Text8 loading, tokenization, and subsampling.
* `save_load.py` - Utilities for saving/loading `.npz` checkpoints.
* `validations.py` - Google Analogy evaluation and plotting.
