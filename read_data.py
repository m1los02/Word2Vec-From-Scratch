import os
import random
import numpy as np
from collections import Counter
from typing import List, Tuple, Optional

from vocab import Vocabulary

class Text8Pipeline:
    """Handles loading, tokenizing, vocabulary creation, and subsampling for the text8 dataset"""
    def __init__(self, file_path: str = "text8"):
        self.file_path = file_path

    def load_tokens(self, max_tokens: Optional[int] = None) -> List[str]:
        """Loads raw tokens from the text8 file"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Could not find '{self.file_path}'. "
            )
        
        print(f"Loading data from {self.file_path}...")
        with open(self.file_path, "r") as f:
            text = f.read()
            tokens = text.split()
        
        # keeps only number of max_tokens (from all the raw tokens) if specified
        if max_tokens:
            tokens = tokens[:max_tokens]
            
        print(f"Loaded {len(tokens):,} raw tokens.")
        return tokens

    def build_vocab(self, tokens: List[str], max_vocab_size: int) -> Tuple[Vocabulary, np.ndarray, Counter]:
        """
        Builds the Vocabulary object and frequency array
        Returns: (vocab_object, frequencies_array, raw_counts)
        """
        counts = Counter(tokens)
        
        # sort by frequency
        sorted_counts = counts.most_common(max_vocab_size)
        
        vocab_tokens = [token for token, _ in sorted_counts]
        freqs_list = [count for _, count in sorted_counts]
        
        # create Vocabulary
        vocab = Vocabulary(tokens=vocab_tokens)
        freqs = np.array(freqs_list, dtype=np.float64)
        
        return vocab, freqs, counts

    
    

    def subsample_tokens(self, tokens: List[str], counts: Counter, threshold: float = 1e-5) -> List[str]:
        """
        Subsampling from original paper - discards frequent words probabilistically
        Formula: P(keep) = (sqrt(freq / threshold) + 1) * (threshold / freq) (clip to 1.0 if bigger than that)
        
        This speeds up training and improves rare word vectors
        """
        total_tokens = len(tokens)
        keep_tokens = []
        
        # probabilities
        word_probs = {}
        for word, count in counts.items():
            freq = count / total_tokens
            p_keep = min(1.0, (np.sqrt(freq / threshold) + 1) * (threshold / freq))
            word_probs[word] = p_keep

        for word in tokens:
            if word in word_probs:
                if random.random() < word_probs[word]:
                    keep_tokens.append(word)
            else:
                keep_tokens.append(word)
                
        print(f"Subsampling complete. Reduced tokens from {len(tokens):,} to {len(keep_tokens):,}")
        return keep_tokens
    