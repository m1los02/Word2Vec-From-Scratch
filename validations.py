import urllib.request
from pathlib import Path
import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt


def download_analogies(data_dir: Path = Path("data")) -> Path:
    """
    Downloads the Google Analogy Test Set (questions-words.txt) if not present
    
    The dataset contains ~19,000 analogy questions across categories like 
    capital-world, currency, city-in-state, and family relations
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "questions-words.txt"
    if path.exists():
        return path
    url = "https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt"
    print(f"Downloading analogy dataset from {url}...")
    urllib.request.urlretrieve(url, path)
    return path


def load_analogies(path: Path) -> List[Tuple[str, Tuple[str, str, str, str]]]:
    """
    Parses the analogy file into a list of category-labeled word tuples
    Args:
        path: Path to the questions-words.txt file
    Returns:
        A list of tuples: (category_name, (word_a, word_b, word_c, word_d))
    """
    items: List[Tuple[str, Tuple[str, str, str, str]]] = []
    current_cat = "overall"
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(":"):
            current_cat = line[1:].strip()
            continue
        parts = line.split()
        if len(parts) == 4:
            items.append((current_cat, tuple(map(str.lower, parts))))
    print(f"Loaded {len(items):,} analogy questions across categories.")
    return items


def analogy_accuracy(
    vocab_to_idx: Dict[str, int],
    embeddings: np.ndarray,
    questions: List[Tuple[str, Tuple[str, str, str, str]]],
    topn: int = 1,
    verbose: bool = False,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Evaluates embedding quality by solving 'A is to B as C is to ?'
    Logic: find vector nearest to (Vb - Va + Vc) excluding Va, Vb, and Vc

    Args:
        vocab_to_idx: Mapping of strings to matrix row indices
        embeddings: The trained word vector matrix (usually the 'center' weights)
        questions: The output from load_analogies
        topn: Rank threshold for a 'correct' hit (default is top-1)

    Returns:
        acc: Dictionary of accuracy scores per category
        counts: Dictionary containing 'evaluated' and 'skipped' totals
    """
    # normalize embeddings for cosine sim (dot product)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    emb_norm = embeddings / norms
    
    correct_per_cat: Dict[str, int] = {}
    total_per_cat: Dict[str, int] = {}
    skipped = 0

    for idx, (cat, (a, b, c, d)) in enumerate(questions):
        if not all(w in vocab_to_idx for w in (a, b, c, d)):
            skipped += 1
            continue
            
        ia, ib, ic, id_ = [vocab_to_idx[w] for w in (a, b, c, d)]
        va, vb, vc = emb_norm[ia], emb_norm[ib], emb_norm[ic]
        
        # calc target vector
        query = vb - va + vc
        query_norm = query / (np.linalg.norm(query) + 1e-9)
        
        # similarities scores
        scores = emb_norm @ query_norm
        
        # mask input words so we don't predict them
        scores[[ia, ib, ic]] = -np.inf
        
        top_indices = np.argsort(scores)[-topn:] if topn > 1 else np.array([np.argmax(scores)])
        
        total_per_cat[cat] = total_per_cat.get(cat, 0) + 1
        if id_ in top_indices:
            correct_per_cat[cat] = correct_per_cat.get(cat, 0) + 1

        if verbose and idx % 1000 == 0:
            print(f"Processed {idx}/{len(questions)} analogies...")

    # aggregate
    all_total = sum(total_per_cat.values())
    all_correct = sum(correct_per_cat.values())
    
    acc = {"overall": all_correct / all_total if all_total else 0.0}
    for cat, tot in total_per_cat.items():
        acc[cat] = correct_per_cat.get(cat, 0) / tot
        
    if verbose:
        print(f"Analogies evaluated: {sum(total_per_cat.values())}, skipped: {skipped}")

    return acc, {"evaluated": all_total, "skipped": skipped}


def plot_analogy_accuracy(acc: Dict[str, float], save_dir: Optional[str] = None):
    """Bar plot of per-category analogy accuracy"""
    cats = list(acc.keys())
    scores = [acc[c] * 100 for c in cats]
    plt.figure(figsize=(12, 5))
    plt.bar(cats, scores)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Accuracy (%)")
    plt.title("Word2Vec Analogy Accuracy per Category")
    plt.grid(axis='y')
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "Anology_acc.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
