import argparse
import numpy as np
import math
import time

from read_data import Text8Pipeline
from model import init_params, SkipGramNeg
from sampler import NegativeSampler
from validations import download_analogies, load_analogies, analogy_accuracy, plot_analogy_accuracy
from utils import get_exact_total_pairs, plot_training_results
from train import generate_pairs, train_epoch
from save_load import save_model


def main():
    parser = argparse.ArgumentParser(description="Word2Vec Skip-Gram Training")
    parser.add_argument("--data_path", type=str, required=True, help="Path to text8 file")
    parser.add_argument("--max_tokens", type=int, default=20_000_000)
    parser.add_argument("--vocab_size", type=int, default=300_000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=200)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--schedule", type=str, default="linear")
    parser.add_argument("--neg_k", type=int, default=5, help="Negatives per pair")
    parser.add_argument("--eval_analogies", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--save_graph_path", type=str, default=None)
    args = parser.parse_args()

    # handle data
    pipeline = Text8Pipeline(args.data_path)
    raw_tokens = pipeline.load_tokens(max_tokens=args.max_tokens)
    vocab, freqs, raw_counts = pipeline.build_vocab(raw_tokens, max_vocab_size=args.vocab_size)
    tokens = pipeline.subsample_tokens(raw_tokens, raw_counts, threshold=1e-5)
    
    # initializations
    rng = np.random.default_rng(123)
    params = init_params(vocab_size=len(vocab), dim=args.dim, rng=rng)
    sampler = NegativeSampler(freqs=freqs, rng=rng)
    model = SkipGramNeg(params=params, K=args.neg_k, sampler=sampler)

    questions = load_analogies(download_analogies()) if args.eval_analogies else None

    # step calculation
    exact_total_pairs = get_exact_total_pairs(tokens, vocab, args.window)
    steps_per_epoch = math.ceil(exact_total_pairs / args.batch_size)
    total_global_steps = steps_per_epoch * args.epochs
    current_global_step = 0
    
    log_freq = max(1, steps_per_epoch // 5)

    print(f"\n{'='*60}\nSTARTING TRAINING: {exact_total_pairs:,} pairs per epoch")
    print(f"Config: dim={args.dim}, vocab={len(vocab)}, epochs={args.epochs}, lr={args.lr} ({args.schedule})\n{'='*60}\n")

    all_step_losses, epoch_losses = [], []
    start_time = time.time()

    # train loop
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        pairs_gen = generate_pairs(tokens, vocab, window=args.window)
        
        avg_loss, step_losses, current_global_step = train_epoch(
            model=model, 
            pairs=pairs_gen, 
            initial_lr=args.lr, 
            batch_size=args.batch_size, 
            log_every=log_freq,
            current_global_step=current_global_step,
            total_global_steps=total_global_steps,
            schedule=args.schedule
        )
        
        all_step_losses.extend(step_losses)
        epoch_losses.append(avg_loss)
        
        print(f"› Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.4f} | Time: {time.time() - epoch_start:.1f}s")

        # evaluate
        if questions:
            acc, counts = analogy_accuracy(vocab.token_to_idx, model.params.W_center, questions)
            total_acc = sum(acc.values()) / len(acc) if acc else 0
            print(f"  Analogy Score: {total_acc*100:.2f}% (Eval: {counts['evaluated']})")

    print(f"\nTraining Complete in {(time.time() - start_time) / 60:.2f} minutes.")

    if args.plot:
        plot_training_results(all_step_losses, epoch_losses, args.dim, args.lr, len(vocab), save_dir=args.save_graph_path)
        plot_analogy_accuracy(acc,  save_dir=args.save_graph_path)

    if args.save_path:
        save_model(args.save_path, model, vocab)

if __name__ == "__main__":
    main()
