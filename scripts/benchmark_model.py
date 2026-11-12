
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import EmbeddingNet


def benchmark(batch_size, iterations=3):
    model = EmbeddingNet(embedding_dim=128)
    model.train()

    x = torch.randn(batch_size, 3, 32, 32)
    target = torch.randn(batch_size, 128)

    # Warm up the model before measuring.
    model.zero_grad(set_to_none=True)
    loss = (model(x) - target).square().mean()
    loss.backward()

    times = []

    for _ in range(iterations):
        model.zero_grad(set_to_none=True)

        start = time.perf_counter()

        embeddings = model(x)
        loss = (embeddings - target).square().mean()
        loss.backward()

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return sum(times) / len(times)


def main():
    print("PyTorch:", torch.__version__)
    print("CPU threads:", torch.get_num_threads())

    print("\nBatch size | Seconds per step")
    print("-----------------------------")

    for batch_size in [16, 32, 64]:
        elapsed = benchmark(batch_size)

        print(f"{batch_size:10d} | {elapsed:.3f}")


if __name__ == "__main__":
    main()
