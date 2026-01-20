"""Minimal PyTorch smoke test for the active virtual environment."""

import torch


def main() -> None:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Simple tensor op to verify core functionality
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    c = a + b

    print("a:", a)
    print("b:", b)
    print("a + b:", c)


if __name__ == "__main__":
    main()
