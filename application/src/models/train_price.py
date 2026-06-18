"""Backward-compatible entry point for demand model training."""

from .train_model import main, train

__all__ = ["main", "train"]


if __name__ == "__main__":
    main()
