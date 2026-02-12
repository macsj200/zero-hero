# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational exercises from Andrej Karpathy's "Neural Networks: Zero to Hero" series. The codebase is primarily Jupyter notebooks implementing neural network concepts from scratch using PyTorch.

## Setup & Commands

- **Install dependencies:** `uv sync`
- **Run Jupyter Lab:** `uv run jupyter lab`
- **Run main.py:** `uv run python main.py`

Uses [uv](https://docs.astral.sh/uv/) for dependency management. No test suite or linter is configured.

## Architecture

The project progresses through increasingly complex neural network implementations:

1. **micrograd_exercises.ipynb** — Custom autograd engine (`Value` class) with manual backpropagation, softmax, and negative log likelihood loss
2. **makemore-bigram.ipynb** — Bigram character-level language model (27x27 weight matrix) for name generation
3. **makemore_exercises.ipynb** — Trigram model with two-character context (54 input dims), train/dev/test splits, and hyperparameter tuning

All models use `data/names.txt` (~32K names) as training data. Character-level processing maps 26 letters + a start/end token to indices 0-26.

## Key Patterns

- Neural networks built from raw PyTorch tensors (no `nn.Module`), with manual gradient descent
- One-hot encoding for character inputs, cross-entropy loss, L2 regularization
- Python 3.12+, dependencies: torch, numpy, matplotlib
