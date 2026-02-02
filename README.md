# Zero To Hero Exercises

This repo houses Python Jupyter notebooks for the exercises in Andrej Karpathy's [Neural Networks Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) youtube series.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Jupyter is included as a dev dependency.

1. **Install uv** (if you don't have it): `curl -LsSf https://astral.sh/uv/install.sh | sh`

2. **Install dependencies**: from the project root, run `uv sync` (this installs the project and its dev dependencies, including Jupyter).

3. **Run Jupyter Lab**: `uv run jupyter lab`

That's it — `uv run` uses the project's locked environment, so you don't need to activate a venv.
