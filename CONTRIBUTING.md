# Contributing to RiskSense-VLA

Thank you for your interest in contributing to RiskSense-VLA. This document outlines development setup, code style, testing, and the pull request process.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/RiskSense-VLA.git
   cd RiskSense-VLA
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

3. **Install the package in editable mode with dev and open_vocab extras**
   ```bash
   pip install -e ".[dev,open_vocab]"
   ```

## Code Style

- **Formatter/Linter**: Use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.
- **Line length**: 100 characters maximum.
- **Target Python**: 3.11 or newer.
- Run Ruff before committing:
  ```bash
  ruff check .
  ruff format .
  ```

## Testing

Tests are run with [pytest](https://pytest.org/). The project uses `src` layout with `tests/` at the repo root.

- **Run all tests**
  ```bash
  pytest
  ```

- **Run unit tests only**
  ```bash
  pytest tests/unit/
  ```

- **Run stress tests**
  ```bash
  pytest tests/stress/
  ```

- **Run smoke tests** (if available)
  ```bash
  pytest tests/ -k smoke
  ```

- **Run with coverage**
  ```bash
  pytest --cov=risksense_vla --cov-report=term-missing
  ```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Make your changes following the code style guidelines.
3. Add or update tests as needed.
4. Run the full test suite and ensure all tests pass.
5. Run `ruff check .` and `ruff format .` to ensure code quality.
6. Submit a pull request with a clear description of the changes.
7. Address any review feedback from maintainers.

## Logging Convention

Use the standard `logging` module for all output instead of `print`:

```python
import logging

_LOG = logging.getLogger(__name__)

_LOG.info("Processing %d items", count)
_LOG.warning("Unexpected value: %s", value)
_LOG.error("Failed to load: %s", path)
```

## Type Hints

Type hints are required for all public functions:

```python
def process_frame(frame: np.ndarray, threshold: float = 0.5) -> list[dict]:
    """Process a single frame and return detections."""
    ...
```

Use `from __future__ import annotations` at the top of files to enable postponed evaluation of annotations (e.g., for forward references).
