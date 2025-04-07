# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test/Lint Commands
- Use .venv/activate to activate the virtualenv you need. It's .venv/bin/activate, NOT activate/bin/activate.
- Install for development: `uv pip install -e .`
- Install development dependencies: `uv pip install -r requirements/requirements-dev.txt`
- Run all tests: `uv run pytest`
- Run single test file: `uv run pytest tests/path/to/test_file.py`
- Run specific test: `uv run pytest tests/path/to/test_file.py::TestClass::test_method`
- Format code: `uv run black --line-length 100 --preview .`
- Sort imports: `uv run isort .`
- Lint code: `uv run flake8`
- Run all pre-commit hooks: `pre-commit run --all-files`
- Update dependencies: `./scripts/pip-compile.sh --upgrade`

## Code Style Guidelines
- Follow PEP 8 with 100 character line length
- Use Black for formatting with `--line-length 100 --preview` options
- Sort imports with isort (Black compatible profile)
- Support Python 3.9+ (3.9, 3.10, 3.11, 3.12)
- No type hints used in this project
- Flake8 ignores E203 and W503
- Follow existing patterns for error handling with meaningful error messages
- Maintain consistent naming conventions (snake_case for variables/functions, PascalCase for classes)
- Add docstrings to public functions and classes
