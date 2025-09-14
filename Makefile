SHELL = /bin/bash

.PHONY: help
help:
	@echo "Commands:"
	@echo "install-dev-env  : install dependencies into virtual environment for development."
	@echo "update-dev-env   : pip install new dev requriements into the environment."
	@echo "install-docs-env : install dependencies into virtual environment for docs+development."
	@echo "test             : run pytests."
	@echo "coverage         : compute test coverage"

# ==============================================================================
# dev
# ==============================================================================

.PHONY: install-dev-env
install-dev-env:
	uv sync --all-extras --group dev --group tests && \
	uv run pre-commit install

.PHONY: update-dev-env
update-dev-env:
	uv sync --all-extras --group dev --group tests


# ==============================================================================
# ci tests
# ==============================================================================

.PHONY: install-ci-env
install-ci-env:
	uv sync --all-extras --group tests

# ==============================================================================
# test
# ==============================================================================

.PHONY: test
test:
	uv run pytest tests

# ==============================================================================
# coverage
# ==============================================================================

.PHONY: coverage
coverage:
	uv run pytest --cov=src --cov-report html  tests
