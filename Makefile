SHELL = /bin/bash

.PHONY: help
help:
	@echo "Commands:"
	@echo "venv    : creates the virtual environment in .venv."
	@echo "install : install dependencies into virtual environment."
	@echo "compile : update the environment requirements after changes to dependencies in pyproject.toml."
	@echo "update  : pip install new requriements into the virtual environment."
	@echo "test    : run pytests."

# create a virtual environment
.PHONY: venv
venv:
	python3 -m venv .venv
	source .venv/bin/activate && \
	python3 -m pip install pip==23.1.2 setuptools==68.0.0 wheel==0.40.0 && \
	pip install pip-tools==6.14.0

# ==============================================================================
# install requirements
# ==============================================================================

req-in := pyproject.toml    # input file for compilation of requirements
req-out := requirements.txt # output file of requirements compilation


# environment for production
.PHONY: install
install: venv
	source .venv/bin/activate && \
	pip-sync $(req-in) && \
	pip install -e . && \
	pre-commit install

# ==============================================================================
# compile requirements
# ==============================================================================

.PHONY: compile
compile:
	source .venv/bin/activate && \
	pip-compile $(req-in) -o $(req-out) --resolver=backtracking

# ==============================================================================
# update requirements and virtual env
# ==============================================================================

.PHONY: update
update:
	source .venv/bin/activate && \
	pip-sync $(req-out) && \
	pip install -e .

# ==============================================================================
# run tests
# ==============================================================================

.PHONY: test
test:
	source .venv/bin/activate && \
	pytest -vx .
