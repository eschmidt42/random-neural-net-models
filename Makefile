SHELL = /bin/bash

.PHONY: help
help:
	@echo "Commands:"
	@echo "venv            : creates the virtual environment in .venv."
	@echo "install         : install dependencies into virtual environment."
	@echo "install-gh-test : install dependencies required in .github/workflows/test.yml."
	@echo "compile         : update the environment requirements after changes to dependencies in requirements/*.in files."
	@echo "compile-upgrade : upgrade the environment requirements."
	@echo "update          : pip install new requriements into the virtual environment."
	@echo "test            : run pytests."

# create a virtual environment
.PHONY: venv
venv:
	python3 -m venv .venv
	source .venv/bin/activate && \
	python3 -m pip install pip==23.1.2 setuptools==68.0.0 wheel==0.40.0 && \
	pip install pip-tools==6.14.0

# ==============================================================================
# compile requirements
# ==============================================================================

req-core-in := requirements/core.in   # input file for compilation of core requirements
req-core-out := requirements/core.txt # output file of core requirements compilation
req-test-in := requirements/test.in   # input file for compilation of test requirements
req-test-out := requirements/test.txt # output file of test requirements compilation
req-dev-in := requirements/dev.in     # input file for compilation of dev requirements
req-dev-out := requirements/dev.txt   # output file of dev requirements compilation

.PHONY: compile
compile:
	source .venv/bin/activate && \
	pip-compile $(req-core-in) -o $(req-core-out) --resolver=backtracking && \
	pip-compile $(req-test-in) -o $(req-test-out) --resolver=backtracking && \
	pip-compile $(req-dev-in) -o $(req-dev-out) --resolver=backtracking

.PHONY: compile-upgrade
compile-upgrade:
	source .venv/bin/activate && \
	pip-compile -U $(req-core-in) -o $(req-core-out) --resolver=backtracking && \
	pip-compile -U $(req-test-in) -o $(req-test-out) --resolver=backtracking && \
	pip-compile -U $(req-dev-in) -o $(req-dev-out) --resolver=backtracking

# ==============================================================================
# install requirements
# ==============================================================================

req-in := pyproject.toml    # input file for compilation of requirements
req-out := requirements.txt # output file of requirements compilation


# default environment (for local development)
.PHONY: install
install: venv
	source .venv/bin/activate && \
	pip-sync $(req-core-out) $(req-test-out) $(req-dev-out) && \
	pip install -e . && \
	pre-commit install

#  github actions test environment
.PHONY: install-gh-test
install-gh-test:
	pip install -r $(req-core-out) -r $(req-test-out) && \
	pip install -e .

# ==============================================================================
# update requirements and virtual env after changes to requirements/*.txt files
# ==============================================================================

.PHONY: update
update:
	source .venv/bin/activate && \
	pip-sync $(req-core-out) $(req-test-out) $(req-dev-out) && \
	pip install -e .

# ==============================================================================
# run tests
# ==============================================================================

.PHONY: test
test:
	source .venv/bin/activate && \
	pytest -vx .
