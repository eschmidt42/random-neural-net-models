SHELL = /bin/bash

.PHONY: help
help:
	@echo "Commands:"
	@echo "venv            : creates the virtual environment in .venv."
	@echo "install         : install dependencies into virtual environment."
	@echo "install-gh-test : install dependencies required in .github/workflows/test.yml."
	@echo "compile         : update the environment requirements after changes to dependencies in requirements/*.in files."
	@echo "compile-dev     : update the dev environment requirements after changes to requirements/dev.in."
	@echo "compile-upgrade : upgrade the environment requirements."
	@echo "update          : pip install new requriements into the virtual environment."
	@echo "test            : run pytests."
	@echo "bump-patch      : bump the patch version."
	@echo "bump-minor      : bump the minor version."

# create a virtual environment
.PHONY: venv
venv:
	python3 -m venv .venv
	source .venv/bin/activate && \
	python3 -m pip install pip==23.3.2 setuptools==69.0.3 wheel==0.42.0 && \
	pip install pip-tools==7.3.0

# ==============================================================================
# compile requirements
# ==============================================================================

req-core-in := requirements/core.in   # input file for compilation of core requirements
req-core-out := requirements/core.txt # output file of core requirements compilation
req-test-in := requirements/test.in   # input file for compilation of test requirements
req-test-out := requirements/test.txt # output file of test requirements compilation
req-publish-in := requirements/publish.in   # input file for compilation of publish requirements
req-publish-out := requirements/publish.txt # output file of publish requirements compilation
req-dev-in := requirements/dev.in     # input file for compilation of dev requirements
req-dev-out := requirements/dev.txt   # output file of dev requirements compilation

.PHONY: compile
compile:
	source .venv/bin/activate && \
	pip-compile $(req-core-in) -o $(req-core-out) --resolver=backtracking && \
	pip-compile $(req-test-in) -o $(req-test-out) --resolver=backtracking && \
	pip-compile $(req-publish-in) -o $(req-publish-out) --resolver=backtracking && \
	pip-compile $(req-dev-in) -o $(req-dev-out) --resolver=backtracking

.PHONY: compile-dev
compile-dev:
	source .venv/bin/activate && \
	pip-compile $(req-dev-in) -o $(req-dev-out) --resolver=backtracking

.PHONY: compile-upgrade
compile-upgrade:
	source .venv/bin/activate && \
	pip-compile -U $(req-core-in) -o $(req-core-out) --resolver=backtracking && \
	pip-compile -U $(req-test-in) -o $(req-test-out) --resolver=backtracking && \
	pip-compile -U $(req-publish-in) -o $(req-publish-out) --resolver=backtracking && \
	pip-compile -U $(req-dev-in) -o $(req-dev-out) --resolver=backtracking

# ==============================================================================
# install requirements
# ==============================================================================

# default environment (for local development)
.PHONY: install
install: venv
	source .venv/bin/activate && \
	pip-sync $(req-core-out) $(req-test-out) $(req-publish-out) $(req-dev-out) && \
	pip install -e . && \
	pre-commit install

# github actions test environment
.PHONY: install-gh-test
install-gh-test:
	pip install -r $(req-core-out) -r $(req-test-out) && \
	pip install -e .

# github actions test environment
.PHONY: install-gh-publish
install-gh-publish:
	pip install -r $(req-core-out) -r $(req-publish-out)

# ==============================================================================
# update requirements and virtual env after changes to requirements/*.txt files
# ==============================================================================

.PHONY: update
update:
	source .venv/bin/activate && \
	pip-sync $(req-core-out) $(req-test-out) $(req-publish-out) $(req-dev-out) && \
	pip install -e .

# ==============================================================================
# run tests
# ==============================================================================

.PHONY: test
test:
	source .venv/bin/activate && \
	pytest -vx .

# ==============================================================================
# bump version
# ==============================================================================

.PHONY: bump-patch
bump-patch:
	source .venv/bin/activate && \
	bumpver update --patch

.PHONY: bump-minor
bump-minor:
	source .venv/bin/activate && \
	bumpver update --minor
