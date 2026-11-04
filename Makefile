# Project variables
PACKAGE   := sophon
DIST_NAME := sophon-bench
VERSION   := $(shell sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)

.DEFAULT_GOAL := help

###################
# Development     #
###################

.PHONY: sync lint format test

sync: ## Sync dependencies into the virtual environment
	uv sync --all-extras

lint: ## Run linter (ruff)
	uv run ruff check .

format: ## Format code and fix lint issues (ruff)
	uv run ruff format .
	uv run ruff check --fix .

test: ## Run the test suite
	uv run pytest

###################
# Release         #
###################

.PHONY: version build check-clean-tree check-version-unpublished check-tag-free tag publish-test publish

version: ## Print the version from pyproject.toml
	@echo $(VERSION)

build: clean ## Build sdist + wheel into dist/
	uv build

check-clean-tree:
	@test -z "$$(git status --porcelain)" || { echo "ERROR: git tree is dirty; commit or stash first"; git status --short; exit 1; }

check-version-unpublished:
	@if curl -sf https://pypi.org/pypi/$(DIST_NAME)/$(VERSION)/json > /dev/null; then \
		echo "ERROR: $(DIST_NAME) $(VERSION) is already on PyPI; bump version in pyproject.toml first"; exit 1; \
	fi
	@echo "PyPI check OK: $(DIST_NAME) $(VERSION) not yet published"

check-tag-free:
	@if git rev-parse -q --verify "refs/tags/v$(VERSION)" > /dev/null; then \
		echo "ERROR: local tag v$(VERSION) already exists"; exit 1; \
	fi
	@if git ls-remote --exit-code --tags origin "v$(VERSION)" > /dev/null 2>&1; then \
		echo "ERROR: tag v$(VERSION) already exists on origin"; exit 1; \
	fi
	@echo "Tag check OK: v$(VERSION) is free"

tag: check-clean-tree check-tag-free check-version-unpublished ## git tag v<version> from pyproject.toml and push it
	git tag v$(VERSION)
	git push origin v$(VERSION)

publish-test: check-clean-tree test build ## Test + build + upload to TestPyPI (token from ~/.pypirc)
	uvx twine upload --repository testpypi dist/*

# Manual fallback. Prefer the GitHub Release workflow (docs/RELEASING.md);
# this path bypasses the reviewer gate. PyPI uploads are irreversible per
# version - bump pyproject.toml first.
publish: check-clean-tree check-version-unpublished test build ## Guarded upload to PyPI (token from ~/.pypirc)
	uvx twine upload dist/*

###################
# Cleanup         #
###################

.PHONY: clean distclean

clean: ## Remove build artifacts and caches
	rm -rf dist/ build/ *.egg-info sophon/*.egg-info
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	find . -type f -name '*.py[cod]' -not -path './.venv/*' -delete
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage .coverage.*

distclean: clean ## clean + remove the virtual environment and lock file
	rm -rf .venv uv.lock

###################
# Help            #
###################

.PHONY: help

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-28s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
