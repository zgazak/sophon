.PHONY: build clean distclean publish publish-test lint format test

build: clean
	uv run python -m build

clean:
	rm -rf dist/ build/ *.egg-info sophon/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.py[cod]' -delete
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage .coverage.*

distclean: clean
	rm -rf .venv uv.lock

publish: build
	uv run twine upload dist/*

publish-test: build
	uv run twine upload --repository testpypi dist/*

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	uv run pytest
