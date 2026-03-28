.PHONY: build clean publish publish-test lint format test

build: clean
	uv run python -m build

clean:
	rm -rf dist/ build/ *.egg-info sophon/*.egg-info

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
