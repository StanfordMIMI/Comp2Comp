autoformat:
	set -e
	isort .
	black --config pyproject.toml .
	flake8

lint:
	set -e
	isort -c .
	black --check --config pyproject.toml .
	flake8

build-docs:
	set -e
	mkdir -p docs/source/_static
	rm -rf docs/build
	rm -rf docs/source/generated
	cd docs && make html

all: autoformat build-docs