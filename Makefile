.PHONY: lint format all


lint:
	poetry run flake8 src
	poetry run mypy --no-strict-optional --warn-unreachable --show-error-codes --ignore-missing-imports src

format:
	poetry run isort -v src
	poetry run black src

all:
	make format
	make lint