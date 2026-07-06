PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python

.PHONY: install test demo full clean

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install -e ".[dev]"

test:
	$(VENV_PYTHON) -m pytest -q

demo:
	$(VENV_PYTHON) -m textgrad_rl.run_experiment --config configs/mac_demo.json

full:
	$(VENV_PYTHON) -m textgrad_rl.run_experiment --config configs/mac_full.json

clean:
	rm -rf runs .pytest_cache **/__pycache__ *.egg-info build dist
