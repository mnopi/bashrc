BUMP := patch  # major|minor|path
.DEFAULT_GOAL := help
all: clean-build clean-pyc clean-test darglint doctest pytest add bump push build upload install merge
.PHONY: all help vars clean test mypy git dist $(all)
export PRINT_HELP_PYSCRIPT
SHELL := $(shell command -v bash)
BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
CURRENT := $(shell git describe --abbrev=0 --tags 2>/dev/null; true)
DIR := $(shell dirname "$(realpath $(firstword $(MAKEFILE_LIST)))")
PROJECT := $(shell basename $(DIR))
COMMAND := source $(DIR)/venv/bin/activate; cd $(DIR)/$(PROJECT);
ACTIVATE := $(DIR)/venv/bin/activate
define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
help: ## print Makefile help
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)
vars:  ## print Makefile vars
	@echo "BUMP: $(BUMP)"
	@echo "BRANCH: $(BRANCH)"
	@echo "CURRENT: $(CURRENT)"
	@echo "DIR: $(DIR)"
	@echo "PROJECT: $(PROJECT)"

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts
clean-build:  ## remove build artifacts
	@/bin/rm -fr build/ > /dev/null
	@/bin/rm -fr dist/ > /dev/null
	@/bin/rm -fr .eggs/ > /dev/null
	@find . -name '*.egg-info' -exec /bin/rm -fr {} +
	@find . -name '*.egg' -exec /bin/rm -f {} +
clean-pyc:  ## remove Python file artifacts
	@find . -name '*.pyc' -exec /bin/rm -f {} +
	@find . -name '*.pyo' -exec /bin/rm -f {} +
	@find . -name '*~' -exec /bin/rm -f {} +
	@find . -name '__pycache__' -exec /bin/rm -fr {} +
clean-test:  ## remove test and coverage artifacts
	@/bin/rm -fr .tox/
	@/bin/rm -fr .pytest_cache
	@find . -name '.mypy_cache' -exec /bin/rm -rf {} +

up:
	git add . --all
	bump2version --allow-dirty $(BUMP)
	git push -u origin $(BRANCH) --tags
	python setup.py sdist
	python setup.py bdist_wheel
	twine upload -r j5pu dist/*
	/usr/local/bin/pip3.8 install -vvvv --upgrade $(PROJECT)


merge:  ## installs in system
ifneq ($(BRANCH),master)
	@echo "BRANCH: $(BRANCH)"
	@git checkout master
	@git merge $(BRANCH)
	@echo "BRANCH: $$(git rev-parse --abbrev-ref HEAD)"
	@gall
	@echo "To delete locally: git branch -d $(BRANCH)"
	@echo "To delete locally: git push origin --delete  $(BRANCH)"
else
	@echo "BRANCH: $(BRANCH) - Nothing to do here."
endif
