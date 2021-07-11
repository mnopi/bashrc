BUMP := patch  # <major|minor|patch>
all: install
.PHONY: venv requirements publish all
SHELL := $(shell command -v bash)
DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PACKAGE := $(shell basename $(DIR))
VENV := $(DIR)venv
ACTIVATE := $(VENV)/bin/activate

venv:
	@cd $(DIR); test -d $(VENV) || python3.9 -m venv $(VENV)

requirements: venv
	@cd $(DIR); source $(ACTIVATE); $(VENV)/bin/python3.9 -m pip install --upgrade -q -r $(DIR)requirements_dev.txt; \
deactivate

publish: requirements
	@cd $(DIR); source $(ACTIVATE); gall.sh; bump2version $(BUMP); gall.sh; \
flit publish --repository $$GITHUB_ORGANIZATION_USERNAME; rm -rf $(DIR)dist/; deactivate

install: publish
	@cd $(DIR); deactivate >/dev/null 2>&1; /usr/local/bin/python3.9 -m pip install --upgrade -q $(PACKAGE); \
/usr/local/bin/python3.9 -m pip install --upgrade -q $(PACKAGE)
