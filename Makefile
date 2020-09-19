BUMP := patch  # major|minor|path
all: upload
.PHONY: venv all vars clean $(all)
SHELL := $(shell command -v bash)
DIR := $(shell dirname "$(realpath $(firstword $(MAKEFILE_LIST)))")
PROJECT := $(shell basename $(DIR))
COMMAND := source $(DIR)/venv/bin/activate; cd $(DIR)/$(PROJECT);
ACTIVATE := $(DIR)/venv/bin/activate

vars:
	@echo "BUMP: $(BUMP)"
	@echo "CURRENT: $(CURRENT)"
	@echo "DIR: $(DIR)"
	@echo "PROJECT: $(PROJECT)"

venv:
	@project-venv.sh
clean:
	@project-clean.sh

upload:
	@bashrc-upload.sh $(BUMP)
