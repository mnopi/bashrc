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

project-venv-git-source:
	@project-venv.sh
project-clean-git-source:
	@project-clean.sh

bashrc-upload-git-source:
	@bashrc-upload.sh $(BUMP)

bashrc-upgrade-other:
	@bashrc-upgrade.sh

bashrc-install-macos:
	@bashrc-install.sh "${PASSWORD}" "${INTERNET}"
bashrc-install-other:
	@bashrc-install.sh "${INTERNET}"

## TODO: el install hace un link con los otros usuarios
secrets-push-git-source:
	@secrets-push.sh
secrets-pull-other:
	@secrets-pull.sh

