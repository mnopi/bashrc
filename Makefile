BUMP := patch  # major|minor|path
all: upload
.PHONY: $(all) all install secrets-push venv vars
SHELL := $(shell command -v bash)
export PROJECT_BASHRC := $(HOME)/bashrc
export TERM := xterm-256color
export GITHUB_USERNAME := j5pu
export GITHUB_EMAIL := j5pu@icloud.com
upload:
	@bashrc-upload.sh $(BUMP)
install:
	@bashrc-install.sh "$${PASSWORD}" "$${INTERNET}"

secrets-push:
	@secrets-push.sh
venv:
	@project-venv.sh

vars:
	@echo $(PROJECT_BASHRC)
	@echo "PASSWORD: $${PASSWORD}"
	@echo "INTERNET: $${INTERNET}"
	@echo "BUMP: $(BUMP)"

upgrade:  ## other  not git source
	@bashrc-upgrade.sh
install-other:  ## other  not git source
	@bashrc-install.sh "$${INTERNET}"
secrets-pull:  ## other  not git source
	@secrets-pull.sh
