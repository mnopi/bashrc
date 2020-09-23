BUMP := patch  # <major|minor|patch>
all: upload
.PHONY: all install secrets-push secrets-pull venv vars
SHELL := $(shell command -v bash)

upload:
	@bashrc-upload.sh $(BUMP)
install:
	@bashrc-install.sh "$${PASSWORD}" "$${INTERNET}"

secrets-push:
	@secrets-push.sh
secrets-pull:  ## other  not git source
	@secrets-pull.sh

venv:
	@project-venv.sh

vars:
	@echo $(BASHRC)
	@echo "PASSWORD: $${PASSWORD}"
	@echo "INTERNET: $${INTERNET}"
	@echo "BUMP: $(BUMP)"


