BUMP := patch  # major|minor|path
all: upload
.PHONY: $(all) all install secrets-push venv vars
SHELL := $(shell command -v bash)
#PASSWORD := $(shell bash -l -c echo "${PASSWORD}" )
INTERNET := $(shell echo $$INTERNET )
PASSWORD = $${PASSWORD}
upload:
	@bashrc-upload.sh $(BUMP)
install:
	@bashrc-install.sh "$${PASSWORD}" "$${INTERNET}"
	@bashrc-install.sh aaaa "Zaragoza$."

secrets-push:
	@secrets-push.sh
venv:
	@project-venv.sh

vars:
	@echo "PASSWORD: $${PASSWORD}"
	@echo "INTERNET: $${INTERNET}"
	@echo "BUMP: $(BUMP)"

upgrade:  ## other  not git source
	@bashrc-upgrade.sh
install-other:  ## other  not git source
	@bashrc-install.sh "$${INTERNET}"
secrets-pull:  ## other  not git source
	@secrets-pull.sh
