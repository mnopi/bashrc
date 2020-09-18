#!/usr/bin/env bash

test -n "${PROJECT_BASHRC}" || { error.sh "PROJECT_BASHRC" "empty"; exit 1; }
cd "${PROJECT_BASHRC}" > /dev/null 2>&1 || { error.sh "${PROJECT_BASHRC}" "does not exists"; exit 1; }

if isuser.sh; then
  clean-project.sh /ala
	git add . --all
	bash -c '$(COMMAND) cd $(DIR); bump2version --allow-dirty $(BUMP)'
	git push -u origin $(BRANCH) --tags

else
  error.sh "${PROJECT_BASHRC}" "can not be uploaded with root"; exit 1
fi
