#!/usr/bin/env bash
# shellcheck disable=SC2034
export starting="${BASH_SOURCE[0]}"; debug.sh starting

test -n "${PROJECT_BASHRC}" || { error.sh "PROJECT_BASHRC" "empty"; exit 1; }

unset VIRTUAL_ENV PYTHONHOME
deactivate > /dev/null 2>&1

project-upgrade.sh "${PROJECT_BASHRC}" || exit 1

unset starting
