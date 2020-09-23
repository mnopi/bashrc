#!/usr/bin/env bash
# shellcheck disable=SC2034
export starting="${BASH_SOURCE[0]}"; debug.sh starting

unset VIRTUAL_ENV PYTHONHOME
deactivate > /dev/null 2>&1

project-upgrade.sh "$( basename "${BASHRC}" )" || exit 1

unset starting
