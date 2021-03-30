#!/usr/bin/env bash
export starting="${BASH_SOURCE[0]}"; debug.sh starting

if ! test -e "${BREW}" || test -n "${1}"; then
    yes yes | /bin/bash -c \
             "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)" || exit 1
fi
