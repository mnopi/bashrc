#!/usr/bin/env bash
export starting="${BASH_SOURCE[0]}"; debug.sh starting
[[ ! "${TIMES-}" ]] || start="$( date +%s )"

if ! test -e "${BREW}" || test -n "${1}"; then
    yes yes | /bin/bash -c \
             "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)" || exit 1
fi
[[ ! "${TIMES-}" ]] || { end="$( date +%s )"; export runtime=$((end-start)); times.sh starting runtime; }
unset start end runtime starting
