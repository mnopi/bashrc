#!/usr/bin/env bash
# source
# shellcheck disable=SC2034
export source="${BASH_SOURCE[0]}"; debug.sh source

path="$( command -v bashrc )"; export path
debug.sh path

## TODO: install for root??
# shellcheck disable=SC1090
if test -f "${path}"; then
  export BASHRC_USERS="$#"
  source "${path}"
else
  error.sh bashrc "not found"; return 1
fi

unset source path BASHRC_USERS
