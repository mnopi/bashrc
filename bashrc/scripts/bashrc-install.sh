#!/usr/bin/env bash
# source
# shellcheck disable=SC2034
export source="${BASH_SOURCE[0]}"; debug.sh source

path="$( command -v bashrc )"; export path
debug.sh "${path}"

# shellcheck disable=SC1090
if test -f "${path}"; then
  source "${path}"
else
  error.sh bashrc "not found"; return 1
fi

unset source path