#!/usr/bin/env bash
# ${1} - name
# shellcheck disable=SC2034
export source="${BASH_SOURCE[0]}"; debug.sh source

test -n "${1}" || { error.sh "Project" "empty"; exit 1; }

if error="$( /usr/local/bin/pip3.8 install -vvvv --upgrade "${1}" 2>&1 )"; then
  info.sh upgrade "${1}"
else
  error.sh upgrade "${1}" "${error}"; exit 1
fi

unset source error
