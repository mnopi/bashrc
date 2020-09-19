#!/usr/bin/env bash
# ${1} - name
# shellcheck disable=SC2034
export source="${BASH_SOURCE[0]}"; debug.sh source

path="${1:-.}"

cd "${path}" > /dev/null 2>&1 || { error.sh "${path}" "invalid"; exit 1; }

path="$( pwd )"
name="$( basename "${path}" )"
export path name; debug.sh path name

deactivate > /dev/null 2>&1

if error="$( /usr/local/bin/pip3.8 install -vvvv --upgrade "${name}" 2>&1 )"; then
  info.sh upgrade "${name}"
else
  error.sh upgrade "${name}" "${error}"; exit 1
fi

unset source error path name
