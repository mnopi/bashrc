#!/usr/bin/env bash
# ${1} - name
# shellcheck disable=SC2034
export starting="${BASH_SOURCE[0]}"; debug.sh starting

name="$( basename pwd )"

unset VIRTUAL_ENV PYTHONHOME
deactivate > /dev/null 2>&1

command="/usr/local/bin/python3.8"
if ! test -n "${DARWIN}"; then
  command="sudo /bin/python3.8"
fi

if error="$( ${command} -m pip install --upgrade "${1:-${name}}" 2>&1 )"; then
  info.sh upgrade "${1:-${name}}"
else
  error.sh upgrade "${1:-${name}}" "${error}"; exit 1
fi

## I do not know why requires twice to get it into the path,
${command} -m pip install --upgrade "${1-${name}}" > /dev/null 2>&1

unset starting error command
