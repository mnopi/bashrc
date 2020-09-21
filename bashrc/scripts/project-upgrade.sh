#!/usr/bin/env bash
# ${1} - name
# shellcheck disable=SC2034
export starting="${BASH_SOURCE[0]}"; debug.sh starting

unset VIRTUAL_ENV PYTHONHOME
deactivate > /dev/null 2>&1

test -n "${DARWIN}" || sudo=sudo
echo $name
if error="$( /usr/local/bin/pip3.8 install --upgrade "${1}" 2>&1 )"; then
  info.sh upgrade "${1}"
else
  error.sh upgrade "${1}" "${error}"; exit 1
fi

## I do not know why requires twice to get it into the path,
${sudo} /usr/local/bin/pip3.8 install --upgrade "${1}" > /dev/null 2>&1

unset starting error path name
