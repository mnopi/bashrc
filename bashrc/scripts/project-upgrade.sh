#!/usr/bin/env bash
# ${1} - name
# shellcheck disable=SC2034
export starting="${BASH_SOURCE[0]}"; debug.sh starting

unset VIRTUAL_ENV PYTHONHOME
deactivate > /dev/null 2>&1

basename="$( basename "$( pwd )" )"
name="${1:-${basename}}"
[[ "${name}" != "${PEN_BASENAME}" ]]  || name="${PEN_GIT}"

command="/usr/local/bin/python3.8"
if ! test -n "${DARWIN}"; then
  command="sudo /bin/python3.8"
fi

if error="$( ${command} -m pip uninstall "${name}" -y 2>&1 )"; then
  info.sh uninstall "${name}"
else
  error.sh uninstall "${name}" "${error}"; exit 1
fi

if error="$( ${command} -m pip install --upgrade "${name}" 2>&1 )"; then
  info.sh upgrade "${name}"
else
  error.sh upgrade "${name}" "${error}"; exit 1
fi

unset starting error command
