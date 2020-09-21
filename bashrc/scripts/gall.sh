#!/usr/bin/env bash
# shellcheck disable=SC2034
export starting="${BASH_SOURCE[0]}"; debug.sh starting

if ! isuser.sh; then
  error.sh "can not be done with root"; exit 1
fi

message="${1:-'auto commit'}"
if name="$( source gtop 2>&1 )"; then
  export name; debug.sh name
  path="$( git rev-parse --show-toplevel 2>&1 )" || exit 1; export path; debug.sh path
  cd "${path}" || { error.sh cd "${path}"; exit 1; }
  if error="$( gadd.sh 2>&1 && gcommit.sh "${message}" 2>&1 && gpush.sh 2>&1 && gpull.sh 2>&1 )"; then
    info.sh gall "${name}"; exit 0
  else
    error.sh gall "${name}" "${error}"; exit 1
  fi
  cd - > /dev/null || exit 1
else
  error.sh gall "${name}"; exit 1
fi

unset starting name path error
