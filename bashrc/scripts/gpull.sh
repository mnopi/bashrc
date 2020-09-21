#!/usr/bin/env bash
# shellcheck disable=SC2034
export starting="${BASH_SOURCE[0]}"; debug.sh starting

if ! isuser.sh; then
  error.sh "can not be done with root"; exit 1
fi

if path="$( git rev-parse --show-toplevel 2>&1 )"; then
  cd "${path}" || { error.sh cd "${path}"; exit 1; }
  name="$( basename "${path}" )"
else
  error.sh "${BASH_SOURCE[0]}" "not a git dir"; exit 1
fi

export name; debug.sh name
path="$( git rev-parse --show-toplevel 2>&1 )"; export path; debug.sh path
cd "${path}" || { error.sh cd "${path}"; exit 1; }
  if error="$( git pull --no-rebase --quiet --no-stat 2>&1 )"; then
    info.sh pull "${name}"; exit 0
  else
    if error1="$( git pull --no-rebase --quiet --no-stat origin master 2>&1 )"; then
      info.sh pull "${name}"; exit 0
    else
      error.sh "pull ${name}" "${error}\n ${error1}"; exit 1
    fi
  fi
cd - > /dev/null || exit 1


unset starting name path error error1
