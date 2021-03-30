#!/usr/bin/env bash
# shellcheck disable=SC2034
export starting="${BASH_SOURCE[0]}"; debug.sh starting

if ! isuser.sh; then
  error.sh "can not be done with root"; exit 1
fi

if name="$( source gtop 2>&1 )"; then
  export name; debug.sh name
  project_path="$( git rev-parse --show-toplevel 2>&1 )"; export project_path; debug.sh project_path
  cd "${project_path}" || { error.sh cd "${project_path}"; exit 1; }
  if error="$( git add . --all 2>&1 )"; then
    info.sh "${name}" add; exit 0
  else
    error.sh "${name}" add "${error}"; exit 1
  fi
  cd - > /dev/null || exit 1
else
  error.sh "${name}" gall; exit 1
fi

unset starting name project_path error
