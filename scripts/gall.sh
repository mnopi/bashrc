#!/usr/bin/env bash
# shellcheck disable=SC2034
export starting="${BASH_SOURCE[0]}"; debug.sh starting
[[ ! "${TIMES-}" ]] || start="$( date +%s )"

if ! isuser.sh; then
  error.sh "can not be done with root"; exit 1
fi

message="${1:-'auto commit'}"
if name="$( source gtop 2>&1 )"; then
  export name; debug.sh name
  project_path="$( git rev-parse --show-toplevel 2>&1 )" || exit 1; export project_path; debug.sh project_path
  cd "${project_path}" || { error.sh cd "${project_path}"; exit 1; }
  if error="$( gadd.sh 2>&1 && gcommit.sh "${message}" 2>&1 && gpush.sh 2>&1 && gpull.sh 2>&1 )"; then
    info.sh "${name}" gall; exit 0
  else
    error.sh "${name}" gall "${error}"; exit 1
  fi
  cd - > /dev/null || exit 1
else
  error.sh "${name}" gall; exit 1
fi
[[ ! "${TIMES-}" ]] || { end="$( date +%s )"; export runtime=$((end-start)); times.sh starting runtime; }
unset start end runtime

unset starting name project_path error
