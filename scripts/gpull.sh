#!/usr/bin/env bash
# shellcheck disable=SC2034
export starting="${BASH_SOURCE[0]}"; debug.sh starting
[[ ! "${TIMES-}" ]] || start="$( date +%s )"

if ! isuser.sh; then
  error.sh "can not be done with root"; exit 1
fi

if name="$( source gtop 2>&1 )"; then
  export name; debug.sh name
  project_path="$( git rev-parse --show-toplevel 2>&1 )"; export project_path; debug.sh project_path
  cd "${project_path}" || { error.sh cd "${project_path}"; exit 1; }
    if error="$( git pull --no-rebase --quiet --no-stat 2>&1 )"; then
      info.sh "${name}" pull; exit 0
    else
      branch="$( git rev-parse --abbrev-ref HEAD )"
      if error1="$( git pull --no-rebase --quiet --no-stat origin "${branch}" 2>&1 )"; then
        info.sh "${name}" pull; exit 0
      else
        error.sh "${name}" pull "${error}\n ${error1}"; exit 1
      fi
    fi
  cd - > /dev/null || exit 1
else
  error.sh "${name}" pull; exit 1
fi
[[ ! "${TIMES-}" ]] || { end="$( date +%s )"; export runtime=$((end-start)); times.sh starting runtime; }
unset start end runtime

unset starting name project_path error error1
