#!/usr/bin/env bash
# shellcheck disable=SC2034
export starting="${BASH_SOURCE[0]}"; debug.sh source
[[ ! "${TIMES-}" ]] || start="$( date +%s )"

if ! isuser.sh; then
  error.sh "can not be done with root"; exit 1
fi

message="${1:-'auto commit'}"
if name="$( source gtop 2>&1 )"; then
  export name; debug.sh name
  project_path="$( git rev-parse --show-toplevel 2>&1 )"; export project_path; debug.sh project_path
  cd "${project_path}" || { error.sh cd "${project_path}"; exit 1; }
  if [[ "$( git status --porcelain | wc -l | sed 's/ //g' > /dev/null 2>&1 )" == "0" ]]; then
    exit 0
  else
    if error="$( git commit --quiet -a -m "${message}" 2>&1 )"; then
      info.sh "${name}" commit; exit 0
    else
      if echo "${error}" | grep "nothing to commit, working tree clean"  > /dev/null 2>&1; then
        info.sh "${name}" commit; exit 0
      else
        error.sh "${name}" commit "${error}"; exit 1
      fi
    fi
  fi
  cd - > /dev/null || exit 1
else
  error.sh "${name}" gall; exit 1
fi
[[ ! "${TIMES-}" ]] || { end="$( date +%s )"; export runtime=$((end-start)); times.sh starting runtime; }
unset start end runtime

unset starting name project_path error
