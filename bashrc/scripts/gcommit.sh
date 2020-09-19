#!/usr/bin/env bash
# shellcheck disable=SC2034
export source="${BASH_SOURCE[0]}"; debug.sh source

if ! isuser.sh; then
  error.sh "can not be done with root"; exit 1
fi

message="${1:-'auto commit'}"
if name="$( source gtop 2>&1 )"; then
  export name; debug.sh name
  path="$( git rev-parse --show-toplevel 2>&1 )"; export path; debug.sh path
  cd "${path}" || { error.sh cd "${path}"; exit 1; }
  if [[ "$( git status --porcelain | wc -l | sed 's/ //g' > /dev/null 2>&1 )" == "0" ]]; then
    exit 0
  else
    if error="$( git commit --quiet -a -m "${message}" 2>&1 )"; then
      info.sh commit "${name}"; exit 0
    else
      if echo "${error}" | grep "nothing to commit, working tree clean"  > /dev/null 2>&1; then
        info.sh commit "${name}"; exit 0
      else
        error.sh commit "${name}" "${error}"; exit 1
      fi
    fi
  fi
  cd - > /dev/null || exit 1
else
  error.sh gall "${name}"; exit 1
fi

unset source name path error
