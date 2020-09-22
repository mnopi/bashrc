#!/usr/bin/env bash
# shellcheck disable=SC2034
export starting="${BASH_SOURCE[0]}"; debug.sh starting

if ! isuser.sh; then
  error.sh "can not be done with root"; exit 1
fi

if name="$( source gtop 2>&1 )"; then
  export name; debug.sh name
  path="$( git rev-parse --show-toplevel 2>&1 )"; export path; debug.sh path
  cd "${path}" || { error.sh cd "${path}"; exit 1; }
  branch="$( git rev-parse --abbrev-ref HEAD )"
  debug.sh branch
  if [[ "${branch}" != 'master' ]]; then
    git checkout master || exit 1
    git merge "${branch}" || exit 1
    gall.sh || exit 1
    info.sh "To delete local branch: " "git branch -d ${branch}"
    info.sh "To delete remote branch: " "git push origin --delete  ${branch}"
  else
    warning.sh merge "${name}" "${branch}"
  fi
  cd - > /dev/null || exit 1
else
  error.sh gall "${name}"; exit 1
fi

unset starting name path error branch
