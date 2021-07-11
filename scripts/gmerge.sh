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
  branch="$( git rev-parse --abbrev-ref HEAD )"
  debug.sh branch
  if [[ "${branch}" != 'main' ]]; then
    git checkout main || exit 1
    git merge "${branch}" || exit 1
    gall.sh || exit 1
    info.sh "${name}" merge "${branch}"
    warning.sh "${name}" "To delete local branch: " "git branch -d ${branch}"
    warning.sh "${name}" "To delete remote branch: " "git push origin --delete  ${branch}"
  else
    info.sh "${name}" merge "${branch}"
  fi
  cd - > /dev/null || exit 1
else
  error.sh "${name}" gall; exit 1
fi
[[ ! "${TIMES-}" ]] || { end="$( date +%s )"; export runtime=$((end-start)); times.sh starting runtime; }
unset start end runtime

unset starting name project_path error branch
