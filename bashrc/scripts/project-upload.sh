#!/usr/bin/env bash
# ${1} - path
# ${2} - bump: <major|minor>
# ${3} - twine: <"${GITHUB_USERNAME}"|"${GITHUB_ORGANIZATION_ID}"|pypi>
# shellcheck disable=SC2034
export source="${BASH_SOURCE[0]}"; debug.sh source

if ! isuser.sh; then
  error.sh "can not be done with root"; exit 1
fi

while (( "$#" )); do
  case "${1}" in
    major) bump="${1}" ;;
    minor) bump="${1}" ;;
    "${GITHUB_USERNAME}") twine="${1}" ;;
    "${GITHUB_ORGANIZATION_ID}") twine="${1}" ;;
    pypi) twine="${1}" ;;
    *) path="${1}";;
  esac; shift
done

path="${path:-.}"
bump="${bump:-patch}"

cd "${path}" > /dev/null 2>&1 || { error.sh "${path}" "invalid"; exit 1; }

path="$( pwd )"
name="$( basename "${path}" )"
export bump twine path name; debug.sh bump twine path name

virtual="${path}/venv/bin"
export virtual; debug.sh virtual

if ! test -d "${virtual}"; then
  if error="$( python3.8 -m venv "${path}/venv" 2>&1 )"; then
    info.sh venv "${name}"
  else
    error.sh venv "${name}" "${error}"; exit 1
  fi
fi

# shellcheck disable=SC1090
source "${virtual}/activate"

while read -r file; do
  export file; debug.sh file
  if error="$( "${virtual}/python3" -m pip install --upgrade pip wheel setuptools && \
               "${virtual}/python3" -m pip install --upgrade -r "${file}" 2>&1 )"; then
    info.sh requirements "${name}"
  else
    error.sh requirements "${name}" "${error}"; exit 1
  fi
done < <( find "${path}" -mindepth 1 -maxdepth 2 -type f -name "requirements*" )

if isuser.sh; then
  project-clean.sh "${path}" || exit 1
	gadd.sh || exit 1
  if error="$( "${virtual}/bump2version" --allow-dirty "${bump}" 2>&1 )"; then
    info.sh bump2version "${name}"
  else
    error.sh bump2version "${name}" "${error}"; exit 1
  fi
  gpush.sh || exit 1
  if error="$( "${virtual}/python3" setup.py sdist 2>&1 )"; then
    info.sh sdist "${name}"
  else
    error.sh sdist "${name}" "${error}"; exit 1
  fi
  if error="$( "${virtual}/python3" setup.py bdist_wheel 2>&1 )"; then
    info.sh wheel "${name}"
  else
    error.sh wheel "${name}" "${error}"; exit 1
  fi
  if error="$( "${virtual}/twine" upload -r "${twine}" dist/* 2>&1 )"; then
    info.sh twine "${name}"
  else
    error.sh twine "${name}" "${error}"; exit 1
  fi
  gmerge.sh || exit 1
  project-clean.sh "${path}" || exit 1
  project-upgrade.sh "${name}" || exit 1
else
  error.sh "${PROJECT_BASHRC}" "can not be uploaded with root"; exit 1
fi

cd - > /dev/null || exit 1

unset source bump twine virtual file error name path
