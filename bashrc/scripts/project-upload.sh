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

export bump twine path; debug.sh bump twine path

path="${path:-.}"
test -n "${path}" || { error.sh "Project Path" "must be specified"; exit 1; }
cd "${path}" > /dev/null 2>&1 || { error.sh "${path}" "invalid"; exit 1; }

virtual="${path}/venv/bin"
export virtual; debug.sh virtual

if ! test -d "${virtual}"; then
  python3.8 -m venv "${path}/venv"
fi

while read -r file; do
  python3.8 -m pip install --upgrade -r "${file}"
done < <( find "${path}" -type f -name "requirements*" )

if isuser.sh; then
  project-clean.sh "${path}" || exit 1
	gadd.sh || exit 1
  "${path}/bump2version" --allow-dirty "${bump:-patch}" || exit 1
  gpush.sh || exit 1
  "${path}/python3" setup.py sdist || exit 1
  "${path}/python3" setup.py bdist_wheel || exit 1
  "${path}/twine" upload -r "${twine}" || exit 1
  gmerge || exit 1
  project-clean.sh "${path}" || exit 1
  project-upgrade.sh "$( basename "${path}" )" || exit 1
else
  error.sh "${PROJECT_BASHRC}" "can not be uploaded with root"; exit 1
fi

cd - > /dev/null || exit 1

unset source bump twine virtual
