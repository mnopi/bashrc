#!/usr/bin/env bash
# shellcheck disable=SC1090
# shellcheck disable=SC2034
# ${1} - path
# ${2} - bump: <major|minor>
# ${3} - twine: <"${GITHUB_USERNAME}"|"${NFERX_GITHUB_USERNAME}"|pypi>
# ${4} - site (default use virtual environment if no defined)
export starting="${BASH_SOURCE[0]}"; debug.sh starting

if ! isuser.sh; then
  error.sh "can not be done with root"; exit 1
fi

while (( "$#" )); do
  case "${1}" in
    major) bump="${1}" ;;
    minor) bump="${1}" ;;
    patch) bump="${1}" ;;
    "${GITHUB_USERNAME}") twine="${1}" ;;
    "${NFERX_GITHUB_USERNAME}") twine="${1}" ;;
    pypi) twine="${1}" ;;
    git) twine="${1}"; git=git ;;
    site) site="${1}" ;;
    *) path="${1}";;
  esac; shift
done

path="${path:-.}"
bump="${bump:-patch}"
test -n "${twine}" || { error.sh "twine repository" "empty"; exit 1; }

cd "${path}" > /dev/null 2>&1 || { error.sh "${path}" "invalid"; exit 1; }

path="$( pwd )"
name="$( basename "${path}" )"
export bump twine path name; debug.sh bump twine path name

if isuserdarwin.sh; then
  if test -z "${site}"; then
    project-venv.sh "${path}"
    virtual="${path}/venv/bin/"
    source "${virtual}activate"
    export virtual; debug.sh virtual
  fi
  project-clean.sh "${path}" || exit 1
  find "${path}" -type d -name scripts -exec chmod -R +x "{}" \;
	gadd.sh || exit 1
  if error="$( "${virtual}bump2version" --allow-dirty "${bump}" 2>&1 )"; then
    info.sh bump2version "${name}"
  else
    error.sh bump2version "${name}" "${error}"; exit 1
  fi
  gpush.sh || exit 1
  if error="$( "${virtual}python3" setup.py sdist 2>&1 )"; then
    info.sh sdist "${name}"
  else
    error.sh sdist "${name}" "${error}"; exit 1
  fi
  if error="$( "${virtual}python3" setup.py bdist_wheel 2>&1 )"; then
    info.sh wheel "${name}"
  else
    error.sh wheel "${name}" "${error}"; exit 1
  fi
  if [[ "${twine}" = 'git' ]] ; then
    warning.sh twine "${name}" "${twine}"
  else
    if error="$( "${virtual}twine" upload -r "${twine}" dist/* 2>&1 )"; then
      info.sh twine "${name}"
    else
      error.sh twine "${name}" "${error}"; exit 1
    fi
  fi
  gmerge.sh || exit 1
  project-clean.sh "${path}" || exit 1
  unset VIRTUAL_ENV PYTHONHOME
  deactivate > /dev/null 2>&1
  project-upgrade.sh "${name}" || exit 1
else
  error.sh "${PROJECT_BASHRC}" "can not be uploaded with root"; exit 1
fi

cd - > /dev/null || exit 1

unset starting bump twine virtual file error name path
