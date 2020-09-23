#!/usr/bin/env bash
# shellcheck disable=SC2034
export starting="${BASH_SOURCE[0]}"; debug.sh starting

if ! isuser.sh; then
  error.sh "can not be done with root"; exit 1
fi

case "${1}" in
  major) bump="${1}" ;;
  minor) bump="${1}" ;;
  patch) bump="${1}" ;;
esac; shift

test -n "${BASHRC}" || { error.sh " BASHRC" "empty"; exit 1; }
cd "${BASHRC}" > /dev/null 2>&1 || { error.sh "${BASHRC}" "does not exists"; exit 1; }

unset VIRTUAL_ENV PYTHONHOME
deactivate > /dev/null 2>&1

project-upload.sh "${BASHRC}" "${bump}" "${GITHUB_USERNAME}" || exit 1

unset VIRTUAL_ENV PYTHONHOME
deactivate > /dev/null 2>&1

project-upgrade.sh "$( basename "${BASHRC}" )" || exit 1

cd - > /dev/null || exit 1

unset starting bump
