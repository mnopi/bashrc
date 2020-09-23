#!/usr/bin/env bash
# shellcheck disable=SC2034
export starting="${BASH_SOURCE[0]}"; debug.sh starting

if isuserdarwin.sh; then
  test -n "${GITHUB_SECRETS_PATH}" || { error.sh GITHUB_SECRETS_PATH 'not defined'; exit 1; }
  cd "${GITHUB_SECRETS_PATH}" || { error.sh "${GITHUB_SECRETS_PATH}" 'not found'; exit 1; }
  sudo chown -R "${USERNAME}":"$( id -g "${USERNAME}" )" "${USERNAME}/.ssh"
  gpull.sh
  cd - > /dev/null || exit 1
fi
unset starting
