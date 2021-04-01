#!/usr/bin/env bash
# shellcheck disable=SC2034
export starting="${BASH_SOURCE[0]}"; debug.sh starting

test -n "${GITHUB_SECRETS_PATH}" || { error.sh GITHUB_SECRETS_PATH 'not defined'; exit 1; }
cd "${GITHUB_SECRETS_PATH}" || { error.sh "${GITHUB_SECRETS_PATH}" 'not found'; exit 1; }
sudo chown -R "${USERNAME}":"$( id -g "${USERNAME}" )" "${USERHOME}/.ssh"

if isuserdarwin.sh; then
  gall.sh
elif isuser.sh && test -z "${DARWIN}"; then
  gpull.sh
fi
cd - > /dev/null || exit 1

unset starting
