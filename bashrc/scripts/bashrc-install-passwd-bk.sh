#!/usr/bin/env bash
# source
export starting="${BASH_SOURCE[0]}"; debug.sh source

test -n "${PASSWD_PATH}" || { error.sh PASSWD_PATH 'not defined'; return 1; }
debug.sh PASSWD_PATH USERHOME

# shellcheck disable=SC1090
source "${PASSWD_PATH}"
debug.sh PASSWORD INTERNET GITHUB_PRIVATE_URL GITHUB_SECRETS_URL
echo "${PASSWORD}" | sudo -S true > /dev/null 2>&1

unset starting


