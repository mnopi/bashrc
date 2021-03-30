#!/usr/bin/env bash
# shellcheck disable=SC1090
# "${1}" - ACCOUNT PASSWORD
# "${2}" - INTERNET PASSWORD
export starting="${BASH_SOURCE[0]}"; debug.sh starting

function install_paswwd() {
  # "${1}" - ACCOUNT PASSWORD
  # "${2}" - INTERNET PASSWORD
  local PASSWORD INTERNET GITHUB_PRIVATE_URL GITHUB_SECRETS_URL
  test -n "${PASSWD_PATH}" || { error.sh PASSWD_PATH 'not defined'; return 1; }
  test -n "${GITHUB_USERNAME}" || { error.sh GITHUB_USERNAME 'not defined'; return 1; }

  if ! test -f "${PASSWD_PATH}"; then
    PASSWORD="${1}"
    if ! test -n "${PASSWORD}"; then
      read -r -p "$( blue.sh "Enter sudo password: " )" PASSWORD
    fi
    if test -n "${DARWIN}"; then
      INTERNET="${2}"
      if ! test -n "${INTERNET}"; then
        read -r -p "$( blue.sh "Enter internet/GitHub password: " )" INTERNET
      fi
    else
      INTERNET="${2:-${PASSWORD}}"
    fi
    if echo "${PASSWORD}" | sudo -S true; then
      info.sh passwd "sudo password."
    else
      error.sh passwd "sudo password."
      return 1
    fi
  GITHUB_PRIVATE_URL="https://${GITHUB_USERNAME}:${INTERNET}@github.com/${GITHUB_USERNAME}"
  # shellcheck disable=SC2034
  GITHUB_SECRETS_URL="${GITHUB_PRIVATE_URL}/.ssh"
  for var in PASSWORD INTERNET GITHUB_PRIVATE_URL GITHUB_SECRETS_URL; do
    echo "export ${var}='${!var}'" | tee -a "${PASSWD_PATH}" > /dev/null 2>&1
  done
    debug.sh PASSWD_PATH GITHUB_USERNAME
    info.sh "${PASSWD_PATH}" created
  fi
}

source "$( command -v bashrc-vars )" || exit 1
source "$( command -v bashrc-paths )" || exit 1
source "$( command -v bashrc-distro )" || exit 1
source "$( command -v bashrc-dirs )" || exit 1

install_paswwd "$@" || exit 1
source "${PASSWD_PATH}" || exit 1

sudoers.sh "$@" || exit 1
essentials.sh "$@" || exit 1
homefiles.sh || exit 1
sshconfig.sh || exit 1
gconfig.sh || exit 1
manpaths.sh || exit 1

## BEGIN: SECRETS
if test -f "${SECRETS_PATH}"; then
  source "${SECRETS_PATH}"
else
  error.sh install "${SECRETS_PATH}" "not found"; exit 1
fi
## END: SECRETS

source "$( command -v bashrc-misc )" || exit 1

unset starting
