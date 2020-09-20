#!/usr/bin/env bash
export starting="${BASH_SOURCE[0]}"; debug.sh starting

test -n "${PASSWD_PATH}" || { error.sh PASSWD_PATH 'not defined'; return 1; }
debug.sh PASSWD_PATH USERHOME

if ! test -f "${PASSWD_PATH}"; then
  PASSWORD="${1}"
  if ! test -n "${PASSWORD}"; then
    read -r -p "$( blue.sh "Enter sudo password: " )" PASSWORD
  fi
  export PASSWORD
  if test -n "${DARWIN}"; then
    INTERNET="${2}"
    if ! test -n "${INTERNET}"; then
      read -r -p "$( blue.sh "Enter internet/GitHub password: " )" INTERNET
    fi
  else
    INTERNET="${PASSWORD}"
  fi
  export INTERNET
  if echo "${PASSWORD}" | sudo -S true; then
    info.sh passwd "sudo password."
  else
    error.sh passwd "sudo password."
    return 1
  fi
export GITHUB_PRIVATE_URL="https://${GITHUB_USERNAME}:${INTERNET}@github.com/${GITHUB_USERNAME}"
export GITHUB_SECRETS_URL="${GITHUB_PRIVATE_URL}/.ssh"
for var in PASSWORD INTERNET GITHUB_PRIVATE_URL GITHUB_SECRETS_URL; do
  echo "export ${var}='${!var}'" | tee -a "${PASSWD_PATH}" > /dev/null 2>&1
done
  info.sh "${PASSWD_PATH}" created
fi

unset starting var
