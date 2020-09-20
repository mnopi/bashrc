#!/usr/bin/env bash
# shellcheck disable=SC1090
export starting="${BASH_SOURCE[0]}"; debug.sh starting

## BEGIN: INIT
source_name="bashrc-init"
source_path="$( command -v "${source_name}" )"; export source_path; debug.sh source_path

if test -f "${source_path}"; then
  source "${source_path}"
else
  error.sh install "${source_name}" "not found"; exit 1
fi
## END: INIT

## BEGIN: PASSWD
script_name="bashrc-install-passwd.sh"
script_path="$( command -v "${script_name}" )"; export script_path; debug.sh script_path
if test -f "${script_path}"; then
  "${script_path}"
else
  error.sh install "${script_name}" "not found"; exit 1
fi

if test -f "${PASSWD_PATH}"; then
  source "${PASSWD_PATH}"
else
  error.sh install "${PASSWD_PATH}" "not found"; exit 1
fi
## END: PASSWD

## BEGIN: BOOTSTRAP
script_name="bashrc-bootstrap.sh"
script_path="$( command -v "${script_name}" )"; export script_path; debug.sh script_path

if test -f "${script_path}"; then
  "${script_path}"
else
  error.sh install "${script_name}" "not found"; exit 1
fi
## END: BOOTSTRAP

## BEGIN: SECRETS
if test -f "${SECRETS_PATH}"; then
  source "${SECRETS_PATH}"
else
  error.sh install "${SECRETS_PATH}" "not found"; exit 1
fi
## END: SECRETS

unset starting source_name source_path script_name script_path
