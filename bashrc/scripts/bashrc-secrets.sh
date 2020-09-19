#!/usr/bin/env bash
# shellcheck disable=SC2034
export source="${BASH_SOURCE[0]}"; debug.sh source

test -n "${GITHUB_SECRETS_FILE_URL}" || { error.sh GITHUB_SECRETS_FILE_URL 'not defined'; return 1; }

path="$( command -v bashrc-secrets )"; export path
debug.sh "${path}"
# shellcheck disable=SC1090
if test -f "${path}"; then
  source "${path}"
else
  error.sh bashrc-secrets "not found"; return 1
fi

if error="$( curl -sL "${GITHUB_SECRETS_FILE_URL}" "${SECRETS_PATH}" --quiet  2>&1 )"; then
  info.sh curl "${GITHUB_SECRETS_URL}"; return 0
else
  error.sh curl "${GITHUB_SECRETS_URL}" "${error}"; return 1
fi

unset source path error
