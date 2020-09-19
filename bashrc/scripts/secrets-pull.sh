#!/usr/bin/env bash
# shellcheck disable=SC2034
export source="${BASH_SOURCE[0]}"; debug.sh source

test -n "${GITHUB_SECRETS_PATH}" || { error.sh GITHUB_SECRETS_PATH 'not defined'; exit 1; }
cd "${GITHUB_SECRETS_PATH}" || { error.sh "${GITHUB_SECRETS_PATH}" 'not found'; exit 1; }
gpull.sh
cd - > /dev/null || exit 1
unset source
