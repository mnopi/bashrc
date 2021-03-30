#!/usr/bin/env bash
# shellcheck disable=SC2034
export starting="${BASH_SOURCE[0]}"; debug.sh starting

if ! isuser.sh; then
  error.sh "can not be done with root"; exit 1
fi

cp "${BAPY}/scripts/"* /usr/local/bin

unset starting name project_path error branch
