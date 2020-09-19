#!/usr/bin/env bash
# shellcheck disable=SC2034
export source="${BASH_SOURCE[0]}"; debug.sh source

test -n "${PASSWD_PATH}" || { error.bash PASSWD_PATH 'not defined'; return 1; }


if ! grep "${BASHRC_FILE}" ~/.bashrc; then
  echo "source \"$( command -v "${BASHRC_FILE}" )\"" >> ~/.bashrc
fi

unset source
