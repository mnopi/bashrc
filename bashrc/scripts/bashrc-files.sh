#!/usr/bin/env bash

test -n "${PASSWD_PATH}" || { error.bash PASSWD_PATH 'not defined'; return 1; }


if ! grep "${BASHRC_FILE}" ~/.bashrc; then
  echo "source \"$( command -v "${}" )\"" >> ~/.bashrc
fi
