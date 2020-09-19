#!/usr/bin/env bash
# shellcheck disable=SC2034
export source="${BASH_SOURCE[0]}"; debug.sh source

test -n "${BASHRC_FILE}" || { error.bash BASHRC_FILE 'not defined'; return 1; }

if ! grep source ~/.bashrc | grep "${BASHRC_FILE}" > /dev/null 2>&1; then
  if bashrc_path="$( command -v "${BASHRC_FILE}" 2>&1 )"; then
    tee -a ~/.bashrc >/dev/null <<EOT
if test -f  \"${bashrc_path}\";then
  source \"${bashrc_path}\"
else
  echo 'bashrc file not found'; return 1
fi
EOT
    info.sh files .bashrc "${bashrc_path}"
  else
    error.sh files .bashrc "${BASHRC_FILE} - command not found"
  fi
fi

unset source bashrc_path
