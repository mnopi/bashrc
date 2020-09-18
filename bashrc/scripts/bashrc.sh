#!/usr/bin/env bash

export USERNAME="jose"
USERHOME="$( [[ "$(uname -s)" == "Darwin" ]] && echo "/Users" || echo "/home" )/${USERNAME}"; export USERHOME
export PASSWD_PATH="${USERHOME}/.passwd"
export BASHRC_FILE="bashrc.sh"
export PROJECT_BASHRC="${USERHOME}/bashrc"

DARWIN="$(uname -a | grep -i darwin 2>/dev/null)"; export DARWIN
KALI="$(uname -a | grep -i kali 2>/dev/null)"; export KALI
DEBIAN="$(uname -a | grep -i debian 2>/dev/null)"; export DEBIAN
UBUNTU="$(uname -a | grep -i ubuntu 2>/dev/null)"; export UBUNTU


for script in passwd.sh bashfiles.sh vars.sh dirs.sh; do
  path="$( command -v vars.sh )"
  # shellcheck disable=SC1090
  if test -f "${path}"; then
    debug.sh "${path}"
    source "${path}"
  else
    error.sh "${script}" "not found"; return 1
  fi
done
