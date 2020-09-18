#!/usr/bin/env bash
# source

test -n "${PASSWD_PATH}" || { error.bash PASSWD_PATH 'not defined'; return 1; }

if ! test -f "${PASSWD_PATH}"; then
read -r -p "$( blue "Enter sudo password: " )" password
if test -n "${DARWIN}"; then
  read -r -p "$( blue "Enter internet/GitHub password: " )" internet
else
  internet="${password}"
fi
if echo "${password}" | sudo -S true; then
  info constants "sudo password."
else
  error constants "sudo password."
  return 1
fi
tee /tmp/bashrc >/dev/null <<EOT
export PASSWORD="${password}"
export INTERNET="${internet}"
EOT
info.bash "${PASSWD_PATH}" created
return 0
fi
