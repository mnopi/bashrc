#!/usr/bin/env bash
export starting="${BASH_SOURCE[0]}"; debug.sh starting
test -n "${DARWIN}" || { error.sh DARWIN 'not defined'; exit 1; }
test -n "${KALI}" || { error.sh KALI 'not defined'; exit 1; }
test -n "${DEBIAN}" || { error.sh DEBIAN 'not defined'; exit 1; }
test -n "${UBUNTU}" || { error.sh UBUNTU 'not defined'; exit 1; }

function darwin() {
  # "${1}" - force
  local group
  for group in staff admin wheel; do
    if ! test -f "/etc/sudoers.d/${group}" || test "${1}" = force; then
      echo "${1}"
#      echo "${PASSWORD}" | sudo -S true
#      echo "%${group} ALL=(ALL) NOPASSWD:ALL" | sudo tee "/etc/sudoers.d/${group}" && info sudoers "${group}"
#      echo 'Defaults !env_reset' | sudo tee -a "/etc/sudoers.d/${group}" >/dev/null 2>&1
#      echo 'Defaults env_delete = "HOME"' | sudo tee -a "/etc/sudoers.d/${group}" >/dev/null 2>&1
#      echo 'Defaults env_delete = "PS1"' | sudo tee -a "/etc/sudoers.d/${group}" >/dev/null 2>&1
#      echo "Defaults: %${group} !logfile, !syslog" | sudo tee -a "/etc/sudoers.d/${group}" >/dev/null 2>&1
    fi
  done
}

function kali() {
  true
}


unset starting
