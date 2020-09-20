#!/usr/bin/env bash
export starting="${BASH_SOURCE[0]}"; debug.sh starting
# shellcheck disable=SC2153
test -n "${PASSWORD}" || { error.sh PASSWORD 'not defined'; exit 1; }

debug.sh DARWIN KALI DEBIAN UBUNTU PASSWORD
function darwin() {
  # "${1}" - force
  # "${1}" - password
  local group
  while (( "$#" )); do
    case "${1}" in
      force) force="${1}" ;;
      *) password="${1}";;
    esac; shift
  done
  password="${password:-${PASSWORD}}"
  for group in staff admin wheel; do
    if ! test -f "/etc/sudoers.d/${group}" || test -n "${force}"; then
      echo "${force}"
      echo "${password}"

#      echo "${password}" | sudo -S true
#      echo "%${group} ALL=(ALL) NOPASSWD:ALL" | sudo tee "/etc/sudoers.d/${group}" && info sudoers "${group}"
#      echo 'Defaults !env_reset' | sudo tee -a "/etc/sudoers.d/${group}" >/dev/null 2>&1
#      echo 'Defaults env_delete = "HOME"' | sudo tee -a "/etc/sudoers.d/${group}" >/dev/null 2>&1
#      echo 'Defaults env_delete = "PS1"' | sudo tee -a "/etc/sudoers.d/${group}" >/dev/null 2>&1
#      echo "Defaults: %${group} !logfile, !syslog" | sudo tee -a "/etc/sudoers.d/${group}" >/dev/null 2>&1
    fi
  done
}

function kali() {
  # "${1}" - force
  # "${1}" - password
  local group
  while (( "$#" )); do
    case "${1}" in
      force) force="${1}" ;;
      *) password="${1}";;
    esac; shift
  done
  password="${password:-${PASSWORD}}"
}

! test -n "${DARWIN}" || darwin "$@" || exit 1
! test -n "${KALI}" || kali "$@" || exit 1

unset starting
