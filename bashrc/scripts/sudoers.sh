#!/usr/bin/env bash
export starting="${BASH_SOURCE[0]}"; debug.sh starting
# shellcheck disable=SC2153
test -n "${PASSWORD}" || { error.sh PASSWORD 'not defined'; exit 1; }

debug.sh DARWIN KALI DEBIAN UBUNTU PASSWORD

function darwin() {
  # "${1}" - force
  # "${1}" - password
  local group file
  while (( "$#" )); do
    case "${1}" in
      force) force="${1}" ;;
      *) password="${1}";;
    esac; shift
  done

  password="${password:-${PASSWORD}}"
  echo "${password}" | sudo -S true  > /dev/null 2>&1 || { error.sh sudoers "${password}" 'sudo -S'; return 1; }

  for group in staff admin wheel; do
    ## TODO: Test
    file="/etc/sudoers.d/${group}"
    file="/tmp/${group}"
    if ! test -f "${file}" || test -n "${force}"; then
      if sudo tee "${file}" >/dev/null <<EOT; then
Defaults env_reset
Defaults !requiretty
%${group} ALL=(ALL) NOPASSWD:ALL
Defaults: %${group} !logfile, !syslog
EOT
        if /usr/sbin/visudo -cf "${file}" > /dev/null ; then
          info.sh sudoers "${file}"
        else
          error.sh sudoers 'visudo -cf' "${file}"; return 1
        fi
      else
        error.sh sudoers tee "${file}"; return 1
      fi
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
