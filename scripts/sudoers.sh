#!/usr/bin/env bash
export starting="${BASH_SOURCE[0]}"; debug.sh starting
[[ ! "${TIMES-}" ]] || start="$( date +%s )"

# shellcheck disable=SC2153
test -n "${PASSWORD}" || { error.sh PASSWORD 'not defined'; exit 1; }

debug.sh DARWIN KALI DEBIAN UBUNTU PASSWORD

function darwin() {
  # "${1}" - force
  # "${1}" - password
  local force password group file
  local starting
  export starting="${FUNCNAME[0]}"; debug.sh starting
  while (( "$#" )); do
    case "${1}" in
      force) force="${1}" ;;
      *) password="${1}";;
    esac; shift
  done

  password="${password:-${PASSWORD}}"
  echo "${password}" | sudo -S true  > /dev/null 2>&1 || { error.sh sudoers "${password}" 'sudo -S'; return 1; }

  for group in staff admin wheel; do
    file="/etc/sudoers.d/${group}"
    if ! test -f "${file}" || test -n "${force}"; then
      if sudo tee "${file}" >/dev/null <<EOT
Defaults env_reset
Defaults !requiretty
%${group} ALL=(ALL) NOPASSWD:ALL
Defaults: %${group} !logfile, !syslog
EOT
      then
        if /usr/sbin/visudo -cf "${file}" > /dev/null ; then
          info.sh sudoers "${file}"
        else
          error.sh sudoers 'visudo -cf' "${file}"; return 1
        fi
      else
        return 1
      fi
    fi
  done
}

function kali() {
  # "${1}" - force
  # "${1}" - password
  local force password group file user
  local starting
  export starting="${FUNCNAME[0]}"; debug.sh starting
  while (( "$#" )); do
    case "${1}" in
      force) force="${1}" ;;
      *) password="${1}";;
    esac; shift
  done
  password="${password:-${PASSWORD}}"
  echo "${password}" | sudo -S true  > /dev/null 2>&1 || { error.sh sudoers "${password}" 'sudo -S'; return 1; }

  for group in sudo kali-trusted; do
    file="/etc/sudoers.d/${group}"
    if ! test -f "${file}" || test -n "${force}"; then
      if sudo tee "${file}" >/dev/null <<EOT
Defaults env_reset
Defaults !requiretty
%${group} ALL=(ALL) NOPASSWD:ALL
Defaults: %${group} !logfile, !syslog
EOT
      then
        if sudo /usr/sbin/visudo -cf "${file}" > /dev/null ; then
          info.sh sudoers "${file}"
        else
          error.sh sudoers 'visudo -cf' "${file}"; return 1
        fi
      else
        error.sh sudoers tee "${file}"; return 1
      fi
    fi
  done

  for user in "${USERNAME}" kali; do
    file="/etc/sudoers.d/${user}"
    if ! test -f "${file}" || test -n "${force}"; then
      if sudo tee "${file}" >/dev/null <<EOT
Defaults env_reset
Defaults !requiretty
${user} ALL=(ALL) NOPASSWD:ALL
Defaults: ${user} !logfile, !syslog
EOT
      then
        if sudo /usr/sbin/visudo -cf "${file}" > /dev/null ; then
          info.sh sudoers "${file}"
        else
          error.sh sudoers 'visudo -cf' "${file}"; return 1
        fi
      else
        return 1
      fi
    fi
  done
}

! test -n "${DARWIN}" || darwin "$@" || exit 1
! test -n "${KALI}" || kali "$@" || exit 1
[[ ! "${TIMES-}" ]] || { end="$( date +%s )"; export runtime=$((end-start)); times.sh starting runtime; }
unset start end runtime

unset starting force password group file user
