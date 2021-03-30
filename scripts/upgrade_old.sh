#!/usr/bin/env bash
# ${1} - name
# shellcheck disable=SC2034
export starting="${BASH_SOURCE[0]}"; debug.sh starting

export VIRTUAL_ENV PYTHONHOME; debug.sh VIRTUAL_ENV PYTHONHOME; unset VIRTUAL_ENV PYTHONHOME
source deactivate > /dev/null 2>&1
export VIRTUAL_ENV PYTHONHOME; debug.sh VIRTUAL_ENV PYTHONHOME; unset VIRTUAL_ENV PYTHONHOME

if [[ "${1-}" ]]; then
  while (( "$#" )); do
    case "${1}" in
      bapy) name="${1}"; url="${name}" ;;
      pen) name="${1}"; url="${PEN_GIT}" ;;
      *) name="bapy"; url="${name}" ;;
    esac; shift
  done
else
  name="bapy"; url="${name}"
fi

export BAPY PEN name url; debug.sh BAPY PEN name url

if [[ "${KALI-}" ]]; then
  unset SUDO
fi
export KALI SUDO; debug.sh KALI SUDO

if [[ "${DARWIN-}" ]]; then
  prefix="--prefix $( brew --prefix )"
fi
export DARWIN prefix; debug.sh DARWIN prefix

function install() {
  local error opt pr previous new
  case "${1}" in
    i) opt="install --upgrade ${url}"; pr=install ;;
    u) opt="uninstall ${name} -y"; pr=uninstall ;;
  esac

  export opt pr; debug.sh opt pr

  previous="$( ${name} v )"
  # shellcheck disable=SC2086
  if error="$( ${SUDO} ${PYTHON39} ${opt} -y 2>&1 )"; then
    new="$( ${name} v )"
    export previous new; debug.sh previous new
    if [[ "${previous}" == "${new}" ]]; then
      warning.sh "${pr}" "${name}" "${previous} ${new}"
    else
      info.sh "${pr}" "${name}" "${previous} ${new}"
    fi
  else
    error.sh "${pr}" "${name} ${previous} ${new}" "${error}"; return 2
  fi
}

if ! error=$( install i ); then
  if  [[ "${error}" == "1" ]]; then
    if error=$( install u ); then
      if ! error=$( install i ); then
         error.sh install "${name}"; exit 1
      fi
    else
      error.sh uninstall "${name}"; exit 1
    fi
  else
    error.sh install "${name}"; exit 1
  fi
fi


unset BAPY PEN starting error command url name error project_path DARWIN prefix previous new pr opt KALI SUDO
