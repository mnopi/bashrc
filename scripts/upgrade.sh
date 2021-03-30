#!/usr/bin/env bash
# ${1} - name
# shellcheck disable=SC2034
export starting="${BASH_SOURCE[0]}"; debug.sh starting

export VIRTUAL_ENV PYTHONHOME; debug.sh VIRTUAL_ENV PYTHONHOME; unset VIRTUAL_ENV PYTHONHOME
source deactivate > /dev/null 2>&1
deactivate > /dev/null 2>&1
export VIRTUAL_ENV PYTHONHOME; debug.sh VIRTUAL_ENV PYTHONHOME; unset VIRTUAL_ENV PYTHONHOME

if [[ "${1-}" ]]; then
  while (( "$#" )); do
    case "${1}" in
      bapy) name="${1}"; url="${name}" ;;
      pen) name="${1}"; url="${PEN_GIT}" ;;
      once) once="${1}" ;;
      *) name="bapy"; url="${name}" ;;
    esac; shift
  done
else
  name="bapy"; url="${name}"
fi

export BAPY PEN name url once; debug.sh BAPY PEN name url once

if [[ "${KALI-}" ]]; then
#  sudo rm -rf /root/.cache/pip
  sudo rm -rf /root/.pip
  cmd="sudo /bin/python3.9"
fi
export KALI SUDO; debug.sh KALI SUDO

if [[ "${DARWIN-}" ]]; then
  sudo rm -rf "${USERHOME}"/.pip
#  rm -rf "${USERHOME}"/Library/Caches/pip
  cmd="/usr/local/bin/python3.9"
  prefix="--prefix $( brew --prefix )"
  unset SUDO
fi
export DARWIN prefix; debug.sh DARWIN prefix

previous="$( /usr/local/bin/${name} v )"

# shellcheck disable=SC2086
if ! ${cmd} -m pip -q install ${prefix} --upgrade pip wheel setuptools ${url}; then
  /usr/local/bin/error.sh "${name}" install "${previous}"; exit 1
fi

new="$( /usr/local/bin/${name} v )"
if [[ ! "${once-}" ]] && [[ "${previous}" == "${new}" ]]; then
  unset once
  ${BASH_SOURCE[0]} "${name}" once
  exit
fi

new="$( /usr/local/bin/${name} v )"

if [[ "${previous}" == "${new}" ]]; then
  /usr/local/bin/warning.sh "${name}" install "${previous} ${new}"
else
  /usr/local/bin/success.sh "${name}" install "${previous} ${new}"
fi

unset BAPY PEN starting error command url name error project_path DARWIN repeat prefix previous new pr opt once \
      KALI SUDO
