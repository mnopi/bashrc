#!/usr/bin/env bash
user="${1:-${USER}}"
if [[ "${user}" == "root" ]]; then
  home="/var/${user}"
  test -n "${DARWIN}" || home="/${user}"
else
  home="/Users/${user}"
  test -n "${DARWIN}" || home="/home/${user}"
fi
if ! test -d "${home}"; then
  exit 1
fi
echo "${home}"
