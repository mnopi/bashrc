#!/usr/bin/env bash
# shellcheck disable=SC1090
export starting="${BASH_SOURCE[0]}"; debug.sh starting

for script_name in bashrc-bootstrap-sudoers.sh bashrc-bootstrap-essentials.sh \
                   bashrc-bootstrap-home.sh bashrc-bootstrap-secrets.sh \
                   bashrc-bootstrap-links.sh; do
  script_path="$( command -v "${script_name}" )"; export script_path
  if test -f "${script_path}"; then
    debug.sh script_path
    "${script_path}"
  else
    error.sh "${script_name}" "not found"; exit 1
  fi
done

unset starting script_name script_path
