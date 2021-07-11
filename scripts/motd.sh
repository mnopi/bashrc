#!/usr/bin/env bash
export starting="${BASH_SOURCE[0]}"; debug.sh starting
[[ ! "${TIMES-}" ]] || start="$( date +%s )"

# "${1}" - to force
file="/etc/default/motd-news"
if ! test -f "${file}" || test -n "${1}"; then
  if sudo tee "${file}" >/dev/null <<EOT; then
ENABLED=0
EOT
    info.sh motd "${file}"
  else
    error.sh motd "${file}"; exit 1
  fi
fi
[[ ! "${TIMES-}" ]] || { end="$( date +%s )"; export runtime=$((end-start)); times.sh starting runtime; }
unset start end runtime
unset starting
