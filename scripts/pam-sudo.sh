#!/usr/bin/env bash
export starting="${BASH_SOURCE[0]}"; debug.sh starting
[[ ! "${TIMES-}" ]] || start="$( date +%s )"

# "${1}" - to force
file="/etc/pam.d/sudo"
if ! test -f "${file}" || test -n "${1}"; then
  if sudo tee "${file}" >/dev/null <<EOT
#%PAM-1.0

@include common-auth
@include common-account
session [success=1 default=ignore] pam_succeed_if.so quiet uid = 0 ruser = "${USERNAME}"
@include common-session-noninteractive
EOT
  then
    sudo chown 644 "${file}"
    info.sh pam-sudo "${file}"
  else
    exit 1
  fi
fi
[[ ! "${TIMES-}" ]] || { end="$( date +%s )"; export runtime=$((end-start)); times.sh starting runtime; }
unset start end runtime
unset starting
