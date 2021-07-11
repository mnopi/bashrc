#!/usr/bin/env bash
export starting="${BASH_SOURCE[0]}"; debug.sh starting
[[ ! "${TIMES-}" ]] || start="$( date +%s )"

# "${1}" - to force
file="/etc/apt/apt.conf.d/99local"
if ! test -f "${file}" || test -n "${1}"; then
  if sudo tee "${file}" >/dev/null <<EOT
APT::Periodic::Enable "0";
APT::Periodic::Update-Package-Lists "0";
APT::Periodic::Unattended-Upgrade "0";
Dpkg::Options {
   "--force-confdef";
   "--force-confold";
}
EOT
  then
    info.sh dpkg-local "${file}"
  fi
fi
[[ ! "${TIMES-}" ]] || { end="$( date +%s )"; export runtime=$((end-start)); times.sh starting runtime; }
unset start end runtime starting
