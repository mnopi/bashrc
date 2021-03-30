#!/usr/bin/env bash
export starting="${BASH_SOURCE[0]}"; debug.sh starting

# "${1}" - to force
file="/etc/apt/apt.conf.d/99local"
if ! test -f "${file}" || test -n "${1}"; then
  if sudo tee "${file}" >/dev/null <<EOT; then
APT::Periodic::Enable "0";
APT::Periodic::Update-Package-Lists "0";
APT::Periodic::Unattended-Upgrade "0";
Dpkg::Options {
   "--force-confdef";
   "--force-confold";
}
EOT
    info.sh dpkg-local "${file}"
  else
    error.sh dpkg-local "${file}"; exit 1
  fi
fi
