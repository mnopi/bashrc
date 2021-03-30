#!/usr/bin/env bash
export starting="${BASH_SOURCE[0]}"; debug.sh starting

# "${1}" - to force
file="/etc/logrotate.d/mongodb"
if ! test -f "${file}" || test -n "${1}"; then
  sudo tee "${file}" >/dev/null <<EOT
/var/log/mongodb/*.log {
  daily
  rotate 5
  size 64M
  compress
  dateext
  missingok
  notifempty
  create 644 mongodb mongodb
  sharedscripts
  postrotate
        sudo service mongod restart
  endscript
}
EOT
    sudo service mongod restart
    info.sh logrotate "${file}"
fi
