#!/usr/bin/env bash

file=/etc/security/limits.conf

if sudo tee "${file}" >/dev/null <<EOT; then
* soft     nproc          65535
* hard     nproc          65535
* soft     nofile         65535
* hard     nofile         65535
root soft     nproc          65535
root hard     nproc          65535
root soft     nofile         65535
root hard     nofile         65535
* soft core 0
* hard core 0
EOT
  info.sh ulimits "${file}"
else
  error.sh ulimits "${file}"
  exit 1
fi
