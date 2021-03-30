#!/usr/bin/env bash

file=/etc/security/limits.conf

if sudo tee "${file}" >/dev/null <<EOT; then
* soft     nproc          131070
* hard     nproc          131070
* soft     nofile         131070
* hard     nofile         131070
root soft     nproc          131070
root hard     nproc          131070
root soft     nofile         131070
root hard     nofile         131070
* soft core 0
* hard core 0
session required pam_limits.so
EOT
  info.sh ulimits "${file}"
else
  error.sh ulimits "${file}"
  exit 1
fi
