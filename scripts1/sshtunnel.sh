#!/usr/bin/env bash
exit

file=/etc/security/limits.conf

if tee "${file}" >/dev/null <<EOT; then
[Unit]
Description=tunnel
After=firewall.service

[Service]
User="${USERNAME}"
Type=simple
KillMode=process
ExecReload=/bin/kill -HUP $MAINPID
ExecStartPre=-/usr/bin/rm -rf /tmp/ssh-*
ExecStart=/usr/bin/ssh -N "${USERNAME}"@{{ item }}.{{ ssh_socks_tld }}
RestartSec=3s
Restart=on-failure
TasksAccounting=yes
TasksMax=1

[Install]
WantedBy=multi-user.target
EOT
  info.sh ulimits "${file}"
else
  error.sh ulimits "${file}"
  exit 1
fi
