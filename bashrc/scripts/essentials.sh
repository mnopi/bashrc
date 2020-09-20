#!/usr/bin/env bash
export starting="${BASH_SOURCE[0]}"; debug.sh starting
# shellcheck disable=SC2153
test -n "${PASSWORD}" || { error.sh PASSWORD 'not defined'; exit 1; }

debug.sh DARWIN KALI DEBIAN UBUNTU PASSWORD

function darwin() {
  # "${1}" - to force
  install-brew.sh || return 1
}

function kali() {
  # "${1}" - to force
  dpkg-local.sh "$@" || return 1
  apt-kali.sh || return 1
  pam-sudo.sh "$@" || return 1
  motd.sh "$@" || return 1
  rm -rf /etc/update-motd.d/
  sudo sed -i 's/^.*" //' /root/.ssh/authorized_keys > /dev/null 2>&1
  sudo sed -i 's/^.*" //' /home/kali/authorized_keys > /dev/null 2>&1
  sudo passwd -d root > /dev/null 2>&1
  sudo passwd -d kali > /dev/null 2>&1
#  sudo sed -i 's/^session.*optional.*pam_motd.so.*/# MOTD DISABLED/' /etc/pam.d/login
  sudo sed -i 's/^session.*optional.*pam_motd.so.*/# MOTD DISABLED/' /etc/pam.d/sshd
  touch /var/lib/cloud/instance/locale-check.skip
  sudo mkdir -p /etc/pki/tls/certs/
  sudo cp -f /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt
  install-brew.sh || return 1
}

! test -n "${DARWIN}" || darwin "$@" || exit 1
! test -n "${KALI}" || kali "$@" || exit 1

unset starting force password group file user
