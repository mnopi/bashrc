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
  ulimits.sh || return 1
  wget -qO - https://www.mongodb.org/static/pgp/server-4.4.asc | sudo apt-key add -
  echo "deb http://repo.mongodb.org/apt/debian buster/mongodb-org/4.4 main" | sudo tee /etc/apt/sources.list.d/mongodb-org-4.4.list
  sudo apt update
  echo "mongodb-org hold" | sudo dpkg --set-selections
  echo "mongodb-org-server hold" | sudo dpkg --set-selections
  echo "mongodb-org-shell hold" | sudo dpkg --set-selections
  echo "mongodb-org-mongos hold" | sudo dpkg --set-selections
  echo "mongodb-org-tools hold" | sudo dpkg --set-selections
  sudo systemctl start mongod
  sudo systemctl daemon-reload
  sudo systemctl enable mongod
  sudo apt-get install -y mongodb-org
  sudo mkdir -p "${PEN}"; sudo chown -R "${USERNAME}":"${USERNAME}" "${PEN}"
  sudo rm -rf /etc/update-motd.d/
  sudo sed -i 's/#Banner none/Banner none/' /etc/ssh/sshd_config
  sudo sed -i 's/#LogLevel INFO/LogLevel QUIET/' /etc/ssh/sshd_config
  sudo service ssh restart || return 1
  sudo sed -i 's/^.*" //' /root/.ssh/authorized_keys > /dev/null 2>&1
  sudo sed -i 's/^.*" //' /home/kali/authorized_keys > /dev/null 2>&1
  sudo passwd -d root > /dev/null 2>&1
  sudo passwd -d kali > /dev/null 2>&1
  sudo sed -i 's/^session.*optional.*pam_motd.so.*/# MOTD DISABLED/' /etc/pam.d/login > /dev/null 2>&1
  sudo sed -i 's/^session.*optional.*pam_motd.so.*/# MOTD DISABLED/' /etc/pam.d/sshd > /dev/null 2>&1
  sudo mkdir -p /var/lib/cloud/instance/
  sudo touch /var/lib/cloud/instance/locale-check.skip > /dev/null 2>&1
  sudo mkdir -p /etc/pki/tls/certs/
  sudo cp -f /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt > /dev/null 2>&1
  install-brew.sh || return 1
}

! test -n "${DARWIN}" || darwin "$@" || exit 1
! test -n "${KALI}" || kali "$@" || exit 1

unset starting force password group file user
