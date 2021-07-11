#!/usr/bin/env bash
export starting="${BASH_SOURCE[0]}"; debug.sh starting
[[ ! "${TIMES-}" ]] || start="$( date +%s )"

# shellcheck disable=SC2153
test -n "${PASSWORD}" || { error.sh PASSWORD 'not defined'; exit 1; }

debug.sh DARWIN KALI DEBIAN UBUNTU PASSWORD

function darwin() {
  # "${1}" - to force
    local starting
    export starting="${FUNCNAME[0]}"; debug.sh starting

  install-brew.sh || return 1
  filelimits.sh || return 1
#  sudo launchctl limit maxfiles 65536 524288
  # https://superuser.com/questions/433746/is-there-a-fix-for-the-too-many-open-files-in-system-error-on-os-x-10-7-1
}

function kali() {
  # "${1}" - to force
  local starting
  export starting="${FUNCNAME[0]}"; debug.sh starting
  dpkg-local.sh "$@" || return 1
  apt-kali.sh || return 1
  pam-sudo.sh "$@" || return 1
  motd.sh "$@" || return 1
  ulimits.sh || return 1
  mongo.sh || return 1
  logrotate.sh || return 1
  sshtunnel.sh || return 1
  docker.sh || return 1
  sudo sysctl -w fs.inotify.max_user_watches=4096000
  sudo sysctl -w fs.file-max=1000000
  sudo sysctl -w kernel.msgmnb=65536
  sudo sysctl -w kernel.msgmax=65536
  sudo sysctl -w kernel.perf_event_paranoid=2
  sudo sysctl -w kernel.shmmax=68719476736
  sudo sysctl -w kernel.shmall=4294967296
  sudo sysctl -w net.core.wmem_max=12582912
  sudo sysctl -w net.core.rmem_max=12582912
  sudo sysctl -w kernel.core_uses_pid=1
  sudo sysctl -w kernel.kptr_restrict=1
  sudo sysctl -w kernel.randomize_va_space=1
  sudo sysctl -w kernel.sysrq=0
  sudo sysctl -w fs.protected_hardlinks=1
  sudo sysctl -w fs.protected_symlinks=1
  sudo sysctl -w fs.suid_dumpable=0
  sudo sysctl -w vm.swappiness=60
  sudo sysctl -w vm.overcommit_memory=1
  sudo sysctl -w net.core.somaxconn=65535
  sudo mkdir -p "${PEN}"; sudo chown -R "${USERNAME}":"${USERNAME}" "${PEN}"
  sudo rm -rf /etc/update-motd.d/
  sudo sed -i 's/#Banner none/Banner none/' /etc/ssh/sshd_config
  sudo sed -i 's/#LogLevel INFO/LogLevel QUIET/' /etc/ssh/sshd_config
  sudo service ssh restart || return 1
  sudo rm -rf /etc/apparmor.d/local
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
  cd /tmp || exit 1
  install-brew.sh || return 1
}

! test -n "${DARWIN}" || darwin "$@" || exit 1
! test -n "${KALI}" || kali "$@" || exit 1
[[ ! "${TIMES-}" ]] || { end="$( date +%s )"; export runtime=$((end-start)); times.sh starting runtime; }
unset start end runtime

unset starting force password group file user
