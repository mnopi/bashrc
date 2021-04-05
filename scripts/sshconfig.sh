#!/usr/bin/env bash
export starting="${BASH_SOURCE[0]}"; debug.sh starting

function authorized_keys() {
  local user home file key tmp
  local starting
  export starting="${FUNCNAME[0]}"; debug.sh starting
  for user in "${USERNAME}" root kali; do
    if home="$( home.sh "${user}" )"; then
      file="${home}/.ssh/authorized_keys"
      sudo -u "${user}" mkdir -p "${home}/.ssh"
      sudo -u "${user}" chmod go-rw "${home}/.ssh"
      sudo -u "${user}" touch "${file}"
      for key in "${GITHUB_USERNAME}" "${GITHUB_ORGANIZATION_USERNAME}"; do
        if tmp="$( curl -sL https://github.com/"${key}".keys )"; then
          if ! grep "${tmp}" "${file}" > /dev/null 2>&1; then
            if echo "${tmp}" | sudo -u "${user}" tee -a "${file}" > /dev/null 2>&1; then
              info.sh authorized_keys "${key}" "${file}"
            else
              info.sh authorized_keys "${key}" "${file}"; return 1
            fi
          fi
        else
          error.sh authorized_keys "${key} ${file}" "${tmp}"; return 1
        fi
      done
    fi
  done
}

function ssh_config() {
  local file
  local starting
  export starting="${FUNCNAME[0]}"; debug.sh starting
  file="${USERHOME}/.ssh/config"
  sudo touch "${file}"
  sudo chown -R "${USERNAME}":"$( id -g "${USERNAME}")" "${file}"
  if tee "${file}" >/dev/null <<EOT
Host *
  AddressFamily inet
  BatchMode yes
  CheckHostIP no
  ControlMaster auto
  ControlPath /tmp/ssh-%h-%r-%p
  ControlPersist 20m
  IdentitiesOnly yes
  IdentityFile ~/.ssh/id_rsa
  LogLevel QUIET
  StrictHostKeyChecking no
  User ${USERNAME}
  UserKnownHostsFile /dev/null

Host github.com
  HostName github.com
  User ${GITHUB_ORGANIZATION_ID}

Host github.com.${GITHUB_USERNAME}
  HostName github.com
  User ${GITHUB_USERNAME}
  IdentityFile ~/.ssh/${GITHUB_USERNAME}
EOT
  then
    info.sh ssh "${file}"
    if secrets.sh; then return 0; else return 1; fi
  else
    return 1
  fi
}

function etc_hosts() {
  local file
  local starting
  export starting="${FUNCNAME[0]}"; debug.sh starting
  file="/etc/hosts"
  if sudo tee "${file}" >/dev/null <<EOT
127.0.0.1       localhost
255.255.255.255 broadcasthost
67.202.15.57 scan.com scan
67.202.15.57 kali.com kali
192.168.1.10 hp.com hp
EOT
  then
    info.sh ssh "${file}"
  else
    return 1
  fi
}

if [[ "${USER}" = "${USERNAME}" ]]  && isuserdarwin.sh; then
  etc_hosts || exit 1
  ssh_config || exit 1
fi

if [[ "${USER}" = "${USERNAME}" ]] ; then
  authorized_keys || exit 1
fi

grep -slR "PRIVATE" ~/.ssh | xargs ssh-add >/dev/null 2>&1

unset starting
