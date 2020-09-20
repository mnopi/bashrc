#!/usr/bin/env bash
## source
# shellcheck disable=SC2034
export starting="${BASH_SOURCE[0]}"; debug.sh starting

test -n "${BASHRC_FILE}" || { error.bash BASHRC_FILE 'not defined'; return 1; }

function home_bashrc() {
  local user home file bashrc_path
  if [[ "${USER}" == "${USERNAME}" ]]; then
    for user in "${USERNAME}" root kali; do
      if home="$( home.sh "${user}" )"; then
        file="${home}/.bashrc"
        if ! sudo -u "${user}" grep source "${file}" | grep "${BASHRC_FILE}" > /dev/null 2>&1; then
          if bashrc_path="$( command -v "${BASHRC_FILE}" 2>&1 )"; then
            if sudo -u "${user}" tee -a "${file}" >/dev/null <<EOT; then

if test -f  "${bashrc_path}";then
  source "${bashrc_path}"
else
  echo 'bashrc file not found'; return 1
fi
EOT
              info.sh files "${file}" "${bashrc_path}"
            else
              error.sh files "${file}"
            fi
          else
            error.sh files "${file}" "${BASHRC_FILE} - command not found"
          fi
        fi
      fi
    done
  fi
}

function home_secrets() {
  local error user home file
  if [[ "${USER}" == "${USERNAME}" ]]; then
    if ! test -d "${GITHUB_SECRETS_PATH}"; then
      if error="$( git clone "${GITHUB_SECRETS_URL}" "${GITHUB_SECRETS_PATH}" --quiet  2>&1 )"; then
        info.sh clone "${GITHUB_SECRETS_URL}"
      else
        error.sh clone "${GITHUB_SECRETS_URL}" "${error}"; return 1
      fi
    fi
  fi
}

function home_links() {
  local error user home file
  cd "${USERHOME}" > /dev/null 2>&1 || { error.sh "${USERHOME}" "invalid"; return 1; }
  for user in root kali; do
    if home="$( home.sh "${user}" )"; then
      sudo -u "${user}" mkdir -p "${home}/.ssh"
      sudo -u "${user}" chmod go-rw "${home}/.ssh"
      touch "${home}/.gitconfig"
      for file in .ssh/config $( find .ssh -type f -exec grep -l "END OPENSSH PRIVATE KEY" "{}" \; ) .gitconfig; do
        if ! test -e "${home}/${file}"; then
          if sudo -u "${user}" ln -s "${file}" "${home}/${file}"; then
            info.sh link "${home}/${file}"
          else
            error.sh link "${home}/${file}"; return 1
          fi
        fi
      done
    fi
    done
  cd - > /dev/null || return 1
}

home_bashrc "$@" || exit 1
home_secrets "$@" || exit 1
home_links "$@" || exit 1
unset starting bashrc_path user home file
