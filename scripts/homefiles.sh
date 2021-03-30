#!/usr/bin/env bash
## source
# shellcheck disable=SC2034
export starting="${BASH_SOURCE[0]}"; debug.sh starting

test -n "${BASHRC_FILE}" || { error.bash BASHRC_FILE 'not defined'; return 1; }

function home_bashrc() {
  local user home file bashrc_path
  if bashrc_path="$( command -v "${BASHRC_FILE}" 2>&1 )"; then
#    echo '# shellcheck disable=SC1090' | sudo -u "${1}" tee "${2}" >/dev/null
    # shellcheck disable=SC2016
#    echo 'test -n "${PS1}" || return' | sudo -u "${1}" tee -a "${2}" >/dev/null
#    if sudo -u "${1}" tee -a "${2}" >/dev/null <<EOT; then
    if sudo -u "${1}" tee "${2}" >/dev/null <<EOT; then
# shellcheck disable=SC1090
if test -f  "${bashrc_path}"; then
  source "${bashrc_path}"
else
  echo 'bashrc file not found'; return 1
fi
EOT
      info.sh homefiles "${2}" "${bashrc_path}"
    else
      error.sh homefiles "${2}"; return 1
    fi
  else
    error.sh files "${2}" "${BASHRC_FILE} - command not found"; return 1
  fi
}

function home_hushlogin () {
  if sudo -u "${1}" touch "${2}"; then
    info.sh homefiles "${2}"
  else
    error.sh homefiles "${2}"; return 1
  fi
}

function home_inputrc () {
  if sudo -u "${1}" cp -f "$( command -v inputrc 2>&1 )" "${2}"; then
    info.sh homefiles "${2}"
  else
    error.sh homefiles "${2}"; return 1
  fi
}

function home_profiles () {
  if sudo -u "${1}" tee "${2}" >/dev/null <<EOT; then
# shellcheck disable=SC1090
PATH='/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin:/sbin:.'
if [[ -f ~/.bashrc ]]; then
  . ~/.bashrc
fi
EOT
    info.sh homefiles "${2}"
  else
    error.sh homefiles "${file}"; return 1
  fi
}

function home_file() {
  local user home file bashrc_path
  for user in "${USERNAME}" root kali; do
    if home="$( home.sh "${user}" )"; then
      file="${home}/.bashrc"
      home_bashrc "${user}" "${file}" || return 1
      file="${home}/.hushlogin"
      home_hushlogin "${user}" "${file}" || return 1
      file="${home}/.inputrc"
      home_inputrc "${user}" "${file}" || return 1
      file="${home}/.profile"
      home_profiles "${user}" "${file}" || return 1
      file="${home}/.bash_profile"
      home_profiles "${user}" "${file}" || return 1
    fi
  done
}

function home_secrets() {
  local error
  if ! test -d "${GITHUB_SECRETS_PATH}"; then
    if error="$( git clone "${GITHUB_SECRETS_URL}" "${GITHUB_SECRETS_PATH}" --quiet  2>&1 )"; then
      info.sh clone "${GITHUB_SECRETS_URL}" "empty dir"
    else
      error.sh clone "${GITHUB_SECRETS_URL}" "${error}"; return 1
    fi
  else
    cd "${GITHUB_SECRETS_PATH}"  || { error.sh "${GITHUB_SECRETS_PATH}" "invalid"; return 1; }
    if ! git log > /dev/null 2>&1; then
      if git clone "${GITHUB_SECRETS_URL}" /tmp/"$( basename "${GITHUB_SECRETS_PATH}" )" --quiet; then
        if rsync -aq /tmp/"$( basename "${GITHUB_SECRETS_PATH}" )" "$( dirname "${GITHUB_SECRETS_PATH}" )"; then
          if gpull.sh; then
          sudo rm -rf /tmp/"$( basename "${GITHUB_SECRETS_PATH}" )"/
            info.sh clone "${GITHUB_SECRETS_URL}" "not empty dir"
          else
            error.sh clone "not valid git after rsync"; return 1
          fi
        else
          sudo rm -rf /tmp/"$( basename "${GITHUB_SECRETS_PATH}" )"/
          error.sh clone "rsync clone to tmp"; return 1
        fi
      else
        error.sh clone "${GITHUB_SECRETS_URL}" /tmp/"$( basename "${GITHUB_SECRETS_PATH}" )"; return 1
      fi
    fi
    cd - > /dev/null || return 1

  fi
}

function home_links() {
  local error user home file
  cd "${USERHOME}" > /dev/null 2>&1 || { error.sh "${USERHOME}" "invalid"; return 1; }
  for user in root kali; do
    if home="$( home.sh "${user}" )"; then
      sudo mkdir -p "${home}/.ssh"
      sudo chmod go-rw "${home}/.ssh"
      sudo touch "${home}/.gitconfig"
      for file in .ssh/config .ssh/gitcredentials .gitconfig \
                  $( find .ssh -type f -exec grep -l "END OPENSSH PRIVATE KEY" "{}" \; ) .gitconfig; do
        sudo cp -rf "${USERHOME}/${file}" "${home}/${file}" > /dev/null 2>&1
        sudo chmod -R go-rw "${home}/.ssh"
        sudo chown -R "${user}":"$( id -g "${user}" )" "${home}/.ssh"
      done
      [[ "${user}" == 'root' ]] || sudo chown -R "${user}" "${home}/.ssh"
      [[ "${user}" == 'root' ]] || sudo chown -R "${user}" "${home}/.gitconfig"
    fi
    done
  cd - > /dev/null || return 1
  if isuserdarwin.sh; then
    mkdir -p "${USERHOME}/Library/Mobile Documents/com~apple~CloudDocs"
    if test -d "${USERHOME}/Library/Mobile Documents/com~apple~CloudDocs" && ! test -d "${ICLOUD}" \
                && ! test -L "${ICLOUD}"; then
      ln -s "${USERHOME}/Library/Mobile Documents/com~apple~CloudDocs" "${ICLOUD}"
    fi

    mkdir -p "${PYCHARM}/scratches/"
    ! test -e "${SCRATCHES}" || rm "${SCRATCHES}"
    if test -d "${PYCHARM}/scratches/" && ! test -d "${SCRATCHES}" && ! test -L "${SCRATCHES}"; then
      ln -s "${PYCHARM}/scratches/" "${SCRATCHES}"
    fi
  fi
}

if isuser.sh; then
  home_secrets || exit 1
  home_file || exit 1
  home_links || exit 1
fi

unset starting bashrc_path
