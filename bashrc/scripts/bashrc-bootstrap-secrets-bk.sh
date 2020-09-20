#!/usr/bin/env bash
# source
# shellcheck disable=SC2034
export starting="${BASH_SOURCE[0]}"; debug.sh starting

test -n "${GITHUB_SECRETS_URL}" || { error.sh GITHUB_SECRETS_URL 'not defined'; return 1; }
test -n "${GITHUB_SECRETS_PATH}" || { error.sh GITHUB_SECRETS_PATH 'not defined'; return 1; }
test -n "${SECRETS_PATH}" || { error.sh SECRETS_PATH 'not defined'; return 1; }

if [[ "${USER}" == "${USERNAME}" ]]; then
  if ! test -d "${GITHUB_SECRETS_PATH}"; then
    if error="$( git clone "${GITHUB_SECRETS_URL}" "${GITHUB_SECRETS_PATH}" --quiet  2>&1 )"; then
      info.sh clone "${GITHUB_SECRETS_URL}"
    else
      error.sh clone "${GITHUB_SECRETS_URL}" "${error}"; return 1
    fi
  fi
  cd "${USERHOME}" > /dev/null 2>&1 || { error.sh "${USERHOME}" "invalid"; return 1; }
  for user in root kali; do
    if home="$( home.sh "${user}" )"; then
      sudo -u "${user}" mkdir -p "${home}/.ssh"
      sudo -u "${user}" chmod go-rw "${home}/.ssh"
      pwd
      for file in .ssh/config $( grep -l "END OPENSSH PRIVATE KEY" .ssh/**/* ) .gitconfig; do
        ! test -e "${home}/${file}" || sudo -u "${user}" ln -s "${file}" "${home}/${file}"
      done
    fi
  done
  cd - > /dev/null || return 1
fi

if test -f "${SECRETS_PATH}"; then
  # shellcheck disable=SC1090
  source "${SECRETS_PATH}";
  test -n "${NFERX_GITHUB_PASSWORD}" || { error.sh NFERX_GITHUB_PASSWORD 'not defined'; return 1; }
  debug.sh NFERX_GITHUB_PASSWORD
else
  error.sh "${SECRETS_PATH}" "not found"; return 1
fi

unset starting error
