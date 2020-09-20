#!/usr/bin/env bash
## source
# shellcheck disable=SC2034
export starting="${BASH_SOURCE[0]}"; debug.sh starting

if [[ "${USER}" == "${USERNAME}" ]]; then
  cd "${USERHOME}" > /dev/null 2>&1 || { error.sh "${USERHOME}" "invalid"; return 1; }
  for user in root kali; do
    if home="$( home.sh "${user}" )"; then
      sudo -u "${user}" mkdir -p "${home}/.ssh"
      sudo -u "${user}" chmod go-rw "${home}/.ssh"
      for file in .ssh/config $( grep -l "END OPENSSH PRIVATE KEY" .ssh/**/* ) .gitconfig; do
        ! test -e || sudo -u "${user}" ln -s "${file}" "${home}/${file}"
      done
    fi
  done
  cd - > /dev/null || return 1
fi

unset starting user home file