#!/usr/bin/env bash
## source
# shellcheck disable=SC2034
export starting="${BASH_SOURCE[0]}"; debug.sh starting

test -n "${BASHRC_FILE}" || { error.bash BASHRC_FILE 'not defined'; return 1; }

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

unset starting bashrc_path user home file
