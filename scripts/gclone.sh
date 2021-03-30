#!/usr/bin/env bash
test -n "${DARWIN}" || exit

function gclone() {
  local action add bitbucket command error execute file gitlab line PARENT parent pagure project_path prefix pwd repo submodules suffix url user
  bitbucket="https://bitbucket.org/"
  gitlab="https://gitlab.com/"
  pagure="https://pagure.io/"
  prefix="${!1:-${GIT_PREFIX}}"

  if [[ "${1-}" ]]; then
    prefix="${!1:-${GIT_PREFIX}}"
    case "${1}" in
      add|bitbucket|github|gitlab) user="${2}"; repo="${3}"; [[ "${1}" != 'bitbucket' ]] || command='hg'; [[ "${1}" != 'add' ]] || { add="yes"; parent="${CLONES}"; } ;;
      pagure) repo="${2}" ;;
      *) case "${1}" in
          j) user="${GITHUB_USERNAME}" ;;
          l) user="${GITHUB_ORGANIZATION}"; prefix="${GITHUB_ORGANIZATION_ID_CLONE_PREFIX_SSH}" ;;
          *) user="${1}" ;;
        esac; repo="${2}"
    esac; user="${user:+${user}/}"; command=${command:-git}

    while (( "$#" )); do
      case "${1}" in
        add|bitbucket|github|gitlab|pagure|"${user//\/}"|"${repo}") true ;;
        __1|__2|__3|__4|__5) suffix="${1}" ;;
        submodules) submodules="--recurse-submodules" ;;
        *)          parent="${1}"; mkdir -p "${parent}" ;;
      esac; shift
    done
    parent="${parent:-.}"
    url="${prefix}${user}${repo}.git";
    project_path="${parent}/${repo}${suffix}"

    if test -d "${project_path}" && grep "^${repo}$" "${REPOS}/HOME/"*.repos > /dev/null 2>&1; then
      #  For project else pull for clones, mnopi and lumen
      cd "${project_path}" || true
      action="gpush"
      execute="${action} clone_function"
    elif test -d "${project_path}"; then
      cd "${project_path}" || true
      action="gpull"
      execute="${action}"
    else
      action="gclone"
      execute="${command} clone ${submodules} ${url} ${project_path} --quiet"
    fi
    pwd="$( pwd )"
    debug action add command desktop execute parent project_path prefix pwd repo submodules url user
    if error="$( ${execute} 2>&1 )"; then
      info "${action}" "${repo##*/}"
      echo -n
      if command -v github > /dev/null 2>&1; then
        github open "${project_path}"
        pkill "${GITHUB_DESKTOP_PROCESS}"
      fi
      if [[ "${add-}" ]] && [[ "${action}" == "gclone" ]]; then
        file="$( find "${REPOS}" -type f -name "*github*" )"
        echo "${user//\/} ${repo}" >> "${file}"
        if error=$( sortfile "${file}" 2>&1 ); then
          info 'gclone add' "${user//\/}/${repo}"
        else
          error sortfile "${file}" "${error}"; export GIT_RC=1; return "${GIT_RC}"
        fi
      fi
      return 0
    else
      error "${action}" "${repo##*/}" "${error}"; export GIT_RC=1; return "${GIT_RC}"
    fi
  else
    if error=$( sortdir "${REPOS}" 2>&1 ); then
      while read -r PARENT; do
        parent="${!PARENT}"
        debug PARENT parent
        while read -r line; do
          debug line
          # shellcheck disable=SC2086
          gclone ${line} "${!PARENT}"
        done < <( grep -REv '^#|^$' "${REPOS}/${PARENT}" | sed 's/#//g' | grep -v "^ " | grep -v "^$" | sed "s|${REPOS}/${PARENT}/||g" | sed 's/.repos:/ /g' )
        if [[ "${PARENT}" =~ ^CLONES$|^LUMEN ]]; then
          gclonesrm "${!PARENT}" "${REPOS}/${PARENT}" "${suffix}"
        fi
      done < <( find "${REPOS}" -type d -mindepth 1 -maxdepth 1 -exec basename "{}" \; )
    else
      error sortdir "${REPOS}" "${error}"; return 1
    fi
  fi
}; export -f gclone
