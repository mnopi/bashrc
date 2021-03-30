#!/usr/bin/env bash
test -n "${DARWIN}" || exit

function gclonesrm() {
  local basename field index repo rm suffix
  suffix=${3}
  if test -d "${1}" && test -d "${2}"; then
      while read -r basename; do
        field='__'
        repo="$( echo "${basename}"  | awk -F "${field}" '{print $1}' )"
        index="$( echo "${basename}"  | awk -F "${field}" '{print $2}' )"
        suffix="${index:+${field}${index}}"
        if ! find "${2}" -type f -name "*.repos" -exec grep -v '#' "{}" \; | grep " ${repo}" | grep " ${suffix}"> /dev/null 2>&1; then
          # shellcheck disable=SC2034
          rm='true'
          for file in "${2}/pagure.repos" "${2}/lumenbiomics.repos"; do
            if test -f "${file}"; then
              if grep "^${repo}" "${file}" > /dev/null 2>&1; then
                 # shellcheck disable=SC2034
                rm=
              fi
            fi
          done
          if [[ "${rm-}" ]]; then
            debug basename repo rm suffix
            ! rm -r "${1:?}/${basename:?}" || warning gclonesrm "${basename}"
          fi
        fi
    done < <( find "${1}" -type d -mindepth 1 -maxdepth 1 -exec basename "{}" \; )
  fi
}; export -f gclonesrm

