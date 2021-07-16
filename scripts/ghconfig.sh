#!/usr/bin/env bash
export starting="${BASH_SOURCE[0]}"
debug.sh starting
[[ ! "${TIMES-}" ]] || start="$(date +%s)"

# shellcheck disable=SC2016
gh alias set loguser --shell 'gh auth login --with-token <<< $GITHUB_USER_TOKEN'
# shellcheck disable=SC2016
gh alias set logwork --shell 'gh auth login --with-token <<< $GITHUB_WORK_TOKEN'
# shellcheck disable=SC2016
gh alias set clone 'repo clone "$1" -- --quiet' # gh clonesilent j5pu/data

info.sh ghconfig
cd "${USERHOME}/git" || exit 1
gall.sh
if secrets.sh; then exit 0; else exit 1; fi

[[ ! "${TIMES-}" ]] || {
  end="$(date +%s)"
  export runtime=$((end - start))
  times.sh starting runtime
}
unset start end runtime starting
