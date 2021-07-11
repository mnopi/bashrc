#!/usr/bin/env bash
export starting="${BASH_SOURCE[0]}"; debug.sh starting
[[ ! "${TIMES-}" ]] || start="$( date +%s )"

test -n "${GIT_STORE}" || { error.sh GIT_STORE 'not defined'; exit 1; }

[[ ${1-} ]] && username="${1}" || username="${GITHUB_WORK}"

case "${username}" in
  "${GITHUB_USER}")
    email="26859654+${username}@users.noreply.github.com"
    ;;
  "${GITHUB_WORK}")
    email="${GITHUB_ORG_EMAIL}"
    ;;
  *)
    error "username must be:" "${GITHUB_USER} or ${GITHUB_WORK}"
    exit 1
    ;;
esac

git config --global user.name "${REALNAME}" || exit 1
git config --global user.username "${username}" || exit 1
git config --global github.username "${username}" || exit 1
git config --global user.email "${email}" || exit 1
if [[ "$(uname -s)" == "Darwin" ]]; then
  git config --global credential.helper osxkeychain
else
  git config --global credential.helper "store --file ${GIT_STORE}" || exit 1
fi
git config --global core.excludesfile "${GITHUB_SECRETS_PATH}/gitignore" || exit 1
git config --global core.editor vi || exit 1
git config --global color.ui true || exit 1
git config --global receive.fsckObjects true || exit 1
git config --global receive.denyNonFastForwards true || exit 1
git config --global core.autocrlf input || exit 1
git config --global bash-it.hide-status 1 || exit 1

git config --global filter.lfs.clean "git-lfs clean -- %f" || exit 1
git config --global filter.lfs.smudge "git-lfs smudge -- %f" || exit 1
git config --global filter.lfs.process "git-lfs filter-process" || exit 1
git config --global filter.lfs.required "true" || exit 1

if [[ "$(uname -s)" != "Darwin" ]]; then
  git config --global \
  "credential.https://${REPO_DEFAULT_HOSTNAME}.${SERVERS_HOST}.helper" "'store --file ${GIT_STORE}'" || exit 1
fi

git config --global http.postBuffer 15728640 || exit 1

git config --global lfs.batch false || exit 1
git config --global \
  "lfs.https://${REPO_DEFAULT_HOSTNAME}.${SERVERS_HOST}/repository/gitlfs-internal/info/lfs.locksverify" true \
  || exit 1
git config --global \
  "lfs.https://${REPO_DEFAULT_HOSTNAME}.${SERVERS_HOST}/repository/gitlfs-internal/.locksverify" true || exit 1
git config --global core.sshCommand /usr/bin/ssh || exit 1
# https://www.lutro.me/posts/different-ssh-keys-per-github-organisation || exit 1
# git config --global url.git@example.github.com:example.insteadOf = git@github.com:example || exit 1
# git config --global url.git@example.github.com:example.insteadOf = https://github.com/example || exit 1

info.sh gconfig
[[ ! "${TIMES-}" ]] || { end="$( date +%s )"; export runtime=$((end-start)); times.sh starting runtime; }
unset start end runtime starting

if secrets.sh; then exit 0; else exit 1; fi
