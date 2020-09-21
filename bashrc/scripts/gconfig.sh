#!/usr/bin/env bash
export starting="${BASH_SOURCE[0]}"; debug.sh starting

test -n "${GIT_STORE}" || { error.sh GIT_STORE 'not defined'; exit 1; }

[[ ${1-} ]] && username="${1}" || username="${NFERX_GITHUB_USERNAME}"

case "${username}" in
  "${GITHUB_USERNAME}")
    email="26859654+${username}@users.noreply.github.com"
    ;;
  "${NFERX_GITHUB_USERNAME}")
    email="${NFERX_EMAIL}"
    ;;
  *)
    error "username must be:" "${GITHUB_USERNAME} or ${NFERX_GITHUB_USERNAME}"
    exit 1
    ;;
esac

git config --global user.name "${REALNAME}" || exit 1
git config --global user.username "${username}" || exit 1
git config --global github.username "${username}" || exit 1
git config --global user.email "${email}" || exit 1
git config --global credential.helper "store --file ${GIT_STORE}" || exit 1
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

git config --global credential.https://repo.nferx.com.helper "'store --file ${GIT_STORE}'" || exit 1

git config --global http.postBuffer 15728640 || exit 1

git config --global lfs.batch false || exit 1
git config --global lfs.https://repo.nferx.com/repository/gitlfs-internal/info/lfs.locksverify true || exit 1
git config --global lfs.https://repo.nferx.com/repository/gitlfs-internal/.locksverify true || exit 1
git config --global core.sshCommand /usr/bin/ssh || exit 1
# https://www.lutro.me/posts/different-ssh-keys-per-github-organisation || exit 1
# git config --global url.git@example.github.com:example.insteadOf = git@github.com:example || exit 1
# git config --global url.git@example.github.com:example.insteadOf = https://github.com/example || exit 1

info.sh gconfig

if secrets-push.sh; then exit 0; else exit 1; fi
