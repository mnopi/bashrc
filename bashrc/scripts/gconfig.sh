#!/usr/bin/env bash
export starting="${BASH_SOURCE[0]}"; debug.sh starting

test -n "${GIT_CONFIG}" || { error.sh GIT_CONFIG 'not defined'; return 1; }
test -n "${GIT_STORE}" || { error.sh GIT_STORE 'not defined'; return 1; }

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

git config user.name "${REALNAME}" || exit 1
git config user.username "${username}" || exit 1
git config github.username "${username}" || exit 1
git config user.email "${email}" || exit 1
git config credential.helper "'store --file ${GIT_STORE}'" || exit 1
git config core.excludesfile "${GITHUB_SECRETS_PATH}/gitignore" || exit 1
git config core.editor vi || exit 1
git config color.ui true || exit 1
git config receive.fsckObjects true || exit 1
git config receive.denyNonFastForwards true || exit 1
git config core.autocrlf input || exit 1
git config bash-it.hide-status 1 || exit 1

git config filter.lfs.clean "git-lfs clean -- %f" || exit 1
git config filter.lfs.smudge "git-lfs smudge -- %f" || exit 1
git config filter.lfs.process "git-lfs filter-process" || exit 1
git config filter.lfs.required "true" || exit 1

git config credential.https://repo.nferx.com.helper "'store --file ${GIT_STORE}'" || exit 1

git config http.postBuffer 15728640 || exit 1

git config lfs.batch false || exit 1
git config lfs.https://repo.nferx.com/repository/gitlfs-internal/info/lfs.locksverify true || exit 1
git config lfs.https://repo.nferx.com/repository/gitlfs-internal/.locksverify true || exit 1
git config core.sshCommand /usr/bin/ssh || exit 1
# https://www.lutro.me/posts/different-ssh-keys-per-github-organisation || exit 1
# git config url.git@example.github.com:example.insteadOf = git@github.com:example || exit 1
# git config url.git@example.github.com:example.insteadOf = https://github.com/example || exit 1

info.sh gconfig  "${GIT_CONFIG}"

if secrets-push.sh; then exit 0; else exit 1; fi
