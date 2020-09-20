#!/usr/bin/env bash
# shellcheck disable=SC2034
export starting="${BASH_SOURCE[0]}"; debug.sh starting

if ! isuser.sh; then
  error.sh "can not be done with root"; exit 1
fi

path="${1:-.}"
path="$( pwd )"
name="$( basename "${path}" )"
export path name; debug.sh path name

cd "${path}" > /dev/null 2>&1 || { error.sh "${path}" "invalid"; exit 1; }

/bin/rm -fr build/ > /dev/null
/bin/rm -fr dist/ > /dev/null
/bin/rm -fr .eggs/ > /dev/null
find . -name '*.egg-info' -exec /bin/rm -fr {} +
find . -name '*.egg' -exec /bin/rm -f {} +
find . -name '*.pyc' -exec /bin/rm -f {} +
find . -name '*.pyo' -exec /bin/rm -f {} +
find . -name '*~' -exec /bin/rm -f {} +
find . -name '__pycache__' -exec /bin/rm -fr {} +
/bin/rm -fr .tox/
/bin/rm -fr .pytest_cache
find . -name '.mypy_cache' -exec /bin/rm -rf {} +

info.sh clean "${name}"
cd - > /dev/null || exit 1

unset starting path name
