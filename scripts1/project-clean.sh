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

sudo /bin/rm -fr build/ > /dev/null 2>&1
sudo /bin/rm -fr dist/ > /dev/null 2>&1
sudo /bin/rm -fr .eggs/ > /dev/null 2>&1
sudo find . -not \( -path "*/venv/*" -prune \) -name '*.egg-info' -exec /bin/rm -fr {} + > /dev/null 2>&1
sudo find . -not \( -path "*/venv/*" -prune \) -name '*.egg' -exec /bin/rm -f {} + > /dev/null 2>&1
sudo find . -not \( -path "*/venv/*" -prune \) -name '*.pyc' -exec /bin/rm -f {} + > /dev/null 2>&1
sudo find . -not \( -path "*/venv/*" -prune \) -name '*.pyo' -exec /bin/rm -f {} + > /dev/null 2>&1
sudo find . -not \( -path "*/venv/*" -prune \) -name '*~' -exec /bin/rm -f {} + > /dev/null 2>&1
sudo find . -not \( -path "*/venv/*" -prune \) -name '__pycache__' -exec /bin/rm -fr {} + > /dev/null 2>&1
sudo /bin/rm -fr .tox/ > /dev/null 2>&1
sudo /bin/rm -fr .pytest_cache > /dev/null 2>&1
sudo find . -not \( -path "*/venv/*" -prune \) -name '.mypy_cache' -exec /bin/rm -rf {} + > /dev/null 2>&1

info.sh clean "${name}"
cd - > /dev/null || exit 1

unset starting path name
