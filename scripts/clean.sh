#!/usr/bin/env bash
# shellcheck disable=SC2034
export starting="${BASH_SOURCE[0]}"; debug.sh starting
[[ ! "${TIMES-}" ]] || start="$( date +%s )"

if ! isuserdarwin.sh || [[ "${USERNAME}" != "${USER}" ]]; then
  error.sh "Can not be done with root and user should be: ${USERNAME}"; exit 1
fi

if [[ "${1-}" ]]; then
  while (( "$#" )); do
    case "${1}" in
      bapy) name="${1}"; project_path="${BAPY}" ;;
      pen) name="${1}"; project_path="${PEN}" ;;
      "${BASH_RC_NAME}")  name="${1}"; project_path="${BASH_RC_PROJECT}" ;;
      *) project_path="${BAPY}"; name="$( basename "${project_path}" )" ;;
    esac; shift
  done
else
  project_path="${BAPY}"; name="$( basename "${BAPY}" )"
fi

[[ "${project_path-}" ]] || { project_path="${BAPY}"; name="$( basename "${project_path}" )"; }
export BAPY PEN project_path name; debug.sh BAPY PEN project_path name

cd "${project_path}" > /dev/null 2>&1 || { error.sh "${project_path}" "invalid"; exit 1; }

sudo /bin/rm -fr build/ > /dev/null 2>&1
sudo /bin/rm -fr dist/ > /dev/null 2>&1
sudo /bin/rm -fr .eggs/ > /dev/null 2>&1
sudo find . -not \( -path "*/venv/*" -prune \) -name "*.egg-info" -exec /bin/rm -fr {} + > /dev/null 2>&1
sudo find . -not \( -path "*/venv/*" -prune \) -name "*.egg" -exec /bin/rm -f {} + > /dev/null 2>&1
sudo find . -not \( -path "*/venv/*" -prune \) -name "*.pyc" -exec /bin/rm -f {} + > /dev/null 2>&1
sudo find . -not \( -path "*/venv/*" -prune \) -name "*.pyo" -exec /bin/rm -f {} + > /dev/null 2>&1
sudo find . -not \( -path "*/venv/*" -prune \) -name "*~" -exec /bin/rm -f {} + > /dev/null 2>&1
sudo find . -not \( -path "*/venv/*" -prune \) -name "__pycache__" -exec /bin/rm -fr {} + > /dev/null 2>&1
sudo /bin/rm -fr .tox/ > /dev/null 2>&1
sudo /bin/rm -fr .pytest_cache > /dev/null 2>&1
sudo find . -not \( -path "*/venv/*" -prune \) -name ".mypy_cache" -exec /bin/rm -rf {} + > /dev/null 2>&1

info.sh "${name}" clean
cd - > /dev/null || exit 1
[[ ! "${TIMES-}" ]] || { end="$( date +%s )"; export runtime=$((end-start)); times.sh starting runtime; }
unset start end runtime

unset starting project_path name
