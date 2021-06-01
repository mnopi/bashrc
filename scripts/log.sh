#!/usr/bin/env bash
case "${1}" in
  b) name="bapy" ;;
  d) name="daemon" ;;
  p) name="pen" ;;
esac

DIR="${HOME}/.${name}/log"
FILE="${DIR}/${name}.log"
rm "${FILE}"
touch "${FILE}"
if [[ "${name}" == "daemon" ]]; then
  sudo touch "${DIR}/scan.log"
  sudo chown root:adm "${DIR}/scan.log"
  sudo chmod +r "${DIR}/scan.log"
fi
clear
tail -f "${FILE}"

