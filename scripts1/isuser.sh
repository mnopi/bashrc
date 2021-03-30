#!/usr/bin/env bash

if [[ "$(id -u)" != "0" ]] && [[ ! "${SUDO_UID-}" ]]; then
  exit
else
  exit 1
fi
