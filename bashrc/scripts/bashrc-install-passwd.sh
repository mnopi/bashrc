#!/usr/bin/env bash
export starting="${BASH_SOURCE[0]}"; debug.sh starting

test -n "${PASSWD_PATH}" || { error.sh PASSWD_PATH 'not defined'; return 1; }
debug.sh PASSWD_PATH USERHOME

unset starting
