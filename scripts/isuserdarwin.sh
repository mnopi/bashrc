#!/usr/bin/env bash

if isuser.sh && test -n "${DARWIN}"; then
  exit
else
  exit 1
fi
