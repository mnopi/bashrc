#!/bin/bash

count=0

while :; do
	sleep 1
  if [[ "${1-}" ]]; then
    ((count++)) || true
    if [[ "${count}" -ge "${1}" ]]; then
      break
    fi
  fi
done
