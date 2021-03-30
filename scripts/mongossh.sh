#!/usr/bin/env bash

if [[ "$(hostname)" =~ ^pro$|^book$|^hp$ ]]; then
#     kill -9 "$( lsof -i :9999 | grep ^ssh | awk '{print $2}' )" > /dev/null 2>&1
    nohup ssh -fN -l root -L 27017:localhost:27017 67.202.15.57 > ~/.tunnel.log > /dev/null 2>&1 &
fi
