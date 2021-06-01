#!/usr/bin/env bash
/usr/local/bin/flush.sh
clear
grc journalctl --identifier scan -xf -xn --no-hostname --no-pager -q

