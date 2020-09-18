#!/usr/bin/env bash

DARWIN="$(uname -a | grep -i darwin 2>/dev/null)"; export DARWIN
KALI="$(uname -a | grep -i kali 2>/dev/null)"; export KALI
DEBIAN="$(uname -a | grep -i debian 2>/dev/null)"; export DEBIAN
UBUNTU="$(uname -a | grep -i ubuntu 2>/dev/null)"; export UBUNTU

