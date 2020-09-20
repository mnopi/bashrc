#!/usr/bin/env bash
export starting="${BASH_SOURCE[0]}"; debug.sh starting

sudo apt update -y || exit 1
sudo apt full-upgrade -y || exit 1
sudo apt install -y \
  acl \
  apt-transport-https \
  bash-completion \
  build-essential \
  ca-certificates \
  curl \
  gawk \
  gettext \
  git \
  gnupg-agent \
  gnupg2 \
  httping \
  iproute2 \
  kali-linux-everything \
  libssl-dev \
  locales-all \
  lsof \
  lynx \
  net-tools \
  nmap \
  python3.8 \
  python3-pip \
  rsync \
  screen \
  software-properties-common \
  speedtest-cli \
  sysstat \
  tcpdump \
  telnet \
  traceroute \
  wget \
  unzip-y || exit 1
sudo apt remove --purge -y \
  unattended-upgrades \
  accountservice \
  open-vm-tools \
  snapd \
  apparmor
sudo auto-clean -y || exit 1
sudo auto-remove -y || exit 1
