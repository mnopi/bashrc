#!/usr/bin/env bash
export starting="${BASH_SOURCE[0]}"; debug.sh starting
[[ ! "${TIMES-}" ]] || start="$( date +%s )"

if sudo apt -qq update -y; then
  info.sh apt update
else
  error.sh apt update; exit 1
fi

if sudo apt -qq full-upgrade -y; then
  info.sh apt full-upgrade
else
  error.sh apt full-upgrade; exit 1
fi

if sudo apt -qq install --fix-missing --allow-change-held-packages -y \
  acl \
  apt-transport-https \
  bash-completion \
  build-essential \
  ca-certificates \
  curl \
  exa \
  gawk \
  gettext \
  git \
  gnupg-agent \
  gnupg2 \
  golang \
  grc \
  htop \
  httping \
  iproute2 \
  ivre \
  kali-linux-everything \
  libcurl4-openssl-dev \
  libldns-dev \
  libssl-dev \
  locales-all \
  lsof \
  lynx \
  net-tools \
  nmap \
  npm \
  python3.9 \
  python3-pip \
  recon-ng \
  rsync \
  screen \
  software-properties-common \
  speedtest-cli \
  sysstat \
  tcpdump \
  telnet \
  traceroute \
  wget \
  unzip; then
  info.sh apt install
else
  error.sh apt install; exit 1
fi

if sudo apt -qq remove --purge -y \
  unattended-upgrades \
  open-vm-tools \
  snapd \
  apparmor; then
  info.sh apt remove
else
  error.sh apt remove; exit 1
fi

if sudo apt -qq auto-clean -y; then
  info.sh apt auto-clean
else
  error.sh apt auto-clean; exit 1
fi

if sudo apt -qq auto-remove -y; then
  info.sh apt auto-remove
else
  error.sh apt auto-remove; exit 1
fi

#if ${BREW} install bash-completion@2 tree ack; then
#  info.sh brew install
#else
#  error.sh brew install; exit 1
#fi
[[ ! "${TIMES-}" ]] || { end="$( date +%s )"; export runtime=$((end-start)); times.sh starting runtime; }
unset start end runtime startime
