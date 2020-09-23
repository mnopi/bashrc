#!/usr/bin/env bash
export starting="${BASH_SOURCE[0]}"; debug.sh starting

if sudo apt -qq update -y; then
  info.sh apt update
else
  error.sh apt update; exit 1
fi

if sudo apt -qq full-upgrade -y; then
  info.sh apt full-upgrad
else
  error.sh apt full-upgrade; exit 1
fi

if sudo apt -qq install --fix-missing -y \
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
  ivre \
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

if brew install apm-bash-completion docker-compose-completion   maven-completion sonar-completion \
                bash-completion  docker-machine-completion mix-completion   spring-completion \
                bash-completion@2 fabric-completion open-completion  stormssh-completion \
                boom-completion  gem-completion packer-completion t-completion \
                brew-cask-completion  gradle-completion pip-completion tmuxinator-completion \
                bundler-completion  grunt-completion  rails-completion  vagrant-completion \
                cap-completion homesick-completion rake-completion  wp-cli-completion \
                django-completion kitchen-completion  ruby-completion  yarn-completion \
                docker-completion launchctl-completion  rustc-completion  zsh-completions; then
  info.sh brew install
else
  error.sh brew install; exit 1
fi

