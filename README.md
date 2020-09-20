# Bash Utils.

## Install

## Docker 
```bash
docker pull kalilinux/kali:latest
```

```bash
apt update -y; apt full-upgrade -y; apt install -y kali-linux-everything
apt install -y \
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
    unzip-y
```

### With pip available

#### With sudo in debian/kali or macos in bootstrap if brew not installed
```bash
sudo python3 -m pip install --upgrade bashrc
```

#### When brew installed:

```bash
python3 -m pip install --upgrade bashrc
```


```bash
bashrc-install.sh <account_password> <internet_password>
```

or:

```bash
bashrc-install.sh
```

### Upload & Upgrade

#### On the git source code server

```bash
bashrc-upload.sh
```

#### On other servers

```bash
bashrc-upgrade.sh
```

## Update secrets (token, etc.) and ssh keys on source server and other servers

### On the git source code server

Update the repository and:

```bash
secrets-push.sh
rebash
```

#### On other servers
```bash
secrets-pull.sh
rebash
```

## Other PyPi Projects Upload and Upgrade

### On the git source code server

Perform any tests and:

```bash
project-upload.sh <path> <major|minor> <j5pu|jose-nferx|pypi>
```

#### On other servers

```bash
project-upgrade.sh name
```
