# Bash Utils.

## Install

### With pip available

```bash
python3 -m pip install --upgrade bashrc
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
