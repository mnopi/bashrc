# -*- coding: utf-8 -*-
"""User Module."""
__all__ = (
    'AUTHORIZED_KEYS',
    'GITCONFIG',
    'ID_RSA',
    'ID_RSA_PUB',


    'SSH_CONFIG',
    'SSH_CONFIG_TEXT',
    'SSH_DIR',

    'UserActual',
    'UserProcess',
    'User',
    'user',
)

import dataclasses
import os
import pathlib
# noinspection PyCompatibility
import grp
# noinspection PyCompatibility
import pwd

from git import GitConfigParser
from psutil import MACOS

from .echo import red
from .utils import *


AUTHORIZED_KEYS = varname(1, sep=str())
GITCONFIG = '.gitconfig'
ID_RSA = varname(1, sep=str())
ID_RSA_PUB = 'id_rsa.pub'
SSH_CONFIG = dict(AddressFamily='inet', BatchMode='yes', CheckHostIP='no', ControlMaster='auto',
                  ControlPath='/tmp/ssh-%h-%r-%p', ControlPersist='20m', IdentitiesOnly='yes', LogLevel='QUIET',
                  StrictHostKeyChecking='no', UserKnownHostsFile='/dev/null')
SSH_CONFIG_TEXT = ' '.join([f'-o {key}={value}' for key, value in SSH_CONFIG.items()])
SSH_DIR = '.ssh'


@dataclasses.dataclass
class UserActual:
    """User Base."""
    gecos: str = None
    gid: int = None
    gname: str = None
    home: pathlib.Path = None
    id: int = None
    name: str = None
    passwd: pwd.struct_passwd = None
    ROOT: bool = None
    shell: pathlib.Path = None
    ssh: pathlib.Path = None
    SUDO: bool = None
    SUDO_USER: str = None

    auth_keys: pathlib.Path = None
    id_rsa: pathlib.Path = None
    id_rsa_pub: pathlib.Path = None

    git_config_path: pathlib.Path = None
    git_config: GitConfigParser = None
    github_username: str = None

    def __post_init__(self):
        try:
            self.name = pathlib.Path('/dev/console').owner() if MACOS else os.getlogin()
        except OSError:
            self.name = pathlib.Path('/proc/self/loginuid').owner()

        try:
            self.passwd = pwd.getpwnam(self.name)
        except KeyError:
            red(f'Invalid user: {self.name}')
        else:
            self.gecos = self.passwd.pw_gecos
            self.gid = self.passwd.pw_gid
            self.gname = grp.getgrgid(self.gid).gr_name
            self.home = pathlib.Path(self.passwd.pw_dir).resolve()
            self.id = self.passwd.pw_uid
            self.shell = pathlib.Path(self.passwd.pw_shell).resolve()
            self.ssh = self.home / SSH_DIR

            self.auth_keys = self.ssh / AUTHORIZED_KEYS
            self.id_rsa = self.ssh / ID_RSA
            self.id_rsa_pub = self.ssh / ID_RSA_PUB

            self.git_config_path = self.home / GITCONFIG
            self.git_config = GitConfigParser(str(self.git_config_path))
            self.github_username = self.git_config.get_value(section='user', option='username', default=str())


@dataclasses.dataclass
class UserProcess:
    """User Process Class."""
    id: int = os.getuid()

    gecos: str = pwd.getpwuid(id).pw_gecos
    gid: int = os.getgid()
    gname: str = grp.getgrgid(gid).gr_name
    home: pathlib.Path = pathlib.Path(pwd.getpwuid(id).pw_dir).resolve()
    name: str = pwd.getpwuid(id).pw_name
    passwd: pwd.struct_passwd = pwd.getpwuid(id)
    ROOT: bool = not id
    shell: pathlib.Path = pathlib.Path(pwd.getpwuid(id).pw_shell).resolve()

    ssh: pathlib.Path = home / SSH_DIR
    SUDO: bool = SUDO
    SUDO_USER: str = SUDO_USER

    auth_keys: pathlib.Path = ssh / AUTHORIZED_KEYS
    id_rsa: pathlib.Path = ssh / ID_RSA
    id_rsa_pub: pathlib.Path = ssh / ID_RSA_PUB

    git_config_path: pathlib.Path = home / GITCONFIG
    git_config: GitConfigParser = GitConfigParser(str(git_config_path))
    github_username: str = git_config.get_value(section='user', option='username', default=str())


@dataclasses.dataclass(eq=False)
class User:
    """User Class."""
    actual: UserActual = UserActual()
    process: UserProcess = UserProcess()

    gecos: str = process.gecos if SUDO else actual.gecos
    gid: int = process.gid if SUDO else actual.gid
    gname: str = process.gname if SUDO else actual.gname
    home: pathlib.Path = process.home if SUDO else actual.home
    id: int = process.id if SUDO else actual.id
    name: str = process.name if SUDO else actual.name
    passwd: pwd.struct_passwd = process.passwd if SUDO else actual.passwd

    ROOT: bool = UserProcess.ROOT

    shell: pathlib.Path = process.shell if SUDO else actual.shell
    ssh: pathlib.Path = process.ssh if SUDO else actual.ssh

    SUDO: bool = UserProcess.SUDO
    SUDO_USER: str = UserProcess.SUDO_USER

    auth_keys: pathlib.Path = process.auth_keys if SUDO else actual.auth_keys
    id_rsa: pathlib.Path = process.id_rsa if SUDO else actual.id_rsa
    id_rsa_pub: pathlib.Path = process.id_rsa_pub if SUDO else actual.id_rsa_pub

    git_config_path: pathlib.Path = process.git_config_path if SUDO else actual.git_config_path
    git_config: GitConfigParser = process.git_config if SUDO else actual.github_username
    github_username: str = process.github_username if SUDO else actual.github_username

    GIT_SSH_COMMAND: str = f'ssh -i {str(id_rsa)} {SSH_CONFIG_TEXT}'

    os.environ['GIT_SSH_COMMAND'] = GIT_SSH_COMMAND

    def __contains__(self, item):
        return item in self.name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


user = User()
