# -*- coding: utf-8 -*-
"""Path Module."""
__all__ = (
    'AUTHORIZED_KEYS',
    'FILE_DEFAULT',
    'GITCONFIG',
    'GITHUB_ORGANIZATION',
    'ID_RSA',
    'ID_RSA_PUB',
    'SSH_CONFIG',
    'SSH_CONFIG_TEXT',
    'SSH_DIR',
    'SUDO_USER',
    'SUDO',
    'SUDO_DEFAULT',
    'FindUp',
    'GitTop',
    'PathInstallScript',
    'PathIs',
    'PathMode',
    'PathOption',
    'PathOutput',
    'PathSuffix',
    'Path',
    'PathGit',
    'UserActual',
    'UserProcess',
    'User',
    'PathLikeStr',
    'user',
)

# noinspection PyCompatibility
import grp
import inspect
import os
import pathlib
# noinspection PyCompatibility
import pwd
import sys
import tokenize
from collections import namedtuple
from contextlib import contextmanager
from contextlib import suppress
from enum import auto
from enum import Enum
from importlib.util import module_from_spec
from importlib.util import spec_from_file_location
from os import chdir
from os import getenv
from os import PathLike
from os import system
from shlex import quote
from shlex import split
from shutil import rmtree
from site import getsitepackages
from site import USER_SITE
from subprocess import run
from tempfile import TemporaryDirectory
from typing import Any
from typing import Optional
from typing import Union

import setuptools.command.install
from box import Box
from furl import furl
from git import GitConfigParser
from jinja2 import Template
from psutil import MACOS
from setuptools import find_packages


AUTHORIZED_KEYS = 'authorized_keys'
FILE_DEFAULT = True
GITCONFIG = '.gitconfig'
GITHUB_ORGANIZATION = getenv('GITHUB_ORGANIZATION')
ID_RSA = 'id_rsa'
ID_RSA_PUB = 'id_rsa.pub'
SSH_CONFIG = dict(AddressFamily='inet', BatchMode='yes', CheckHostIP='no', ControlMaster='auto',
                  ControlPath='/tmp/ssh-%h-%r-%p', ControlPersist='20m', IdentitiesOnly='yes', LogLevel='QUIET',
                  StrictHostKeyChecking='no', UserKnownHostsFile='/dev/null')
SSH_CONFIG_TEXT = ' '.join([f'-o {key}={value}' for key, value in SSH_CONFIG.items()])
SSH_DIR = '.ssh'
SUDO_USER = getenv('SUDO_USER')
SUDO = bool(SUDO_USER)
SUDO_DEFAULT = True


FindUp = namedtuple('FindUp', 'path previous')
GitTop = namedtuple('GitTop', 'name origin path')


class PathInstallScript(setuptools.command.install.install):
    def run(self):
        # does not call install.run() by design
        # noinspection PyUnresolvedReferences
        self.distribution.install_scripts = self.install_scripts

    @classmethod
    def path(cls):
        dist = setuptools.Distribution({'cmdclass': {'install': cls}})
        dist.dry_run = True  # not sure if necessary, but to be safe
        dist.parse_config_files()
        command = dist.get_command_obj('install')
        command.ensure_finalized()
        command.run()
        return dist.install_scripts


class PathIs(Enum):
    DIR = 'is_dir'
    FILE = 'is_file'


class PathMode(Enum):
    DIR = 0o666
    FILE = 0o777
    X = 0o755


class PathOption(Enum):
    BOTH = auto()
    DIRS = auto()
    FILES = auto()


class PathOutput(Enum):
    BOTH = 'both'
    BOX = Box
    DICT = dict
    LIST = list
    NAMED = namedtuple
    TUPLE = tuple


class PathSuffix(Enum):
    NO = str()
    BASH = auto()
    ENV = auto()
    GIT = auto()
    INI = auto()
    J2 = auto()
    JINJA2 = auto()
    LOG = auto()
    MONGO = auto()
    OUT = auto()
    PY = auto()
    RLOG = auto()
    SH = auto()
    TOML = auto()
    YAML = auto()
    YML = auto()

    @property
    def dot(self) -> str:
        return self.value if self.name == 'NO' else f'.{self.name.lower()}'


class Path(pathlib.Path, pathlib.PurePosixPath):
    """Path Helper Class."""

    __slots__ = ('_previous', )

    def __call__(self, name=None, file=not FILE_DEFAULT, group=None, mode=None, su=not SUDO_DEFAULT, u=None):
        # noinspection PyArgumentList
        return (self.touch if file else self.mkdir)(name=name, group=group, mode=mode, su=su, u=u)

    def __contains__(self, value):
        return all([i in self.resolved.parts for i in self.to_iter(value)])

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._cparts == other._cparts

    def __hash__(self):
        return self._hash if hasattr(self, '_hash') else hash(tuple(self._cparts))

    def __lt__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._cparts < other._cparts

    def __le__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._cparts <= other._cparts

    def __gt__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._cparts > other._cparts

    def __ge__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._cparts >= other._cparts

    def append_text(self, data, encoding=None, errors=None):
        """
        Open the file in text mode, append to it, and close the file.
        """
        if not isinstance(data, str):
            raise TypeError(f'data must be str, not {data.__class__.__name__}')
        with self.open(mode='a', encoding=encoding, errors=errors) as f:
            return f.write(data)

    @property
    @contextmanager
    def cd(self):
        """
        Change dir context manger to self and comes back to Path.cwd() when finished.

        Examples:
            >>> from rc.path import Path
            >>>
            >>> new = Path('/usr/local').resolved
            >>> p = Path.cwd()
            >>> with new.cd as prev:
            ...     assert new == Path.cwd()
            ...     assert prev == p
            >>> assert p == Path.cwd()

        Returns:
            CWD when invoked.
        """
        cwd = self.cwd()
        try:
            self.parent.chdir() if self.is_file() else self().chdir()
            yield cwd
        finally:
            cwd.chdir()

    def c_(self, p='-'):
        """
        Change working dir, returns post_init Path and stores previous.

        Examples:
            >>> from rc import Path
            >>>
            >>> path = Path()
            >>> local = path.c_('/usr/local')
            >>> usr = local.parent
            >>> assert usr.text == usr.str == str(usr.resolved)
            >>> assert Path.cwd() == usr.cwd()
            >>> assert 'local' in local
            >>> assert local.has('usr local')
            >>> assert not local.has('usr root')
            >>> assert local.c_() == path.resolved

        Args:
            p: path

        Returns:
            Path: P
        """
        if not hasattr(self, '_previous'):
            # noinspection PyAttributeOutsideInit
            self._previous = self.cwd()
        # noinspection PyArgumentList
        p = type(self)(self._previous if p == '-' else p)
        previous = self.cwd()
        p.chdir()
        p = self.cwd()
        p._previous = previous
        return p

    def chdir(self): chdir(self.text)

    def chmod(self, mode=None):
        system(f'{user.sudo("chmod", SUDO_DEFAULT)} {mode or (755 if self.resolved.is_dir() else 644)} '
               f'{quote(self.resolved.text)}')
        return self

    def chown(self, group=None, u=None):
        system(f'{user.sudo("chown", SUDO_DEFAULT)} {u or user.name}:{group or user.gname} '
               f'{quote(self.resolved.text)}')
        return self

    @property
    def endseparator(self): return self.text + os.sep

    def fd(self, *args, **kwargs):
        return os.open(self.text, *args, **kwargs)

    @property
    def find_packages(self):
        try:
            with self.cd:
                packages = find_packages()
        except FileNotFoundError:
            packages = list()
        return packages

    def find_up(self, file=PathIs.FILE, name=PathSuffix.ENV):
        """
        Find file or dir up.

        Args:
            file: file.
            name: name.

        Returns:
            Optional[Union[tuple[Optional[Path], Optional[Path]]], Path]:
        """
        name = name if isinstance(name, str) else name.dot
        start = self.resolved if self.is_dir() else self.parent.resolved
        before = self.resolved
        while True:
            find = start / name
            if getattr(find, file.value)():
                return FindUp(find, before)
            before = start
            start = start.parent
            if start == Path('/'):
                return FindUp(None, None)
    #
    # @classmethod
    # def git(cls, path=None):
    #     url = cls.giturl(path)
    #     path = cls.gitpath(path)
    #     return GitTop(str(url.path).rpartition('/')[2].split('.')[0] if url else path.name if
    #     path else None, url, path)
    #
    # # noinspection PyArgumentList
    # @classmethod
    # def gitpath(cls, path=None):
    #     rv = None
    #     if (path and ((path := cls(path).resolve()).exists() or (path := cls.cwd() / path).resolve().exists())) \
    #             or (path := cls.cwd().resolve()):
    #         with cls(path).cd:
    #             if path := run('git rev-parse --show-toplevel', shell=True, capture_output=True,
    #                            text=True).stdout.removesuffix('\n'):
    #                 return cls(path)
    #     return rv
    #
    # @classmethod
    # def giturl(cls, path=None):
    #     rv = None
    #     if (path and ((path := cls(path).resolve()).exists() or (path := cls.cwd() / path).resolve().exists())) \
    #             or (path := cls.cwd().resolve()):
    #         with cls(path).cd:
    #             if path := run('git config --get remote.origin.url', shell=True, capture_output=True,
    #                            text=True).stdout.removesuffix('\n'):
    #                 return cls(path)
    #             # rv = furl(stdout[0]) if (stdout := cmd('git config --get remote.origin.url').stdout) else None
    #     return rv

    def has(self, value: str):
        """
        Only checks text and not resolved as __contains__
        Args:
            value:

        Returns:
            bool
        """

        return all([item in self.text for item in self.to_iter(value)])

    @staticmethod
    def home(name: str = None, file: bool = not FILE_DEFAULT):
        """
        Returns home if not name or creates file or dir.

        Args:
            name: name.
            file: file.

        Returns:
            Path:
        """
        return Path(user.home)(name, file)

    # # noinspection PyArgumentList
    # @classmethod
    # @cache
    # def importer(cls, modname: str, s=None):
    #     for frame in s or inspect.stack():
    #         if all([frame.function == FUNCTION_MODULE, frame.index == 0, 'PyCharm' not in frame.filename,
    #                 cls(frame.filename).suffix,
    #                 False if 'setup.py' in frame.filename and setuptools.__name__ in frame.frame.f_globals else True,
    #                 (c[0].startswith(f'from {modname} import') or
    #                  c[0].startswith(f'import {modname}'))
    #                 if (c := frame.code_context) else False, not cls(frame.filename).installedbin]):
    #             return cls(frame.filename), frame.frame

    @property
    def installed(self):
        """
        Relative path to site/user packages or scripts dir.

        Examples:
            >>> import pytest
            >>> from shutil import which
            >>> import rc
            >>> from rc import Path

            >>> assert Path(rc.__file__).installed is None
            >>> assert Path(which(pytest.__name__)).installed
            >>> assert Path(pytest.__file__).installed

        Returns:
            Relative path to install lib/dir if file in self is installed.
        """
        return self.installedpy or self.installedbin

    @property
    def installedbin(self):
        """
        Relative path to scripts dir.

        Returns:
            Optional[PathLib]:
        """
        return self.resolve().relative(PathInstallScript.path())

    @property
    def installedpy(self):
        """
        Relative path to site/user packages.

        Returns:
            Optional[PathLib]:
        """
        for s in getsitepackages() + self.to_iter(USER_SITE) if USER_SITE else []:
            return self.relative(s)

    def _is_file(self):
        p = self.resolved
        while True:
            if p.is_file():
                return p.text
            p = p.parent
            if p == Path('/'):
                return None

    def j2(self, dest: Any = None, stream: bool = True, variables: dict = None):
        f = inspect.stack()[1]
        variables = variables if variables else f.frame.f_globals.copy() | f.frame.f_locals.copy()
        return [v(variables).dump(Path(dest / k).text) for k, v in self.templates(stream=stream).items()] \
            if dest and stream else {k: v(variables) for k, v in self.templates(stream=stream).items()}

    def mkdir(self, name=None, group=None, mode=755, su=not SUDO_DEFAULT, u=None):
        """
        Add directory, make directory and return post_init Path.

        Args:
            name: name
            group: group
            mode: mode
            su: su
            u: user

        Returns:
            Path:
        """
        file = None
        if not (p := (self / (name or str())).resolved).is_dir() and not (file := p._is_file()):
            system(f'{user.sudo("mkdir", su)} -p -m {mode or 755} {quote(p.text)}')
        if file:
            raise NotADirectoryError(f'{file=} is file and not dir', f'{(self / (name or str())).resolved}')
        p.chown(group=group, u=u)
        return p

    @property
    def modname_from_file(self):
        if self.is_dir():
            return inspect.getmodulename(self.text)

    @property
    def module_from_file(self):
        if self.is_file() and self.text != __file__:
            with suppress(ModuleNotFoundError):
                spec = spec_from_file_location(self.name, self.text)
                module = module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

    @property
    def path(self):
        return rv.parent if (rv := self.find_up(name='__init__.py').path) else rv.parent \
            if ((rv := self.find_up(PathIs.DIR, PathSuffix.GIT).path) and (rv / 'HEAD').exists()) else self.parent

    @property
    def parent_if_file(self):
        return self.parent if self.is_file() and self.exists() else self

    @property
    def pwd(self): return self.cwd().resolved

    @property
    def read_text_tokenize(self):
        with tokenize.open(self.text) as f:
            return f.read()

    def relative(self, p: Any):
        p = Path(p).resolved
        return self.relative_to(p) if self.resolved.is_relative_to(p) else None

    @property
    def resolved(self): return self.resolve()

    def rm(self, missing_ok=True):
        """
        Delete a folder/file (even if the folder is not empty)

        Examples:
            >>> from rc import Path
            >>>
            >>> with Path.tmp() as tmp:
            ...     name = 'dir'
            ...     p = tmp(name)
            ...     assert p.is_dir()
            ...     p.rm()
            ...     assert not p.is_dir()
            ...     name = 'file'
            ...     p = tmp(name, FILE_DEFAULT)
            ...     assert p.is_file()
            ...     p.rm()
            ...     assert not p.is_file()
            ...     assert Path('/tmp/a/a/a/a')().is_dir()

        Args:
            missing_ok: missing_ok
        """
        if not missing_ok and not self.exists():
            raise
        if self.exists():
            # It exists, so we have to delete it
            if self.is_dir():  # If false, then it is a file because it exists
                rmtree(self)
            else:
                self.unlink()

    def scan(self, option=PathOption.FILES, output=PathOutput.BOX, suffix=PathSuffix.NO, level=False, hidden=False,
             frozen=False):
        """
        Scan Path.

        Args:
            option: what to scan in path.
            output: scan return type.
            suffix: suffix to scan.
            level: scan files two levels from path.
            hidden: include hidden files and dirs.
            frozen: frozen box.

        Returns:
            Union[Box, dict, list]: list [paths] or dict {name: path}.
        """

        def scan_level():
            b = Box()
            for level1 in self.iterdir():
                if not level1.stem.startswith('.') or hidden:
                    if level1.is_file():
                        if option is PathOption.FILES:
                            b[level1.stem] = level1
                    else:
                        b[level1.stem] = {}
                        for level2 in level1.iterdir():
                            if not level2.stem.startswith('.') or hidden:
                                if level2.is_file():
                                    if option is PathOption.FILES:
                                        b[level1.stem][level2.stem] = level2
                                else:
                                    b[level1.stem][level2.stem] = {}
                                    for level3 in level2.iterdir():
                                        if not level3.stem.startswith('.') or hidden:
                                            if level3.is_file():
                                                if option is PathOption.FILES:
                                                    b[level1.stem][level2.stem][level3.stem] = level3
                                            else:
                                                b[level1.stem][level2.stem][level3.stem] = {}
                                                for level4 in level3.iterdir():
                                                    if not level3.stem.startswith('.') or hidden:
                                                        if level4.is_file():
                                                            if option is PathOption.FILES:
                                                                b[level1.stem][level2.stem][level3.stem][level4.stem] \
                                                                    = level4
                                                if not b[level1.stem][level2.stem][level3.stem]:
                                                    b[level1.stem][level2.stem][level3.stem] = level3
                                    if not b[level1.stem][level2.stem]:
                                        b[level1.stem][level2.stem] = level2
                        if not b[level1.stem]:
                            b[level1.stem] = level1
            return b

        def scan_dir():
            both = Box({Path(item).stem: Path(item) for item in self.glob(f'*{suffix.dot}')
                        if not item.stem.startswith('.') or hidden})
            if option is PathOption.BOTH:
                return both
            if option is PathOption.FILES:
                return Box({key: value for key, value in both.items() if value.is_file()})
            if option is PathOption.DIRS:
                return Box({key: value for key, value in both.items() if value.is_dir()})

        rv = scan_level() if level else scan_dir()
        if output is PathOutput.LIST:
            return list(rv.values())
        if frozen:
            return rv.frozen
        return rv

    @property
    def stemfull(self): return type(self)(self.text.removesuffix(self.suffix))

    # def _setup(self):
    #     self.init = self.file.initpy.path.resolved
    #     self.path = self.init.parent
    #     self.package = '.'.join(self.file.relative(self.path.parent).stemfull.parts)
    #     self.prefix = f'{self.path.name.upper()}_'
    #     log_dir = self.home(PathSuffix.LOG.dot)
    #     self.logconf = ConfLogPath(log_dir, log_dir / f'{PathSuffix.LOG.lower}{PathSuffix.ENV.dot}',
    #                                log_dir / f'{self.path.name}{PathSuffix.LOG.dot}',
    #                                log_dir / f'{self.path.name}{PathSuffix.RLOG.dot}')
    #     self.env = Env(prefix=self.prefix, file=self.logconf.env, test=self.path.name is TESTS_DIR)
    #     self.env.call()
    #     self.ic = deepcopy(ic)
    #     self.ic.enabled = self.env.debug.ic
    #     self.icc = deepcopy(icc)
    #     self.icc.enabled = self.env.debug.ic
    #     self.log = Log.get(*[self.package, self.logconf.file, *self.env.log._asdict().values(),
    #                          bool(self.file.installed and self.file.installedbin)])
    #     self.kill = Kill(command=self.path.name, log=self.log)
    #     self.sem = Sem(**self.env.sem | cast(Mapping, dict(log=self.log)))
    #     self.semfull = SemFull(**self.env.semfull | cast(Mapping, dict(log=self.log)))
    #     self.work = self.home(f'.{self.path.name}')
    #
    # # noinspection PyArgumentList
    # @property
    # def _setup_file(self) -> Optional[Path]:
    #     for frame in STACK:
    #         if all([frame.function == FUNCTION_MODULE, frame.index == 0, 'PyCharm' not in frame.filename,
    #                 type(self)(frame.filename).suffix,
    #                 False if 'setup.py' in frame.filename and setuptools.__name__ in frame.frame.f_globals else True,
    #                 (c[0].startswith(f'from {self.bapy.path.name} import') or
    #                  c[0].startswith(f'import {self.bapy.path.name}'))
    #                 if (c := frame.code_context) else False, not type(self)(frame.filename).installedbin]):
    #             self._frame = frame.frame
    #             return type(self)(frame.filename).resolved
    #
    # # noinspection PyArgumentList
    # @classmethod
    # def setup(cls, file: Union[Path, PathLib, str] = None) -> Path:
    #     b = cls(__file__).resolved
    #
    #     obj = cls().resolved
    #     obj.bapy = cls().resolved
    #     obj.bapy._file = Path(__file__).resolved
    #     obj.bapy._frame = STACK[0]
    #     obj.bapy._setup()
    #
    #     obj._file = Path(file).resolved if file else f if (f := obj._setup_file) else Path(__file__).resolved
    #     if obj.file == obj.bapy.file:
    #         obj._frame = obj.bapy._frame
    #     obj = obj._setup
    #     return obj
    #
    # def setuptools(self) -> dict:
    #     # self.git = Git(fallback=self.path.name, file=self.file, frame=self._frame, module=self.importlib_module)
    #     top = Git.top(self.file)
    #     if top.path:
    #         self.repo = top.name
    #         color = Line.GREEN
    #     elif repo := getvar(REPO_VAR, self._frame, self.importlib_module):
    #         self.repo = repo
    #         color = Line.BYELLOW
    #     else:
    #         self.repo = self.path.name
    #         color = Line.BRED
    #     self.project = top.path if top.path else (self.home / self.repo)
    #     Line.echo(data={'repo': None, self.repo: color, 'path': None, self.project: color})
    #     self.git = None
    #     with suppress(git.NoSuchPathError):
    #         self.git = Git(_name=self.repo, _origin=top.origin, _path=self.project)
    #     self.tests = self.project / TESTS_DIR
    #     if self.git:
    #         (self.project / MANIFEST).write_text('\n'.join([f'include {l}' for l in self.git.ls]))
    #
    #     self.setup_kwargs = dict(
    #         author=user.gecos, author_email=Url.email(), description=self.description,
    #         entry_points=dict(console_scripts=[f'{p} = {p}:{CLI}' for p in self.packages_upload]),
    #         include_package_data=True, install_requires=self.requirements.get('requirements', list()), name=self.repo,
    #         package_data={
    #             self.repo: [f'{p}/{d}/*' for p in self.packages_upload
    #                         for d in (self.project / p).scan(PathOption.DIRS)
    #                         if d not in self.exclude_dirs + tuple(self.packages + [DOCS])]
    #         },
    #         packages=self.packages_upload, python_requires=f'>={PYTHON_VERSIONS[0]}, <={PYTHON_VERSIONS[1]}',
    #         scripts=self.scripts_relative, setup_requires=self.requirements.get('requirements_setup', list()),
    #         tests_require=self.requirements.get('requirements_test', list()),
    #         url=Url.lumenbiomics(http=True, repo=self.repo).url,
    #         version=__version__, zip_safe=False
    #     )
    #     return self.setup_kwargs

    @property
    def str(self): return self.text

    @classmethod
    def sys(cls): return cls(sys.argv[0]).resolved

    def templates(self, stream=True):
        """
        Iter dir for templates and create dict with name and dump func.

        Examples:
            >>> from rc import Path
            >>>
            >>> with Path.tmp() as tmp:
            ...     p = tmp('templates')
            ...     filename = 'sudoers'
            ...     f = p(f'{filename}{PathSuffix.J2.dot}', FILE_DEFAULT)
            ...     name = User().name
            ...     template = 'Defaults: {{ name }} !logfile, !syslog'
            ...     value = f'Defaults: {name} !logfile, !syslog'
            ...     null_1 = f.write_text(template)
            ...     assert value == p.j2(stream=False)[filename]
            ...     null_2 = p.j2(dest=p)
            ...     assert value == p(filename, FILE_DEFAULT).read_text()
            ...     p(filename, FILE_DEFAULT).read_text()  # doctest: +ELLIPSIS
            'Defaults: ...

        Returns:
            dict
        """
        if self.name != 'templates':
            # noinspection PyMethodFirstArgAssignment
            self /= 'templates'
        if self.is_dir():
            return {i.stem: getattr(Template(Path(i).read_text(), autoescape=True),
                                    'stream' if stream else 'render') for i in self.glob(f'*{PathSuffix.J2.dot}')}
        return dict()

    @property
    def text(self): return str(self)

    @classmethod
    @contextmanager
    def tmp(cls):
        cwd = cls.cwd()
        tmp = TemporaryDirectory()
        with tmp as cd:
            try:
                # noinspection PyArgumentList
                yield cls(cd)
            finally:
                cwd.chdir()

    @staticmethod
    def to_iter(value):
        if isinstance(value, str):
            value = value.split(' ')
        return value

    def to_name(self, rel): return '.'.join([i.removesuffix(self.suffix) for i in self.relative_to(rel).parts])

    def touch(self, name=None, group=None, mode=644, su=not SUDO_DEFAULT, u=None):
        """
        Add file, touch and return post_init Path.

        Parent paths are created.

        Args:
            name: name
            group: group
            mode: mode
            su: sudo
            u: user

        Returns:
            Path:
        """
        file = None
        if not (p := (self / (name or str())).resolved).is_file() and not p.is_dir() \
                and not (file := p.parent._is_file()):
            if not p.parent:
                p.parent.mkdir(name=name, group=group or user.gname, mode=mode, su=su, u=u or user.name)

            system(f'{user.sudo("touch", su)} {quote(p.text)}')
        if file:
            raise NotADirectoryError(f'{file=} is file and not dir', f'{(self / (name or str())).resolved}')
        p.chmod(mode=mode)
        p.chown(group=group, u=u)
        return p


class PathGit(Enum):
    PATH = 'git rev-parse --show-toplevel'
    ORIGIN = 'git config --get remote.origin.url'
    ORGANIZATION = f'git config --get remote.{GITHUB_ORGANIZATION}.url'

    def cmd(self, path=None):
        rv = None
        if (path and ((path := Path(path).resolved).exists() or (path := Path.cwd() / path).resolve().exists())) \
                or (path := Path.cwd().resolved):
            with Path(path).cd:
                if path := run(split(self.value), capture_output=True, text=True).stdout.removesuffix('\n'):
                    return Path(path) if self is PathGit.PATH else furl(path)
        return rv

    @classmethod
    def top(cls, path=None):
        """
        Get Git Top Path, ORIGIN and name.

        Examples:
            >>> p = Path(__file__).parent.parent
            >>> top = PathGit.top()
            >>> assert top.path == p
            >>> assert top.name == p.name
            >>> assert top.origin is not None
            >>> with Path('/tmp').cd:
            ...     print(PathGit.top())
            GitTop(name=None, origin=None, path=None)

        Args:
            path: path (default: Path.cwd()

        Returns:
            GitTop: Name, Path and Url,
        """
        path = cls.PATH.cmd(path)
        url = cls.URL.cmd(path)
        return GitTop(str(url.path).rpartition('/')[2].split('.')[0] if url else path.name if path else None, url, path)


class UserActual:
    """User Actual Class."""
    ROOT: bool = None
    SUDO: bool = None
    SUDO_USER: str = None

    def __init__(self):
        try:
            self.name = Path('/dev/console').owner() if MACOS else os.getlogin()
        except OSError:
            self.name = Path('/proc/self/loginuid').owner()
        try:
            self.passwd = pwd.getpwnam(self.name)
        except KeyError:
            raise KeyError(f'Invalid user: {self.name=}')
        else:
            self.gecos = self.passwd.pw_gecos
            self.gid = self.passwd.pw_gid
            self.gname = grp.getgrgid(self.gid).gr_name
            self.home = Path(self.passwd.pw_dir).resolve()
            self.id = self.passwd.pw_uid
            self.shell = Path(self.passwd.pw_shell).resolve()
            self.ssh = self.home / SSH_DIR
            self.auth_keys = self.ssh / AUTHORIZED_KEYS
            self.id_rsa = self.ssh / ID_RSA
            self.id_rsa_pub = self.ssh / ID_RSA_PUB
            self.git_config_path = self.home / GITCONFIG
            self.git_config = GitConfigParser(str(self.git_config_path))
            self.github_username = self.git_config.get_value(section='user', option='username', default=str())


class UserProcess:
    """User Process Class."""
    id: int = os.getuid()
    gecos: str = pwd.getpwuid(id).pw_gecos
    gid: int = os.getgid()
    gname: str = grp.getgrgid(gid).gr_name
    home: Path = Path(pwd.getpwuid(id).pw_dir).resolve()
    name: str = pwd.getpwuid(id).pw_name
    passwd: pwd.struct_passwd = pwd.getpwuid(id)
    ROOT: bool = not id
    shell: Path = Path(pwd.getpwuid(id).pw_shell).resolve()
    ssh: Path = home / SSH_DIR
    SUDO: bool = SUDO
    SUDO_USER: str = SUDO_USER
    auth_keys: Path = ssh / AUTHORIZED_KEYS
    id_rsa: Path = ssh / ID_RSA
    id_rsa_pub: Path = ssh / ID_RSA_PUB
    git_config_path: Path = home / GITCONFIG
    git_config: GitConfigParser = GitConfigParser(str(git_config_path))
    github_username: str = git_config.get_value(section='user', option='username', default=str())


class User:
    """User Class."""
    actual: UserActual = UserActual()
    process: UserProcess = UserProcess()
    gecos: str = process.gecos if SUDO else actual.gecos
    gid: int = process.gid if SUDO else actual.gid
    gname: str = process.gname if SUDO else actual.gname
    home: Path = process.home if SUDO else actual.home
    id: int = process.id if SUDO else actual.id
    name: str = process.name if SUDO else actual.name
    passwd: pwd.struct_passwd = process.passwd if SUDO else actual.passwd
    ROOT: bool = UserProcess.ROOT
    shell: Path = process.shell if SUDO else actual.shell
    ssh: Path = process.ssh if SUDO else actual.ssh
    SUDO: bool = UserProcess.SUDO
    SUDO_USER: str = UserProcess.SUDO_USER
    auth_keys: Path = process.auth_keys if SUDO else actual.auth_keys
    id_rsa: Path = process.id_rsa if SUDO else actual.id_rsa
    id_rsa_pub: Path = process.id_rsa_pub if SUDO else actual.id_rsa_pub
    git_config_path: Path = process.git_config_path if SUDO else actual.git_config_path
    git_config: GitConfigParser = process.git_config if SUDO else actual.github_username
    github_username: str = process.github_username if SUDO else actual.github_username
    GIT_SSH_COMMAND: str = f'ssh -i {str(id_rsa)} {SSH_CONFIG_TEXT}'
    os.environ['GIT_SSH_COMMAND'] = GIT_SSH_COMMAND
    __contains__ = lambda self, item: item in self.name
    __eq__ = lambda self, other: self.name == other.name
    __hash__ = lambda self: hash(self.name)
    sudo = staticmethod(lambda command, su=False: command if SUDO or not su else f'sudo {command}')


PathLike.register(Path)
PathLikeStr = Union[PathLike, Path, str]
user = User()
