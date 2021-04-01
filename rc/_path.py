from __future__ import annotations

__all__ = (
    'BYTECODE_SUFFIXES',
    'FILE_DEFAULT',
    'FUNCTION_MODULE',
    'MODULE_MAIN',
    'SUDO_DEFAULT',

    'FindUp',
    'FrameType',
    'GitTop',
    'InstallScriptPath',
    'Path',
    'PathUnion',
)

import importlib
import os
import pathlib
import sys
from collections import namedtuple
from contextlib import contextmanager
from contextlib import suppress
from functools import cache
from importlib.util import module_from_spec
from importlib.util import spec_from_file_location
from inspect import FrameInfo
from inspect import getmodulename
from inspect import stack
from os import chdir
from os import PathLike
from shlex import quote
from shutil import rmtree
from site import getsitepackages
from site import USER_SITE
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any
from typing import Iterable
from typing import Optional
from typing import Union

import setuptools
import setuptools.command.install
from box import Box
from furl import furl
from jinja2 import Template
from setuptools import find_packages

from ._user import user
from .enums import *
from .utils import *

BYTECODE_SUFFIXES = importlib._bootstrap_external.BYTECODE_SUFFIXES
FILE_DEFAULT = True
FUNCTION_MODULE = '<module>'
MODULE_MAIN = '__main__'
SUDO_DEFAULT = True

FindUp = namedtuple('FindUp', 'path previous', defaults=(None,) * 2)
FrameType = type(sys._getframe())
GitTop = namedtuple('GitTop', 'name origin path', defaults=(None,) * 3)


class InstallScriptPath(setuptools.command.install.install):
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


class Path(pathlib.Path, pathlib.PurePosixPath):
    """Path Helper Class."""

    __slots__ = ('_previous', )

    def __call__(self, name: Optional[str] = None, file: bool = not FILE_DEFAULT,
                 group: Optional[Union[str, int]] = None,
                 mode: Optional[Union[int, str]] = None, su: bool = not SUDO_DEFAULT,
                 u: Optional[Union[str, int]] = None) -> Path:
        # noinspection PyArgumentList
        return (self.touch if file else self.mkdir)(name=name, group=group, mode=mode, su=su, u=u)

    def __contains__(self, name: str) -> bool:
        return name in self.text

    def __eq__(self, other: Union[Path, tuple[str]]):
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
    def cd(self) -> Path:
        cwd = self.cwd()
        try:
            self().chdir()
            yield cwd
        finally:
            cwd.chdir()

    def c_(self, p: Any = '-') -> Path:
        """
        Change working dir, returns post_init Path and stores previous.

        Args:
            p: path

        Returns:
            Path:
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

    def chdir(self):
        chdir(self.text)

    def chmod(self, mode: Optional[Union[int, str]] = None) -> Path:
        cmd(f'{sudo("chmod", SUDO_DEFAULT)} '
            f'{mode or (755 if self.resolved.is_dir() else 644)} {quote(self.resolved.text)}', exc=True)
        return self

    def chown(self, group: Optional[Union[str, int]] = None, u: Optional[Union[str, int]] = None) -> Path:
        cmd(f'{sudo("chown", SUDO_DEFAULT)} {u or user.name}:{group or user.gname} '
            f'{quote(self.resolved.text)}',
            exc=True)
        return self

    @property
    def endseparator(self) -> str:
        """
        Add trailing separator at the end of path if does not exist.

        Returns:
            Str: path with separator at the end.
        """
        return self.text + os.sep

    def fd(self, *args, **kwargs):
        return os.open(self.text, *args, **kwargs)

    @property
    def find_packages(self) -> list:
        try:
            with self.cd:
                packages = find_packages()
        except FileNotFoundError:
            packages = list()
        return packages

    def find_up(self, file: PathIs = PathIs.FILE, name: Union[str, PathSuffix] = PathSuffix.ENV) -> FindUp:
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
                # noinspection PyArgumentList
                return FindUp()

    @classmethod
    def gittop(cls, path: Any = None) -> GitTop:
        url = cls.gittopurl(path)
        path = cls.gittoppath(path)
        return GitTop(str(url.path).rpartition('/')[2].split('.')[0] if url else path.name if path else None, url, path)

    # noinspection PyArgumentList
    @classmethod
    def gittoppath(cls, path: Any = None) -> Optional[Path]:
        rv = None
        if (path and ((path := cls(path).resolve()).exists() or (path := cls.cwd() / path).resolve().exists())) \
                or (path := cls.cwd().resolve()):
            with cls(path).cd:
                rv = cls(stdout[0]) if (stdout := cmd('git rev-parse --show-toplevel').stdout) else None
        return rv

    # noinspection PyArgumentList
    @classmethod
    def gittopurl(cls, path: Any = None) -> Optional[furl]:
        rv = None
        if (path and ((path := cls(path).resolve()).exists() or (path := cls.cwd() / path).resolve().exists())) \
                or (path := cls.cwd().resolve()):
            with cls(path).cd:
                rv = furl(stdout[0]) if (stdout := cmd('git config --get remote.origin.url').stdout) else None
        return rv

    def has(self, value: Iterable[str]) -> bool:
        return all([i in self for i in to_iter(value)])

    @staticmethod
    def home(name: str = None, file: bool = not FILE_DEFAULT) -> Path:
        """
        Returns home if not name or creates file or dir.

        Args:
            name: name.
            file: file.

        Returns:
            Path:
        """
        return Path(user.home)(name, file)

    # noinspection PyArgumentList
    @classmethod
    @cache
    def importer(cls, modname: str, s: list[FrameInfo] = None) -> tuple[Any, FrameType]:
        for frame in s or stack():
            if all([frame.function == FUNCTION_MODULE, frame.index == 0, 'PyCharm' not in frame.filename,
                    cls(frame.filename).suffix,
                    False if 'setup.py' in frame.filename and setuptools.__name__ in frame.frame.f_globals else True,
                    (c[0].startswith(f'from {modname} import') or
                     c[0].startswith(f'import {modname}'))
                    if (c := frame.code_context) else False, not cls(frame.filename).installedbin]):
                return cls(frame.filename), frame.frame

    def initpy(self) -> Path:
        return rv.path.resolved if (rv := self.find_up(name='__init__.py')) else None

    @property
    def installed(self) -> Path:
        """
        Relative path to site/user packages or scripts dir.

        Returns:
            Optional[Path]:
        """
        return self.installedpy or self.installedbin

    @property
    def installedbin(self) -> Path:
        """
        Relative path to scripts dir.

        Returns:
            Optional[PathLib]:
        """
        return self.resolve().relative(InstallScriptPath.path())

    @property
    def installedpy(self) -> Path:
        """
        Relative path to site/user packages.

        Returns:
            Optional[PathLib]:
        """
        for s in getsitepackages() + to_iter(USER_SITE) if USER_SITE else []:
            return self.relative(s)

    def _is_file(self) -> Optional[str]:
        p = self.resolved
        while True:
            if p.is_file():
                return p.text
            p = p.parent
            if p == Path('/'):
                return None

    def j2(self, dest: Any = None, stream: bool = True, variables: dict = None) -> Union[list, dict]:
        f = stack()[1]
        variables = variables if variables else f.frame.f_globals.copy() | f.frame.f_locals.copy()
        return [v(variables).dump(Path(dest / k).text) for k, v in self.templates(stream=stream).items()] \
            if dest and stream else {k: v(variables) for k, v in self.templates(stream=stream).items()}

    def mkdir(self, name: Optional[str] = None, group: Optional[Union[str, int]] = None,
              mode: Optional[Union[int, str]] = 755, su: bool = not SUDO_DEFAULT,
              u: Optional[Union[str, int]] = None) -> Path:
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
            cmd(f'{sudo("mkdir", su)} -p -m {mode or 755} {quote(p.text)}', exc=True)
        if file:
            raise NotADirectoryError(f'{file=} is file and not dir', f'{(self / (name or str())).resolved}')
        p.chown(group=group, u=u)
        return p

    @property
    def modname_from_file(self) -> Optional[str]:
        if self.is_dir():
            return getmodulename(self.text)

    @property
    def module_from_file(self) -> Optional[ModuleType]:
        if self.is_file() and self.text != __file__:
            with suppress(ModuleNotFoundError):
                spec = spec_from_file_location(self.name, self.text)
                module = module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

    @property
    def parent_if_file(self) -> Path:
        return self.parent if self.is_file() and self.exists() else self

    @property
    def pwd(self) -> Path:
        return self.cwd().resolved

    def relative(self, p: Any) -> Path:
        p = Path(p).resolved
        return self.relative_to(p) if self.resolved.is_relative_to(p) else None

    @property
    def resolved(self) -> Path:
        return self.resolve()

    def rm(self, missing_ok=True):
        """
        Delete a folder/file (even if the folder is not empty)

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

    def scan(self, option: PathOption = PathOption.FILES,
             output: PathOutput = PathOutput.BOX, suffix: PathSuffix = PathSuffix.NO,
             level: bool = False, hidden: bool = False, frozen: bool = False) -> Union[Box, dict, list]:
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
    def stemfull(self) -> Path:
        # noinspection PyArgumentList
        return type(self)(self.text.removesuffix(self.suffix))

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

    @staticmethod
    def sys() -> Path:
        return Path(sys.argv[0]).resolved

    def templates(self, stream: bool = True) -> dict[str, Union[Template.stream, Template.render]]:
        """
        Iter dir for templates and create dict with name and dump func

        Returns:
            dict:
        """
        if self.name != 'templates':
            # noinspection PyMethodFirstArgAssignment
            self /= 'templates'
        if self.is_dir():
            return {i.stem: getattr(Template(Path(i).read_text(), autoescape=True),
                                    'stream' if stream else 'render') for i in self.glob(f'*{PathSuffix.J2.dot}')}
        return dict()

    @property
    def text(self) -> str:
        return str(self)

    @classmethod
    @contextmanager
    def tmp(cls) -> Path:
        cwd = cls.cwd()
        tmp = TemporaryDirectory()
        with tmp as cd:
            try:
                # noinspection PyArgumentList
                yield cls(cd)
            finally:
                cwd.chdir()

    def touch(self, name: Optional[str] = None, group: Optional[Union[str, int]] = None,
              mode: Optional[Union[int, str]] = 644, su: bool = not SUDO_DEFAULT,
              u: Optional[Union[str, int]] = None) -> Path:
        """
        Add file, touch and return post_init Path.

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
            cmd(f'{sudo("touch", su)} {quote(p.text)}', exc=True)
        if file:
            raise NotADirectoryError(f'{file=} is file and not dir', f'{(self / (name or str())).resolved}')
        p.chmod(mode=mode)
        p.chown(group=group, u=u)
        return p

    @property
    def str(self) -> str:
        return self.text


PathLike.register(Path)
PathUnion = Union[Path, pathlib.Path, str]
