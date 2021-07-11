#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
Bash Utils Module.

Import always logger from loguru to have it in globals when using getlog() or trace().
    - logger is unconfigured.
    - log is configured logger.


>>> # noinspection PyUnresolvedReferences
>>> from loguru import logger
>>> from rc import Action
>>> from rc import Env
>>> from rc import getlog
>>> from rc import LogCall
>>> from rc import trace
>>>
>>> # noinspection PyShadowingNames
>>> env = Env()  # env.top.logger unconfigured imported logger from globals/locals.
>>> # noinspection PyShadowingNames
>>> log = getlog()  # `l` or e.top.logger or v.get('logger', None) or unconfigured `rc.logger`.
>>>
>>> @trace(start=LogCall.ERROR)
>>> def test_log():
...     log.info('test')
...     test_log.action = Action.RELEASED
"""
__all__ = (
    'subprocess',
    'gitpython',
    'typer',

    'ALL',
    'AnyPath',
    'clirunner',
    'cliinvoke',
    'FILE',
    'conf',
    'Configuration',
    'ExceptionUnion',
    'FILE',
    'GITHUB_API_URL',
    'GitUser',
    'IPv',
    'LOG',
    'LogCall',
    'NEWLINE',
    'TMP',

    'App',

    'Pis',
    'Pname',

    'copylogger',
    'elementadd',
    'findup',
    'getlog',
    'getstdout',
    'is_giturl',
    'jlogload',
    'noexc',
    'parent',
    'TMPDIR',
    'shell',
    'strip',
    'top',
    'trace',

    'Env',
    'env',
    'GitHub',
    'Command',
    'Distro',
    'Install',
    'Version',
)

import enum
import inspect
import io
import itertools
import json
import signal
import subprocess as subprocess
import sys
import venv
from collections import ChainMap
from collections import namedtuple
from contextlib import redirect_stdout
from copy import deepcopy
from functools import cache
from functools import partial
from functools import wraps
from ipaddress import ip_address
from ipaddress import IPv4Address
from ipaddress import IPv6Address
from os import getcwd
from os import PathLike
from pathlib import Path
from pathlib import PurePath
from tempfile import gettempprefix
from typing import cast
from typing import Iterable
from typing import Type
from typing import Union

import click
import environs
import envtoml
import git as gitpython
import marshmallow
import marshmallow.validate
import requests
import rich.box
import setuptools.command.install
import toml
import typer as typer
import typer.testing
from build.__main__ import main as build_main
from furl import furl
from git import TagReference
from icecream import ic
from loguru import logger
from loguru._defaults import LOGURU_FORMAT
from pip._internal.network.session import PipSession
# noinspection PyCompatibility
from pip._internal.req import parse_requirements
from psutil import MACOS
from rich.table import Table
from semver import VersionInfo
from setuptools.config import read_configuration
from strip_ansi import strip_ansi
from typer.main import get_click_type as get_type

Action = enum.Enum('Action', {__i: enum.auto() for __i in ('ACQUIRED', 'CANCELLED', 'CONSUMED', 'ERROR', 'FINISHED',
                                                           'LOCKED', 'NONE', 'PASS', 'PRODUCED', 'QUEUED', 'RELEASED',
                                                           'START', 'WAITING')})
ALL = 'all'
AnyPath = Union[str, bytes, PathLike, Path]
cachemodule = {}
"""cls: Class Methods/Properties using trace decorator.\n
log: Logger instance for function in trace decorator.\n
qualname: Methods/Properties names in trace decorator."""
clirunner = typer.testing.CliRunner(mix_stderr=False)
cliinvoke = clirunner.invoke
FILE = Path(__file__)
conf = envtoml.load(FILE.parent / '.toml')
Configuration = namedtuple('Configuration', 'pyproject_toml setup_cfg setup_py')
ExceptionUnion = Union[tuple[Type[Exception]], Type[Exception]]
GITHUB_API_URL = furl('https://api.github.com')
GitUser = namedtuple('GitUser', 'blog id https key login name org pip repos ssh token url')
IPv = Union[IPv4Address, IPv6Address]
LOG = Path.home() / 'log'
LogCall = enum.Enum('LogCall', {__i: getattr(logger.__class__, __i.lower()) for __i in conf['log']['number']})

NEWLINE = '\n'
Top = namedtuple('Top', 'exists file git_dir init_py installed jlogfile jlogload logger logfile name path prefix '
                        'pyproject_toml rlogfile root setup_cfg setup_py tmp venv work')
"""Attributes:\n
exists: Found. File exists?.\n
file: Found or Provided.\n
git_dir: Found.\n
init_py: Found.\n
installed: Found.\n
jlogfile: Generated (not touched).\n
jlogload: Generated. partial of `rc.jlogload()` with jlogfile (kwarg: chainmap).\n
logger: Found.\n
logfile: Generated (not touched).\n
name: Found (stem from path).\n
path: Found (path=file if __init__.py not found, or init_py.parent if found).\n
prefix: Found (from name).\n
pyproject_toml: Found.\n
rlogfile: Generated (not touched).\n
root: Found (git, pyproject, setup cfg/py, *-packages, pyvenv.cfg or python3. Not found: path.parent if path.is_file).\n
setup_cfg: Found.\n
setup_py: Found.\n
tmp: Generated (not made).\n
venv: Generated (not made).\n
work: Generated (not made).\n
"""
TMP = Path('/') / gettempprefix()


# <editor-fold desc="Patches">
def _callprocesserror(self):
    if self.returncode and self.returncode < 0:
        try:
            return "Command '%s' died with %r." % (
                self.cmd, signal.Signals(-self.returncode))
        except ValueError:
            return "Command '%s' died with unknown signal %d." % (
                self.cmd, -self.returncode)
    else:
        return f'Command={repr(self.cmd)}, returncode={repr(self.returncode)}, stderr={repr(self.stderr)}, ' \
               f'stdout={repr(self.stdout)}'


def _get_click_type(*, annotation, parameter_info):
    """Click Patch for Enum."""
    try:
        return get_type(annotation=annotation, parameter_info=parameter_info)
    except RuntimeError:
        if isinstance(annotation, type) and hasattr(annotation, '_value_') and isinstance(annotation, Iterable):
            return click.Choice([item._value_ for item in annotation], case_sensitive=parameter_info.case_sensitive, )
        raise RuntimeError(f"Type not yet supported: {annotation}")  # pragma no cover


class App(typer.Typer):
    def __init__(self, *args, **kwargs):
        kwargs = conf['app'] | kwargs
        super().__init__(*args, **kwargs)


subprocess.CalledProcessError.__str__ = _callprocesserror
typer.main.get_click_type = _get_click_type

app = App()
# </editor-fold>


# <editor-fold desc="Path Enums">
class _Pis(enum.Enum):
    def _generate_next_value_(self, *args):
        return getattr(Path, self.lower())


class Pis(_Pis):
    """Path Is Dir or File Enum Class."""
    EXISTS = enum.auto()
    IS_DIR = enum.auto()
    IS_FILE = enum.auto()


class _Pname(enum.Enum):
    def _generate_next_value_(self, *args):
        return Path((f'__{self.rstrip("_")}__' if self.endswith('_') else '' if self == 'NONE' else self).lower())


class Pname(_Pname):
    """Path Suffix Enum Class."""
    NONE = enum.auto()  # ''
    BASH = enum.auto()
    CFG = enum.auto()
    ENV = enum.auto()
    GIT = enum.auto()
    GITCONFIG = enum.auto()
    INI = enum.auto()
    INIT = enum.auto()
    INIT_ = enum.auto()
    J2 = enum.auto()
    JINJA2 = enum.auto()
    JSON = enum.auto()
    LOG = enum.auto()
    MONGO = enum.auto()
    OUT = enum.auto()
    PY = enum.auto()
    PYI = enum.auto()
    PYPROJECT = enum.auto()
    REQUIREMENTS = enum.auto()
    RLOG = enum.auto()
    SCRIPTS = enum.auto()
    SETUP = enum.auto()
    SH = enum.auto()
    SHELVE = enum.auto()
    SSH = enum.auto()
    TEMPLATES = enum.auto()
    TEST = enum.auto()
    TOML = enum.auto()
    TXT = enum.auto()
    YAML = enum.auto()
    YML = enum.auto()
    def bash(self, name=None): return Path((name or self.name) + Pname.BASH.dot.name)
    def cfg(self, name=None): return Path((name or self.name) + Pname.CFG.dot.name)
    @property
    def dot(self): return Path('.' + self.value.name)
    def env(self, name=None): return Path((name or self.name) + Pname.ENV.dot.name)
    def git(self, name=None): return Path((name or self.name) + Pname.GIT.dot.name)
    def j2(self, name=None): return Path((name or self.name) + Pname.J2.dot.name)
    def json(self, name=None): return Path((name or self.name) + Pname.JSON.dot.name)
    def log(self, name=None): return Path((name or self.name) + Pname.LOG.dot.name)
    @property
    def name(self): return self.value.name
    def py(self, name=None): return Path((name or self.name) + Pname.PY.dot.name)
    def pyi(self, name=None): return Path((name or self.name) + Pname.PYI.dot.name)
    def rlog(self, name=None): return Path((name or self.name) + Pname.RLOG.dot.name)
    def sh(self, name=None): return Path((name or self.name) + Pname.SH.dot.name)
    def test(self, name=None): return Path((name or self.name) + Pname.TEST.dot.name)
    def toml(self, name=None): return Path((name or self.name) + Pname.TOML.dot.name)
    def txt(self, name=None): return Path((name or self.name) + Pname.TXT.dot.name)
    def yml(self, name=None): return Path((name or self.name) + Pname.YML.dot.name)
    def yaml(self, name=None): return Path((name or self.name) + Pname.YAML.dot.name)


# </editor-fold>


# <editor-fold desc="Functions">
def copylogger(l, add=False):
    """
    Deep Copy Logger

    Args:
        l: logger.
        add: copy handlers in new copy.

    Returns:
        logger.
    """
    values = {i: getattr(l, i) for i in ('cached', 'configured', 'deepcopied', 'top_name', ) if hasattr(l, i)}
    handlers = l._core.handlers.copy()
    l._core.handlers = {}
    # for i in l._core.handlers:
    #     l.remove(i)
    new = deepcopy(l)
    l._core.min_level = min(i.levelno for i in handlers.values())
    l._core.handlers = handlers
    for k, v in values.items():
        setattr(new, k, v)
    new.deepcopied = True
    new._core.handlers_count, new._core.min_level = itertools.count(), float('inf')
    if add:
        new._core.handlers_count = itertools.count()
        new._core.min_level = l._core.min_level
        new._core.handlers = {}
        for k, v in handlers.items():
            try:
                c = deepcopy(v)
                i = next(new._core.handlers_count)
                new._core.handlers |= {i: c}
            except TypeError as exception:
                if "cannot pickle '_io.TextIOWrapper'" in repr(exception):
                    if v._name == '<stderr>':
                        new.add(sys.stderr)
                else:
                    i = next(new._core.handlers_count)
                    new._core.handlers |= {i: v}
    return new


def elementadd(name, closing=False):
    """
    Converts to HTML element.

    >>> from rc import elementadd
    >>>
    >>> assert elementadd('light-black') == '<light-black>'
    >>> assert elementadd('light-black', closing=True) == '</light-black>'
    >>> assert elementadd(('green', 'bold',)) == '<green><bold>'
    >>> assert elementadd(('green', 'bold',), closing=True) == '</green></bold>'

    Args:
        name: text or iterable text.
        closing: True if closing/end, False if opening/start.

    Returns:
        Str
    """
    return ''.join(f'<{"/" if closing else ""}{i}>' for i in ((name,) if isinstance(name, str) else name))


def findup(path=None, kind=Pis.IS_FILE, name=Pname.NONE.env, uppermost=False):
    """
    Find up if name exists or is file or directory.

    >>> import email.mime.application
    >>> from os import chdir
    >>> from pathlib import Path
    >>> import rc
    >>> from rc import findup, Pname, parent
    >>>
    >>> chdir(parent(rc.__file__))
    >>> file = Path(email.mime.application.__file__)
    >>>
    >>> pyproject_toml = findup(rc.__file__, name=Pname.PYPROJECT.toml)
    >>> setup_cfg = findup(rc.__file__, name=Pname.SETUP.cfg)
    >>> setup_py = findup(rc.__file__, name=Pname.SETUP.py)
    >>> assert any([pyproject_toml, setup_cfg, setup_py])
    >>>
    >>> assert findup(kind=Pis.EXISTS, name=Pname.INIT_.py) == Path(rc.__file__)
    >>> assert findup(kind=Pis.IS_DIR, name=rc.__name__) == Path(rc.__name__).parent.resolve()
    >>>
    >>> assert findup(file, kind=Pis.EXISTS, name=Pname.INIT_.py) == file.parent / Pname.INIT_.py()
    >>> assert findup(file, name=Pname.INIT_.py) == file.parent / Pname.INIT_.py()
    >>> assert findup(file, name=Pname.INIT_.py, uppermost=True) == file.parent.parent / Pname.INIT_.py()

    Args:
        path: CWD if None or Path.
        kind: Exists, file or directory.
        name: File or directory name.
        uppermost: Find uppermost found if True (return the latest found if more than one) or first if False.

    Returns:
        Path if found.
    """
    name = name if isinstance(name, str) else name.name if isinstance(name, Path) else name() \
        if callable(name) else name.value
    start = parent(path or getcwd())
    latest = None
    while True:
        if kind.value(find := start / name):
            if not uppermost:
                return find
            latest = find
        if (start := start.parent) == Path('/'):
            return latest


def getlog(e=None, l=None, extra_add=(), std_add=(), std_default=False, copy=False,
           *,
           trace_='light-black', debug='light-blue', info='light-magenta', success=('green', 'bold',),
           warning=('yellow', 'bold',), error=('red', 'bold',), critical=('RED', 'bold',),
           time='light-black', level='level', name='light-black', function='cyan', line='light-black',
           extra='level', message='level', colon='light-white', vertical='light-white'):
    """
    Get Logger and Configure.

    Default std format only has function record field. Use std_add=('name',) to add more record fields.

    Examples:
        >>> from loguru import logger
        >>> from rc import getlog
        >>>
        >>> # noinspection PyShadowingNames
        >>> env = Env()
        >>>
        >>> # default format for std ('function') + 'name'
        >>> # noinspection PyShadowingNames
        >>> log = getlog(e=env, l=logger, std_add=('name',))
        >>>
        >>> # default loguru format for std
        >>> # noinspection PyShadowingNames,PyRedeclaration
        >>> log = getlog(std_default=True)

    Args:
        e: :class:`rc.Env` instance for log environment values. Order: `e` or v.get('env', None) or Env(index=3)
            (default: None).
        l: Unconfigured logger to use. Order: env.top.logger, globals/locals or unconfigured`rc.logger`
            (default: None).
        extra_add: extra field names to add to std. If add extra added to file (default: ()).
        std_add: record field names to add to std (default: ()).
        std_default: Use default loguru format for std.
        copy: Deepcopy logger before configure. It always deepcopy first time for package/top (default: ()).
        trace_: trace level color.
        debug: debug level color.
        info: info level color.
        success: success level color.
        warning: warning level color.
        error: error level color.
        critical: critical level color.
        time: time record key color.
        level: level record key color.
        name: name  record key color.
        function: function record key color.
        line: line record key color.
        extra: extra record key color.
        message: message record key color.
        colon: colon output separator color.
        vertical: vertical output separator color.

    Returns:
        logger.
    """
    # TODO: añadir module a format, igual añadir el top como extra.
    #   si añado el module mirar si parche o no.
    std_width = 15
    if getlog not in cachemodule:
        cachemodule[getlog] = {}
    if not e or not l:
        frame = inspect.stack()[1].frame
        v = frame.f_globals | frame.f_locals
        e = e or v.get('env', None) or Env(index=3)
        l = l or e.top.logger or v.get('logger', None) or logger
    in_cache = e.top.name in cachemodule[getlog]
    cache_it = False
    # First for package/top will be deepcopied.
    if copy or (cache_it := not copy and not in_cache):
        new = copylogger(l)
        if cache_it:
            cachemodule[getlog][e.top.name] = new.cached = True
    else:
        new = l
        new._core.handlers = {}
        new._core.deepcopied, new._core.handlers_count, new._core.min_level = False, itertools.count(), float('inf')
    new.configured, new.top_name = True, e.top.name

    extra_add = (extra_add,) if isinstance(extra_add, str) else extra_add
    std_add = (std_add, 'function', ) if isinstance(std_add, str) else std_add + ('function', )
    message = elementadd(message) + '{message}' + elementadd(message, closing=True)
    colon = elementadd(colon) + ':' + elementadd(colon, closing=True)
    vertical = ' ' + elementadd(vertical) + '|' + elementadd(vertical, closing=True) + ' '

    f = elementadd(time) + '{time:MM-DD HH:mm:ss}' + elementadd(time, closing=True) + vertical + \
        elementadd(level) + '{level: <8}' + elementadd(level, closing=True) + vertical + \
        elementadd(name) + '{name}' + elementadd(name, closing=True) + colon + \
        elementadd(function) + '{function}' + elementadd(function, closing=True) + colon + \
        elementadd(line) + '{line}' + elementadd(line, closing=True) + vertical
    f = f + ((elementadd(extra) + '{extra}' + elementadd(extra, closing=True) + vertical) if extra_add else '') + \
        message

    std = LOGURU_FORMAT
    if not std_default:
        std = (colon.join(elementadd(extra) + '{extra[' + i + ']}' + elementadd(extra, closing=True)
                          for i in extra_add) + vertical) if extra_add else ''
        for field in std_add:
            kw = locals().get(field, level)
            _map = {
                'function': ': <15',
                'level': ': <8',
                'line': ': <5',
                'name': ': <10',
            }
            _field = '{' + field + _map.get(field, _map['function']) + '}'
            std = std + elementadd(kw) + _field + elementadd(kw, closing=True) + vertical
        std += message

    new.configure(
        handlers=[
            dict(sink=sys.stderr, level=e.level_std, format=std,
                 backtrace=True, diagnose=True),
            dict(sink=e.top.logfile, level=e.level_file, format=f,
                 backtrace=True, diagnose=True,
                 retention='10 days', rotation='500 MB', compression='zip', enqueue=True,
                 colorize=True),
            dict(sink=e.top.jlogfile, level=e.level_file,
                 backtrace=True, diagnose=True,
                 retention='5 days', rotation='500 MB', compression='zip', enqueue=True,
                 serialize=True),
        ],
        levels=[
            dict(name='TRACE', icon='✏️', color=elementadd(trace_)),
            dict(name='DEBUG', icon='🐞', color=elementadd(debug)),
            dict(name='INFO', icon='ℹ️', color=elementadd(info)),
            dict(name='SUCCESS', icon='✔️', color=elementadd(success)),
            dict(name='WARNING', icon='⚠️', color=elementadd(warning)),
            dict(name='ERROR', icon='❌', color=elementadd(error)),
            dict(name='CRITICAL', icon='☠️', color=elementadd(critical)),
        ],
    )
    return new


def getstdout(func, *args, ansi=False, new=True, **kwargs):
    """
    Redirect stdout for func output and remove ansi and/or new line.

    Args:
        func: callable.
        *args: args to callable.
        ansi: strip ansi.
        new: strip new line.
        **kwargs: kwargs to callable.

    Returns:
        str | Iterable[str, str]:
    """
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        func(*args, **kwargs)
    return strip(buffer.getvalue(), ansi=ansi, new=new) if ansi or new else buffer.getvalue()


def is_giturl(remote):
    """
    Is Valid Git Remote Url.

    >>> from rc import is_giturl
    >>>
    >>> assert is_giturl('https://github.com/python/cpython') is True
    >>> assert is_giturl('https://github.com/python/ipython') is False

    Args:
        remote: remote url.

    Returns:
        Bool.
    """
    return not subprocess.run(f'git ls-remote {str(remote)} CHECK_GIT_REMOTE_URL_REACHABILITY', shell=True).returncode


def jlogload(file, chain=True):
    """
    Load Repo Json Log Path.

    Args:
        file: file.
        chain: return ChainMap.

    Returns:
        Load Repo Json Log Path.
    """
    rv = [json.loads(line)['record'] for line in reversed(file.read_text().splitlines())]
    if chain:
        return ChainMap(*rv)
    return rv


def noexc(func, *args, default_=None, exc_=Exception, **kwargs):
    """
    Execute function suppressing exceptions.

    Examples:
        >>> from rc import noexc
        >>>
        >>> assert noexc(dict(a=1).pop, 'b', default_=2, exc_=KeyError) == 2

    Args:
        func: callable.
        *args: args.
        default_: default value if exception is raised.
        exc_: exception or exceptions.
        **kwargs: kwargs.

    Returns:
        Any: Function return.
    """
    try:
        return func(*args, **kwargs)
    except exc_:
        return default_


def parent(path=FILE, home=False, tmp=False):
    """
    Parent if File or None if does not exist.

    Args:
        path: file or dir.
        home: create dir in home with dotted parent name and return.
        tmp: create dir in tmp with parent name and return.

    Returns:
        Path
    """
    def mk(p, dot=''): r = p / f'{dot}{rv.name}'; r.mkdir(parents=True, exist_ok=True); return r
    parent.mk = mk
    rv = path.parent if (path := Path(path)).is_file() else path if path.is_dir() else None
    if rv and home:
        rv = mk(Path.home(), '.')
    elif rv and tmp:
        rv = mk(TMP)
    return rv


TMPDIR = parent(tmp=True)


def shell(cmd, ansi=False, cwd=None, environment=None, exc=True, executable=None, stdout=True, sudo=None):
    """
    Execute Command in Shell Text and Strips End Line.

    >>> import rc
    >>> from rc import shell, parent
    >>>
    >>> assert shell('git rev-parse --show-toplevel', cwd=parent(rc.__file__))
    >>> remote = 'origin'
    >>> assert shell(f'git config --get remote.{remote}.url', cwd=parent(rc.__file__))
    >>> assert shell(f'git remote', cwd=parent(rc.__file__))

    Args:
        cmd: command.
        ansi: strip ansi.
        cwd: cwd.
        environment: environment.
        exc: check return code and raise exception.
        executable: executable.
        stdout: return stdout only.
        sudo: (default: None)
            - ``True``: adds sudo for everything but ``MACOS``.
            - ``False``: adds always.
            - ``None``: dor not add (skip).

    Raises:
        CalledProcessError.

    Returns:
        CompletedProcess or Stdout.
    """
    cmd = f'sudo {cmd}' if (sudo is False) or (sudo is True and not MACOS) else cmd
    rv = subprocess.run(cmd, capture_output=True, cwd=cwd, env=environment, executable=executable,
                        shell=True, text=True)
    out = str(rv.stdout).rstrip(NEWLINE)
    err = str(rv.stderr).rstrip(NEWLINE)
    if ansi:
        out = strip_ansi(out)
        err = strip_ansi(err)
    if exc and rv.returncode != 0:
        subprocess.CalledProcessError(rv.returncode, rv.args, rv.stdout, cast(str, rv.stderr.splitlines()))
    if stdout:
        return out if rv.returncode == 0 else None
    return subprocess.CompletedProcess(rv.args, rv.returncode, out, err)


def strip(data, ansi=False, new=True):
    """
    Strips ``\n`` And/Or Ansi from string or Iterable.

    Args:
        data: object or None for redirect stdout
        ansi: ansi (default: False)
        new: newline (default: True)

    Returns:
        Same type with NEWLINE removed.
    """

    def rv(x):
        if isinstance(x, str):
            x = x.removesuffix(NEWLINE) if new else x
            x = strip_ansi(x) if ansi else x
        if isinstance(x, bytes):
            x = x.removesuffix(b'\n') if new else x
        return x

    cls = type(data)
    if isinstance(data, str):
        return rv(data)
    return cls(rv(i) for i in data)


def top(data=2, cwd=False, imp_logger=False):
    """
    Get Top Level Package/Module Path.

    Args:
        data: stack index, stack, FrameInfo, FrameType or file name.
        cwd: Use cwd as start instead of data.
        imp_logger: import logger. First for package/top will be deepcopied..

    Raises:
        AttributeError: __file__ not found.
    """
    if cwd:
        file = Path.cwd()
    elif isinstance(data, int):
        file = inspect.stack()[data].filename
    elif isinstance(data, inspect.FrameInfo):
        file = data.filename
    elif isinstance(data, str) or isinstance(data, (Path, PurePath)):
        file = data
    else:
        file = getattr(data, '__file__', None)
    if file is None:
        raise AttributeError(f'File not found in: {data}')
    file = Path(file)
    exists = file.exists()
    name = None if exists else 'missing'
    if top not in cachemodule:
        cachemodule[top] = {}
    if file in cachemodule[top]:
        return cachemodule[top][file]

    _jlogload = git_dir = init_py = installed = jlogfile = path = pyproject_toml = setup_cfg = setup_py = None
    start = parent(file)
    root = Path(rv) if (rv := shell('git rev-parse --show-toplevel', cwd=start, exc=False)) else None
    v = root / venv.__name__ if root else None

    if exists:
        while True:
            if (rv := start / Pname.GIT.dot).is_dir():
                git_dir = rv
            if (rv := start / Pname.INIT_.py()).is_file():
                init_py, path = rv, start
            if (rv := start / Pname.PYPROJECT.toml()).is_file():
                pyproject_toml = rv
            if (rv := start / Pname.SETUP.cfg()).is_file():
                setup_cfg = rv
            if (rv := start / Pname.SETUP.py()).is_file():
                setup_py = rv
            if not cwd and any([start.name == 'dist-packages', start.name == 'site-packages',
                                start.name == Path(sys.executable).resolve().name, (start / 'pyvenv.cfg').is_file()]):
                installed, root = True, start
                break
            finish = root.parent if root else None
            if (start := start.parent) == (finish or Path('/')):
                break
        path = path or file
        name = path.stem
        if root is None:
            for i in (pyproject_toml, setup_cfg, setup_py):
                if i:
                    root = i.parent
                    break
            else:
                root = path if path.is_dir() else path.parent
    if imp_logger:
        from loguru import logger as _logger
    else:
        frame = inspect.stack()[1].frame
        v = frame.f_globals | frame.f_locals
        _logger = v.get('logger', None) or logger
    new = None
    # First for package/top will be deepcopied.
    if name not in cachemodule[top]:
        new = copylogger(_logger, add=True)
        cachemodule[top][name] = new.cached = True
    __l = new or _logger
    __l.configured, __l.top_name = False, name
    jlogfile = LOG / Pname.NONE.json(name)
    cachemodule[top][file] = Top(exists=exists, file=file, git_dir=git_dir, init_py=init_py, installed=installed,
                                 jlogfile=jlogfile, jlogload=partial(jlogload, jlogfile), logger=__l,
                                 logfile=LOG / Pname.NONE.log(name), name=name,
                                 path=path, prefix=f'{name.upper()}_', pyproject_toml=pyproject_toml,
                                 rlogfile=LOG / Pname.NONE.rlog(name), root=root, setup_cfg=setup_cfg,
                                 setup_py=setup_py, tmp=TMP / name, venv=v, work=Path.home() / f'.{name}')
    return cachemodule[top][file]


def trace(l=None, add_end=(), add_start=(), patch=True, qual=False, reraise=True,
          *,
          acquired=LogCall.TRACE, finish=LogCall.INFO, pass_=LogCall.DEBUG,
          produced=LogCall.DEBUG,
          released=LogCall.TRACE, start=None, waiting=LogCall.TRACE):
    """
    Trace Logging Decorator.

    Uses ``action`` attribute in func or self/args[0].

    Examples:

    #
    # To use default loguru format for stderr.
    #
    >>> @trace(l=getlog(std_default=True, copy=True), add_end=('test', ), released=LogCall.INFO, start=LogCall.ERROR)
    ... def log_trace_loguru_default():
    ...     name = log_trace_loguru_default.__name__
    ...     log.info(name)
    ...     log_trace_loguru_default.test = name
    ...     log_trace_loguru_default.action = Action.RELEASED
    >>> log_trace_loguru_default()  # __init__ since file does not exists in doctest and not patch.
    >>> # 2021-07-10 15:44:56.840 | ERROR    | __init__:log_trace_loguru_default:68 - START

    #
    # To add 'name' record field to default stderr which only has 'function'.
    #
    >>> @trace(l=getlog(std_add=('name',), copy=True), add_end=('test', ), released=LogCall.SUCCESS, start=LogCall.ERROR)
    ... def log_trace_std_add_name():
    ...     name = log_trace_std_add_name.__name__
    ...     log.info(name)
    ...     log_trace_std_add_name.test = name
    ...     log_trace_std_add_name.action = Action.RELEASED
    >>> log_trace_std_add_name()  # __init__ since file does not exists in doctest and not patch.
    >>> # __init__        | log_trace_std_add_name | START
    >>> # __init__        | log_trace_std_add_name | RELEASED: test='log_trace_std_add_name'

    #
    # To use globals `log` which add 'name' and 'line' record fields to default stderr which only has 'function'.
    #
    >>> # noinspection PyShadowingNames
    >>> env = Env()
    >>> # noinspection PyShadowingNames
    >>> log = getlog(std_add=('name', 'line', ))
    >>>
    >>> @trace(add_end=('test', ), add_start=True, released=LogCall.SUCCESS, start=LogCall.WARNING)
    ... def log_trace_globals(arg1, kwarg_1='kwarg_1'):
    ...     name = log_trace_globals.__name__
    ...     log.info(name)
    ...     log_trace_globals.test = name
    ...     log_trace_globals.action = Action.RELEASED
    >>> log_trace_globals(1)
    >>> # __init__   | 966   | log_trace_globals | START: arg1=1, kwarg_1='kwarg_1'
    >>> # __init__   | 964   | log_trace_globals | RELEASED: test='log_trace_globals'

    Args:
        l: log to use. Order: configured globals/locals `log`, unconfigured globals/locals `logger`,
            or rc.logger  (default: None).
        add_end: add attributes from func or self/args[0] to end message.
        add_start: add attributes from func args, kwargs to start message. If `True` adds all but `cls` and `self`.
            It will overwrite `start` if it is `None`.
        patch: patch name, lineno and func_name.
        qual: Use qualname instead of name patching function (default: False).
        reraise: Re-raise exception (default: True).
        acquired: Level method name to use when ACQUIRED action (default: LogCall.TRACE).
        finish: Level method name to use when FINISHED action (default: LogCall.INFO).
        pass_: Level method name to use when PASS action (default: LogCall.DEBUG).
        produced: Level method name to use when PRODUCED action (default: LogCall.DEBUG).
        released: Level method name to use when RELEASED action (default: LogCall.TRACE).
        start: Level method name to use if logging start of function (default: None or LogCall.DEBUG if add_start).
        waiting: Level method name to use when WAITING action (default: LogCall.TRACE).

    Returns:
        decorating:
    """
    start_default = LogCall.DEBUG

    def decorating(func):
        if trace not in cachemodule:
            cachemodule[trace] = dict(cls={}, func_name={}, log=dict(arg={}, stack={}), patch={}, module_function={},
                                      start={})

        if func not in cachemodule[trace]['func_name']:
            qualname = func.__qualname__
            cls, _, func_name = qualname.partition('.')
            cachemodule[trace]['module_function'][func] = not bool(func_name)
            func_name = func_name or cls
            if cls not in cachemodule[trace]['cls']:
                cachemodule[trace]['cls'][cls] = []
            cachemodule[trace]['cls'][cls].append(func_name)
            __l = l
            cachemodule[trace]['patch'][func] = {}
            if not l or patch:
                frame = inspect.stack()[1]
                _path = Path(frame.filename)
                if _path.exists() and patch:
                    cachemodule[trace]['patch'][func] = dict(
                        line=frame.lineno, name=_path.parent.name if (_rv := _path.stem) == '__init__' else _rv
                    )
                v = frame.frame.f_globals | frame.frame.f_locals
                __l = l or v.get('log', None) or v.get('logger', None) or logger
            cachemodule[trace]['log']['stack'][func] = __l
            cachemodule[trace]['func_name'][func] = qualname if qual else func_name
            cachemodule[trace]['start'][func] = start_default if start is None and add_start else start
        func_name = cachemodule[trace]['func_name'][func]
        module_function = cachemodule[trace]['module_function'][func]
        trace_start = cachemodule[trace]['start'][func]

        is_async_generator = inspect.isasyncgenfunction(func)
        is_async = inspect.iscoroutinefunction(func)
        is_generator = inspect.isgeneratorfunction(func)

        def _action(*args, _func=None, _end=False, **kwargs):
            self = args[0] if len(args) >= 1 else None
            _func = _func or func
            action = getattr(self, 'action', None) or getattr(_func, 'action', None)
            if _end:
                _add = {i: getattr(obj, i) for obj in (self, _func, )
                        for i in add_end if hasattr(obj, i)} if add_end else {}
            else:
                count = itertools.count()
                _add = {_k: args[next(count)] if _v.default == inspect._empty else kwargs.get(_k, _v.default)
                        for _k, _v in inspect.signature(func).parameters.items()
                        if (add_start is True or _k in add_start) and _k not in ('cls', 'self',)} if add_start else {}
            return action, _add

        def _log(*args):
            if func not in cachemodule[trace]['log']['arg']:
                value = getattr(args[0], 'log', None) or getattr(args[0], 'logger', None) if len(args) >= 1 else None
                cachemodule[trace]['log']['arg'][func] = value or cachemodule[trace]['log']['stack'][func]
            return cachemodule[trace]['log']['arg'][func]

        def _module_func():
            _func = func
            if module_function:  # get func from globals/locals for attributes added (action, add).
                _f = inspect.stack()[2].frame
                _v = _f.f_globals | _f.f_locals
                _func = _v.get(func_name, None)
                return _func

        def _patch(action=None, _add=None, end=False, _l=None):
            _l = _l.patch(lambda record: record.update(function=func_name, **cachemodule[trace]['patch'][func]))
            _add = ': ' + ', '.join(f'{_k}={repr(_v)}' for _k, _v in _add.items()) if _add else ''
            if end:
                method = finish
                if action and action == Action.PASS:
                    method = pass_
                elif action and action == Action.ACQUIRED:
                    method = acquired
                elif action and action == Action.ERROR:
                    method = LogCall.ERROR
                elif action and action == Action.PRODUCED:
                    method = produced
                elif action and action == Action.RELEASED:
                    method = released
                elif action and action == Action.WAITING:
                    method = waiting
                # noinspection PyCallingNonCallable
                method(_l, (action or Action.FINISHED).name + _add)
            elif trace_start:
                trace_start(_l, (action or Action.START).name + _add)

        if is_async_generator:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                action, _l = _action(*args, **kwargs), _log(*args)
                with _l.catch(reraise=reraise):
                    if start:
                        _patch(action=action[0], _add=action[1], _l=_l)
                    try:
                        async for result in func(*args, **kwargs):
                            yield result
                    finally:
                        # Module Function attrs only for end.
                        action, _l = _action(*args, _func=_module_func(), _end=True), _log(*args)
                        _patch(action=action[0], _add=action[1], end=True, _l=_l)
        elif is_async:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                action, _l = _action(*args, **kwargs), _log(*args)
                with _l.catch(reraise=reraise):
                    if start:
                        _patch(action=action[0], _add=action[1], _l=_l)
                    rv = await func(*args, **kwargs)
                    # Module Function attrs only for end.
                    action, _l = _action(*args, _func=_module_func(), _end=True), _log(*args)
                    _patch(action=action[0], _add=action[1], end=True, _l=_l)
                    return rv
        elif is_generator:
            @wraps(func)
            def wrapper(*args, **kwargs):
                action, _l = _action(*args, **kwargs), _log(*args)
                with _l.catch(reraise=reraise):
                    if start:
                        _patch(action=action[0], _add=action[1], _l=_l)
                    if is_generator:
                        try:
                            yield from func(*args, **kwargs)
                        finally:
                            _patch(action=action[0], _add=action[1], end=True, _l=_l)
                    else:
                        rv = func(*args, **kwargs)
                        # Module Function attrs only for end.
                        action, _l = _action(*args, _func=_module_func(), _end=True), _log(*args)
                        _patch(action=action[0], _add=action[1], end=True, _l=_l)
                        return rv
        else:
            @wraps(func)
            def wrapper(*args, **kwargs):
                _func = func
                action, _l = _action(*args, **kwargs), _log(*args)
                with _l.catch(reraise=reraise):
                    if start:
                        _patch(action=action[0], _add=action[1], _l=_l)
                rv = func(*args, **kwargs)
                # Module Function attrs only for end.
                action, _l = _action(*args, _func=_module_func(), _end=True), _log(*args)
                _patch(action=action[0], _add=action[1], end=True, _l=_l)
                return rv
        return wrapper
    return decorating


# </editor-fold>


class Env(environs.Env):
    """Env Class."""
    def __init__(self, *, eager=True, expand_vars=False, ignore_case=True, index=2,
                 override=True, path=None, prefix=None, use_prefix=False):
        """

        Args:
            eager:
            expand_vars:
            ignore_case: ignore case.
            index: stack index, stack, FrameInfo, FrameType or file name for top().
            override:
            path:
            prefix:
            use_prefix: use prefix.
        """
        super().__init__(eager=eager, expand_vars=expand_vars)
        if path:
            path = Path(path)
            if path.is_file():
                name = path.name
                path = path.parent
            elif path.is_dir():
                path = path
                name = Pname.ENV.dot
            else:
                path = Path.cwd()
                name = str(Path)
        else:
            path = Path.cwd()
            name = Pname.ENV.dot
        self.path = findup(path=path, name=name)
        super().read_env(str(self.path), override=override)
        self.ignore_case = ignore_case
        self.top = top(data=index)
        self.prefix = prefix or self.top.prefix
        self.use_prefix = use_prefix

    def __call__(self, name, *args, use_prefix=False, **kwargs):
        """
        Parse if value is level or integer as log level name.

        >>> from os import environ
        >>> from rc import Env
        >>>
        >>> e = Env()
        >>>
        >>> var = 'VAR'
        >>>
        >>> environ[var] = '24'
        >>> assert e.level(var, default=None) is None
        >>>
        >>> environ[var] = '30'
        >>> assert e.level(var) == 'WARNING'
        >>>
        >>> environ[var] = 'deb'
        >>> assert e.level(var, default=None) is None
        >>>
        >>> environ[var] = 'debug'
        >>> assert e.level(var) == 'DEBUG'
        >>>
        >>> environ[var] = 'DEBUG'
        >>> assert e.level(var) == 'DEBUG'

        Args:
            value: value to parse.
            default: default.
            use_prefix: use prefix if prefix.
            **kwargs: default.
                    subcast type

        Returns:
            Any.
        """
        name = name.upper() if self.ignore_case else name
        if (use_prefix or self.use_prefix) and self.prefix:
            with self.prefixed(self.prefix):
                return super().__call__(name, *args, **kwargs)
        return super().__call__(name, *args, **kwargs)

    def __getstate__(self): return self.__dict__

    def __setstate__(self, state): self.__dict__ = state.copy()

    def ip(self, name, exc=False, **kwargs):
        """
        Parse as IP.

        >>> from os import environ
        >>> from pathlib import Path
        >>> from rc import Env
        >>>
        >>> e, host, var = Env(), 'google.com', 'URL'
        >>>
        >>> environ[var] = str(Path.home())
        >>> assert e.furl(var, default=None) is None
        >>>
        >>> environ[var] = host
        >>> assert e.furl(var).host == host
        >>>
        >>> environ[var] = 'https://google.com'
        >>> assert e.furl(var).host == host

        Args:
            name: var name.
            exc: exception if no default.
            **kwargs: default.

        Returns:
            IPv.
        """
        kw = kwargs.copy()
        found = False
        if exc and 'default' not in kwargs:
            found = True
            kw = kwargs.copy() | dict(default=None)
        if rv := noexc(self, name, **kw):
            return self.ip_parser(rv, exc, **kwargs)
        if exc and found:
            raise marshmallow.ValidationError(f'Not a valid ip: {name=}, {rv=}')

    @staticmethod
    def ip_parser(value, exc=False, **kwargs):
        """
        Parse IP.

        >>> from ipaddress import IPv4Address
        >>> from rc import Env
        >>>
        >>> assert Env.ip_parser('8.8', exc=False) is None
        >>> assert Env.ip_parser('8.8.8.8') == IPv4Address('8.8.8.8')

        Args:
            value: value to parse.
            exc: raise ValidationError if not default.
            **kwargs: default.

        Raises:
            ValidationError.

        Returns:
            IPv.
        """
        default = kwargs.pop('default', None)
        if rv := noexc(ip_address, value):
            return rv
        if default:
            return default
        if exc:
            raise marshmallow.ValidationError(f'Not a valid log level: {value}')

    def furl(self, name, exc=False, **kwargs):
        """
        Parse as Furl.

        >>> from ipaddress import IPv4Address
        >>> from os import environ
        >>> from rc import Env
        >>>
        >>> e, var = Env(), 'IP'
        >>>
        >>> environ[var] = '8.8'
        >>> assert e.ip(var, default=None) is None
        >>>
        >>> environ[var] = '8.8.8.8'
        >>> assert e.ip(var) == IPv4Address('8.8.8.8')

        Args:
            name: var name.
            exc: exception if no default.
            **kwargs: default.

        Returns:
            Furl.
        """
        kw = kwargs.copy()
        found = False
        if exc and 'default' not in kwargs:
            found = True
            kw = kwargs.copy() | dict(default=None)
        if rv := noexc(self, name, **kw):
            return self.furl_parser(rv, exc, **kwargs)
        if exc and found:
            raise marshmallow.ValidationError(f'Not a valid url: {name=}, {rv=}')

    @staticmethod
    def furl_parser(value, exc=False, **kwargs):
        """
        Parse IP.

        >>> from ipaddress import IPv4Address
        >>> from pathlib import Path
        >>> from rc import Env
        >>>
        >>> host = 'google.com'
        >>> assert Env.furl_parser(str(Path.home()), exc=False) is None
        >>> assert Env.furl_parser(host).host == host
        >>> assert Env.furl_parser('https://google.com').host == host

        Args:
            value: value to parse.
            exc: raise ValidationError if not default.
            **kwargs: default.

        Raises:
            ValidationError.

        Returns:
            IPv.
        """
        default = kwargs.pop('default', None)
        if rv := noexc(furl, value):
            if noexc(marshmallow.validate.URL(), rv.url):
                return rv
            if '.' in value and (rv := noexc(furl, scheme='https', host=value)):
                if noexc(marshmallow.validate.URL(), rv.url):
                    return rv
        if default:
            return default
        if exc:
            raise marshmallow.ValidationError(f'Not a valid url: {value}')

    def level(self, name, exc=False, use_prefix=True, **kwargs):
        """
        Log Level.

        >>> from os import environ
        >>> from rc import Env, conf
        >>>
        >>> e, var, prefix = Env(), 'level', 'TEST_'
        >>> VAR = (prefix + var).upper()
        >>>
        >>> environ[VAR] = '24'
        >>> assert e.level(prefix + var, None) is None
        >>>
        >>> environ[VAR] = '30'
        >>> assert e.level(prefix + var) == 'WARNING'
        >>>
        >>> environ[VAR] = 'deb'
        >>> assert e.level(prefix + var, None) is None
        >>>
        >>> environ[VAR] = 'debug'
        >>> assert e.level(prefix + var) == 'DEBUG'
        >>>
        >>> environ[VAR] = 'DEBUG'
        >>> assert e.level(prefix + var) == 'DEBUG'

        Args:
            name: var name.
            exc: exception if no default.
            use_prefix: use prefix.
            **kwargs: default: default.

        Returns:
            str.
        """
        kw = kwargs.copy()
        found = False
        if exc and 'default' not in kwargs:
            found = True
            kw = kwargs.copy() | dict(default=None)
        if rv := noexc(self, name, use_prefix=use_prefix, **kw):
            return self.level_parser(rv, exc, **kwargs)
        if exc and found:
            raise marshmallow.ValidationError(f'Not a valid log level: {name=}, {rv=}')

    @staticmethod
    def level_parser(value, exc=False, **kwargs):
        """
        Parse if value is level or integer as log level name.

        >>> from rc import Env
        >>>
        >>> assert Env.level_parser('24', exc=False) is None
        >>> assert Env.level_parser('30') == 'WARNING'
        >>> assert Env.level_parser('deb', exc=False) is None
        >>> assert Env.level_parser('debug') == 'DEBUG'
        >>> assert Env.level_parser('DEBUG') == 'DEBUG'

        Args:
            value: value to parse.
            exc: raise ValidationError if not default.
            **kwargs: default.

        Raises:
            ValidationError.

        Returns:
            Log Level.
        """
        levels = conf['log']['number']
        v = value.upper()
        default = kwargs.pop('default', None)
        if v in levels:
            return v
        elif integer := noexc(marshmallow.fields.Integer().deserialize, value):
            for k, v in levels.items():
                if v == integer:
                    return k
        if default:
            return default
        if exc:
            raise marshmallow.ValidationError(f'Not a valid log level: {value}')

    @property
    def level_file(self):
        """
        Log Level File Level.

        >>> from os import environ
        >>> from rc import Env, conf
        >>>
        >>> e, var, prefix = Env(), 'level_file', 'TEST_'
        >>> e.prefix = prefix
        >>> VAR = (prefix + var).upper()
        >>>
        >>> environ[VAR] = '24'
        >>> assert e.level_file == conf['log']['default'][var]
        >>>
        >>> environ[VAR] = '30'
        >>> assert e.level_file == 'WARNING'
        >>>
        >>> environ[VAR] = 'deb'
        >>> assert e.level_file == conf['log']['default'][var]
        >>>
        >>> environ[VAR] = 'debug'
        >>> assert e.level_file == 'DEBUG'
        >>>
        >>> environ[VAR] = 'DEBUG'
        >>> assert e.level_file == 'DEBUG'

        Returns:
            str.
        """
        name = self.__class__.level_file.fget.__name__
        return self.level(name, default=conf['log']['default'][name])

    @property
    def level_std(self):
        """
        Log Level Stderr Level.

        >>> from os import environ
        >>> from rc import Env, conf
        >>>
        >>> e, var, prefix = Env(), 'level_std', 'TEST_'
        >>> e.prefix = prefix
        >>> VAR = (prefix + var).upper()
        >>>
        >>> environ[VAR] = '24'
        >>> assert e.level_std == conf['log']['default'][var]
        >>>
        >>> environ[VAR] = '30'
        >>> assert e.level_std == 'WARNING'
        >>>
        >>> environ[VAR] = 'deb'
        >>> assert e.level_std == conf['log']['default'][var]
        >>>
        >>> environ[VAR] = 'debug'
        >>> assert e.level_std == 'DEBUG'
        >>>
        >>> environ[VAR] = 'DEBUG'
        >>> assert e.level_std == 'DEBUG'

        Returns:
            str.

        """
        name = self.__class__.level_std.fget.__name__
        return self.level(name, default=conf['log']['default'][name])

    def version(self, name, exc=False, **kwargs):
        """
        Parse as Version.

        >>> from os import environ
        >>> from pathlib import Path
        >>> from rc import Env, Version
        >>>
        >>> e, var = Env(), 'VERSION'
        >>>
        >>> environ[var] = 'name@test'
        >>> assert e.version(var, default=None) is None
        >>>
        >>> environ[var] = '1'
        >>> assert e.version(var, default=None) is None
        >>>
        >>> environ[var] = '1.'
        >>> assert e.version(var, default=None) is None
        >>>
        >>> environ[var] = '1.3.0'
        >>> assert e.version(var) == Version.parse('1.3.0')
        >>>
        >>> environ[var] = 'v1.3.0'
        >>> assert e.version(var) == Version.parse('1.3.0')

        Args:
            name: var name.
            exc: exception if no default.
            **kwargs: default.

        Returns:
            Version.
        """
        kw = kwargs.copy()
        found = False
        if exc and 'default' not in kwargs:
            found = True
            kw = kwargs.copy() | dict(default=None)
        if rv := noexc(self, name, **kw):
            return self.version_parser(rv, exc, **kwargs)
        if exc and found:
            raise marshmallow.ValidationError(f'Not a valid version: {name=}, {rv=}')

    @staticmethod
    def version_parser(value, exc=False, **kwargs):
        """
        Parse IP.

        >>> from rc import Env, Version
        >>>
        >>> assert Env.version_parser('name@test') is None
        >>> # noinspection PyTypeChecker
        >>> assert Env.version_parser(1, default=None) is None
        >>> assert Env.version_parser('1.', default=None) is None
        >>> assert Env.version_parser('1.0', default=None) is None
        >>> assert Env.version_parser('1.3.0') == Version.parse('1.3.0')
        >>> assert Env.version_parser('v1.3.0') == Version.parse('1.3.0')

        Args:
            value: value to parse.
            exc: raise ValidationError if not default.
            **kwargs: default.

        Raises:
            ValidationError.

        Returns:
            Version.
        """
        default = kwargs.pop('default', None)
        if rv := noexc(Version.parse, value):
            return rv
        if default:
            return default
        if exc:
            raise marshmallow.ValidationError(f'Not a valid Version: {value}')


env = Env()
log = getlog(std_add=('name',))
log.error('rc')


class _GitHub(enum.Enum):
    # _all = 'all'
    # _argument = None
    # _autocomplete = ()
    # _default = ''
    # _repos = {}

    @classmethod
    def envvar(cls, name, token=False):
        """
        Get Env Var for name using: GITHUB_NAME/GITHUB_NAME_TOKEN.

        Args:
            name: name to be added to var.
            token: add token.

        Returns:
            Str.
        """
        return env(f'{cls.__name__.lstrip("_")}_{name}' + ('_TOKEN' if token else ''), default=None)

    @classmethod
    def user(cls, name, alias=None):
        """
        GitHub User API Info.

        Args:
            name: GitHub user name.
            alias: alias for env var name.

        Raises:
            FileNotFoundError: No cache file found when GitHub rate limit.

        Returns:
            GitUser.
        """
        msg, user, passwd, url = '', env('GITHUB_WORK'), env('GITHUB_WORK_TOKEN'), GITHUB_API_URL / 'users' / name
        kw = dict(auth=(user, passwd)) if user and passwd else {}
        if name is None or ((rv := requests.get(url, **kw).json()) and ((msg := rv.get('message', '')) == 'Not Found')):
            return
        file = TMPDIR / f'{cls.user.__qualname__}_{name}{f"_{alias}" if alias else ""}'
        if 'API rate limit exceeded' in msg:
            if file.is_file():
                rv = json.load(file.open())
            else:
                raise FileNotFoundError(f'{file=}, {rv=}')
        else:
            json.dump(rv, file.open(mode='w'))
        https, login, org = furl(rv['html_url']), rv['login'], f'org-{rv["id"]}'
        host = https.host
        return GitUser(blog=furl(rv['blog']), id=rv['id'], https=https,
                       key=requests.get(https.url + '.keys').text.rstrip(NEWLINE),
                       login=login, name=rv['name'], org=org, pip=furl(f'git+ssh://git@{host}/{login}'),
                       repos=furl(rv['repos_url']), ssh=furl(f'{org}@{host}:{login}'),
                       token=_GitHub.envvar(alias if alias else name, token=True), url=furl(rv['url']))
        # repos = requests.get('https://j5pu.github.io/repos/repos/').json()
        # j5pu = requests.get('https://api.github.com/users/j5pu').json()
        # j5pu_blog = j5pu['blog']  # 'https://j5pu.github.io/projects/repos/'
        # j5pu_login = j5pu['login']  # j5pu
        # j5pu_company = j5pu['company']  # lumenbiomics
        # j5pu_name = j5pu['name']  # José
        # j5pu_id = j5pu['id']  # 26859654
        # j5pu_html = j5pu['html_url']  # 'https://github.com/j5pu',
        # j5pu_repos = j5pu['repos_url']  # 'https://api.github.com/users/j5pu/repos'
        # j5pu_org_id = f'org-{j5pu_id}'  # org-26859654
        # j5pu_ssh = requests.get('https://github.com/j5pu.keys').text.rstrip(NEWLINE)
        # # git clone org-26859654@github.com:j5pu/climaxporn.git
        #
        # jose_nferx = requests.get('https://api.github.com/users/jose-nferx').json()
        # jose_nferx_login = jose_nferx['login']  # jose_nferx
        # jose_nferx_name = jose_nferx['name']  # José A. Puértolas
        # jose_nferx_organizations_url = jose_nferx['organizations_url']
        # 'https://api.github.com/users/jose-nferx/orgs'
        # jose_nferx_organizations = requests.get(jose_nferx_organizations_url).json()
        #
        # jose_nferx_lumenbiomics = jose_nferx_organizations[0]['url']  # 'https://api.github.com/orgs/lumenbiomics',
        #
        # jose_nferx_ssh = requests.get('https://github.com/jose-nferx.keys').text.rstrip(NEWLINE)
        #
        # lumenbiomics = requests.get(f'https://api.github.com/users/{j5pu["company"]}').json()
        # lumenbiomics_blog = lumenbiomics['blog']  # 'https://nference.ai'
        # lumenbiomics_login = lumenbiomics['login']  # lumenbiomics
        # lumenbiomics_name = lumenbiomics['name']  # nference
        # lumenbiomics_id = lumenbiomics['id']  # 4379404
        # lumenbiomics_html = lumenbiomics['html_url']  # 'https://github.com/lumenbiomics',
        # lumenbiomics_repos = lumenbiomics['repos_url']  # 'https://api.github.com/users/lumenbiomics/repos'
        # lumenbiomics_org_id = f'org-{lumenbiomics_id}'  # org-4379404

    def _generate_next_value_(self: str, *args):
        return _GitHub.user(name=_GitHub.envvar(self), alias=self)


class GitHub(_GitHub):
    """
    GitHub User API Info Enum Class.

    >>> from furl import furl
    >>> from rc import GitHub, GitUser
    >>>
    >>> for i in GitHub:
    ...     assert i.value is not None
    ...     for k, v in i.value._asdict().items():
    ...         if isinstance(v, furl):
    ...             assert v.url
    ...         else:
    ...             if i is not GitHub.ORG and k != 'key':
    ...                 assert v
    """
    ORG = enum.auto()
    USER = enum.auto()
    WORK = enum.auto()

    @classmethod
    def argument(cls):
        """
        Repos Typer Argument.

        Returns:
            Argument with Repos Names.
        """
        try:
            return cachemodule[cls.argument]
        except KeyError:
            cachemodule[cls.argument] = rv = typer.Argument(
                cls._default, autocompletion=lambda: cls.autocomplete, case_sensitive=False,
                help=f'File, Directory, Repository name (<"{ALL}"> to perform command in all repositories): '
                     f'{cls.autocomplete} or CWD. Default: <""> for CWD.')
            return rv

    @classmethod
    def autocomplete(cls):
        """
        Repos Autocomplete.

        Returns:
            Tuple Repo Names.
        """
        try:
            return cachemodule[cls.autocomplete]
        except KeyError:
            cachemodule[cls.autocomplete] = rv = (ALL,) + tuple(sorted(cls.repos()))
            return rv

    @classmethod
    def https(cls, name, auth=False):
        """
        GitHub HTTPs URL.

        >>> from rc import GitHub, is_giturl
        >>>
        >>> assert is_giturl(GitHub.https('bashrc'))
        >>>
        >>> assert is_giturl(GitHub.https('bashrc', auth=True))

        Args:
            name: repo name.
            auth: include username and password.

        Returns:
            furl GitHub HTTPs URL.
        """
        key = 'organization'
        organization = cls.map()[key][cls.repos()[name][key]]
        rv = organization.value.https / Pname.NONE.git(name).name
        if auth:
            rv.password = organization.value.token
            rv.username = organization.value.login
        return rv

    @classmethod
    def map(cls):
        """
        Mapping cls.repos keys to values.

        Returns:
            Dict.
        """
        # TODO: Probar el logger. Meter el lock.
        # TODO: Aqui antes de ir a por el logger para guardar los que no se encontraban
        try:
            return cachemodule[cls.map]
        except KeyError:
            cachemodule[cls.map] = rv = dict(organization={True: GitHub.ORG, False: GitHub.WORK, None: GitHub.USER},
                                             pypi={True: GitHub.WORK.value.login, False: GitHub.USER.value.login,
                                                   None: None})
            log.warning(f'{cachemodule[cls.map]=}')
            return rv

    @classmethod
    def pip(cls, name, auth=True, ssh=True):
        """
        GitHub PIP URL.

        >>> from rc import GitHub, is_giturl
        >>>
        >>> assert is_giturl(GitHub.pip('bashrc'))

        Args:
            name: repo name.
            auth: include username and password for https.
            ssh: ssh or https.

        Returns:
            furl GitHub PIP URL.
        """
        if ssh:
            key = 'organization'
            organization = cls.map()[key][cls.repos()[name][key]]
            return organization.value.pip / Pname.NONE.git(name).name
        rv = cls.https(name, auth=auth)
        rv.scheme = 'git+https'
        return rv

    @classmethod
    def py(cls, name):
        """
        GitHub SSH URL.

        >>> from rc import GitHub
        >>>
        >>> assert GitHub.py('bashrc') is True

        Args:
            name: repo name.

        Returns:
            furl GitHub SSH URL.
        """
        return cls.repos()[name][cls.py.__name__]

    @classmethod
    def pypi(cls, name):
        """
        GitHub SSH URL.

        >>> from rc import GitHub, env
        >>>
        >>> assert GitHub.pypi('bashrc') == env('GITHUB_USER')

        Args:
            name: repo name.

        Returns:
            furl GitHub SSH URL.
        """
        key = cls.pypi.__name__
        if cls.py(name):
            return cls.map()[key][cls.repos()[name][key]]

    @classmethod
    def repos(cls):
        """
        Repo Information.

        [repos](https://j5pu.github.io/projects/repos/)

        1. **organization**: *GitHub User*.

          * `true`: organization.
          * `false`: work.
          * `null`: personal.

        2. **py**: *Python Project*.

          * `true`: `pyproject.toml`.
          * `false`.

        3. **pypi**: *PyPi Upload*.

          * `true`: organization.
          * `false`: personal.
          * `null`: GitHub only.

        Returns:
            GitRepo.
        """
        try:
            return cachemodule[cls.repos]
        except KeyError:
            cachemodule[cls.repos] = rv = requests.get(cls.USER.value.blog.url).json()
            return rv

    @classmethod
    def ssh(cls, name):
        """
        GitHub SSH URL.

        >>> from rc import GitHub, is_giturl
        >>>
        >>> assert is_giturl(GitHub.ssh('bashrc'))
        >>> assert is_giturl(GitHub.ssh('pen'))

        Args:
            name: repo name.

        Returns:
            furl GitHub SSH URL.
        """
        key = 'organization'
        organization = cls.map()[key][cls.repos()[name][key]]
        return organization.value.ssh / Pname.NONE.git(name).name


class Command:
    __slots__ = ()

    @staticmethod
    @app.command()
    def git(data: str = ''):
        """
        Git Top Level Path (stdout: True) or :class:`git.Repo` if Valid Repository.

        Obtained from: File, Directory, Name (relative to ``HOME``) or CWD.

        Args:
            data: File, Directory, Name (relative to ``HOME``) or '' for cwd (default: '').

        Returns:
            None.
        """
        d = Distro(data=data)
        if rv := d.git:
            print(rv)

    @staticmethod
    @app.command()
    def home(data: str = ''):
        """
        Repository/Development/Clone Path.

        Obtained from: File, Directory, Name (relative to ``HOME``) or CWD.

        Args:
            data: File, Directory, Name (relative to ``HOME``) or '' for cwd (default: '').

        Returns:
            None.
        """
        print(Distro(data=data).home)


class Distro:
    """
    Distribution Helper Class.

    HOME Obtained from: File, Directory, Name (relative to ``HOME``) or CWD.

    Examples:
        >>> from pathlib import Path
        >>> import rc
        >>> from rc import Distro


    """
    __slots__ = ('_configuration', '_dict', '_doproject', '_git', '_https', '_modules', '_name',
                 '_packages', '_piphttps', '_pipssh', '_py', '_pypi', '_pyproject_toml', '_setup', '_setup_cfg',
                 '_ssh', '_url', '_urls', 'default', 'detached_exc', 'doit', 'home', 'rm', 'table')

    def __init__(self, data='', clone=None, detached_exc=True, doit=False, rm=False, ssh=True):
        """
        Distribution Helper Class.

        Args:
            data: File, Directory, Name (relative to ``HOME``) or '' for cwd (default: '').
            clone: clone if not found (if None clone always if MACOS).
            detached_exc: Raise Exception if Project and Detached.
            doit: Ignore has_changes (dirty, porcelain, remotediff, untracked_files) and do it.
            rm: remove if clone and exists.
            ssh: use ssh as default url.

        Return:
            None.
        """
        for i in self.__slots__:
            self.__setattr__(i, None)
        self.detached_exc, self.doit, self.home, found = detached_exc, doit, parent(data) if data else Path.cwd(), False
        self.default, self.rm = self.__class__.ssh if ssh else self.__class__.https, rm
        if rv := self.git:
            self.home, found = Path(rv.working_tree_dir), True
        elif rv := self.configuration.pyproject_toml:
            self.home, found = rv.parent, True
        if not found and data:
            self.home = Path.home() / data
            if self.py and self.home.is_dir() and not (file := self.home / 'pyproject.toml').is_file():
                file.write_text(f"[tool.{conf['distro']['tool']}]\nversion = '0.0.0'")
                self.configuration = None
        if not self.home.is_dir():
            if clone is None and MACOS:
                clone = True
            if clone:
                self.clone()

        self.table = Table(title=f'[bold red]{self.__class__.__name__}:[/] '
                                 f'[bold blue]{self.name}[/]={self.url}', box=rich.box.ROUNDED)
        self.table.add_column('Attribute', style='magenta')
        self.table.add_column('Value', style='green', overflow='fold')

    def __repr__(self): return f'{self.__class__.__name__}(' \
                               f'{", ".join(f"{i}={repr(self._get(i))}" for i in conf["distro"]["repr"])})'

    # noinspection PyUnusedLocal
    def __rich_console__(self, console, options):
        for i in conf['distro']['repr']:
            self.table.add_row(i, str(self._get(i)))
        yield self.table

    def _get(self, attr): return rv() if (rv := object.__getattribute__(self, attr)) and callable(rv) else rv

    @property
    def branch(self):
        """
        Active Branch Name.

        >>> from rc import Distro
        >>>
        >>> d = Distro()
        >>> assert d.branch is not None

        Returns:
            Branch Name (None if detached HEAD)
        """
        if (rv := self.git) and not rv.head.is_detached:
            return rv.active_branch.name

    def branchdefault(self, remote='origin'):
        """
        Branch Default.

        >>> from rc import Distro
        >>>
        >>> assert Distro().branchdefault() == 'main'
        >>> assert Distro().branchdefault(remote=None) == {'origin': 'main'}

        Returns:
            Default Branch Name.
        """
        def cmd(name):
            return shell(f'git symbolic-ref refs/remotes/{name}/HEAD | sed "s@^refs/remotes/{name}/@@"',
                         cwd=self.home)
        if (rv := self.git) and not rv.head.is_detached:
            if not remote:
                return {r.name: cmd(r.name) for r in rv.remotes}
        return cmd(remote)

    @property
    def branches(self):
        """
        Branches Names.

        >>> from rc import Distro
        >>>
        >>> assert 'main' in Distro().branches

        Returns:
            Branches Names.
        """
        if rv := self.git:
            return tuple(i.name for i in cast(Iterable, rv.branches))

    def build(self):
        """
        Build

        >>> from rc import Distro
        >>>
        >>> d = Distro()
        >>> d.build()

        Returns:
            None.
        """
        # shell(f'{sys.executable} {self.configuration.setup_py} sdist', ansi=True, cwd=self.home)
        build_main([str(self.home), '--sdist', '--no-isolation'])

    def clone(self):
        """
        Clone Repository.

        Returns:
            None.
        """
        if self.home.is_dir() and self.rm:
            self.home.unlink(missing_ok=True)
        gitpython.Repo.clone_from(self.default(self).url, self.home)

    @property
    def configuration(self):
        """
        Configuration Files Paths: pyproject.toml, setup.cfg and setup.py.

        >>> from rc import Distro, Pname
        >>>
        >>> d = Distro()
        >>> configuration = d.configuration
        >>> if (d.home / Pname.PYPROJECT.toml()).is_file():
        ...     assert d.configuration.pyproject_toml
        >>> if (d.home / Pname.SETUP.cfg()).is_file():
        ...     assert d.configuration.setup_cfg
        >>> if (d.home / Pname.SETUP.py()).is_file():
        ...     assert d.configuration.setup_py

        Returns:
            Configuration (pyproject.toml, setup.py and setup.py).
        """
        if not self._configuration:
            self._configuration = Configuration(
                pyproject_toml=findup(self.home, name=Pname.PYPROJECT.toml),
                setup_cfg=self.home / Pname.SETUP.cfg(),
                setup_py=self.home / Pname.SETUP.py()
            )
        return self._configuration

    @property
    def detached(self):
        """
        Is Detached Head?. You should create a branch.

        >>> from rc import Distro
        >>>
        >>> d = Distro()
        >>> assert d.detached is not None

        Returns:
            None if not git or True if detached.
        """
        if rv := self.git:
            return rv.head.is_detached
        return None

    def dict(self):
        """
        Project Configuration Data.

        Returns:
            Project Configuration Data Dict.
        """
        if not self.doproject:
            print('Nothing to do here!')
        if self._dict is None:
            configuration = self.configuration
            pyproject = toml.load(configuration.pyproject_toml)

    @property
    def dirty(self):
        """
        Is repo dirty?

        >>> from rc import Distro
        >>>
        >>> d = Distro()
        >>> assert d.dirty is not None

        Returns:
            True if dirty or None if HEAD detached.
        """
        if rv := self.git:
            return None if rv.head.is_detached else rv.is_dirty()

    @property
    def doproject(self):
        """
        Do Project? Is Project and Has Changes (dirty, porcelain, remotediff, untracked_files) and doit.

        Returns:
            True if project has changed or action is required.
        """
        if self._doproject is None:
            self._doproject = all([self.configuration.pyproject_toml, self.git, self.has_changes or self.doit])
        return self._doproject

    @property
    def git(self):
        """
        Instance :class:`git.Repo` if Valid Repository.

        Obtained From: File, Directory, Name (relative to ``HOME``) or '' for cwd.

        >>> from rc import Distro
        >>>
        >>> d = Distro()
        >>> assert d.git is not None

        Raises:
            RuntimeError: Detached Head and Project. You should create a branch.

        Returns:
            Instance :class:`git.Repo` if Valid Repository.
        """
        if self._git is None:
            if rv := shell('git rev-parse --show-toplevel', cwd=self.home, exc=False):
                self._git = gitpython.Repo(rv, search_parent_directories=True)
                if self._git.head.is_detached and self.configuration.pyproject_toml:
                    raise RuntimeError(f"Detached Head and Project. You should create a branch: '{self.home=}'")
        return self._git

    @property
    def has_changes(self):
        """
        Has Changes (untracked files, dirty, porcelain or remote diff)?.

        >>> from rc import Distro
        >>>
        >>> d = Distro()
        >>> assert d.has_changes is not None

        Returns:
            True if Has Changes (untracked files, dirty or remote diff)?..
        """
        if self.git:
            return any([self.dirty, self.porcelain, self.remotediff(), self.untracked_files])

    @property
    def https(self):
        """
        HTTPs Url (from repos).

        >>> from rc import Distro, is_giturl
        >>>
        >>> d = Distro()
        >>>
        >>> assert is_giturl(d.https)

        Returns:
            HTTPs Url (from repos).
        """
        if not self._https:
            self._https = GitHub.https(self.name, auth=True)
        return self._https

    @property
    def ls(self):
        """
        Git LS.

        >>> from pathlib import Path
        >>> import rc
        >>> from rc import Distro
        >>>
        >>> assert len(Distro().ls) > 1

        Returns:
            Tuple of Files under Git.
        """
        if rv := self.branch:
            return tuple(map(Path, shell(f'git ls-tree --name-only -r {rv}', cwd=self.home).splitlines()))

    def manifest(self):
        """
        Updates MANIFEST.in

        >>> from pathlib import Path
        >>> import rc
        >>> from rc import Distro, conf
        >>>
        >>> d = Distro()
        >>> d.manifest()
        >>> ls, text, p = d.ls, (d.home / 'MANIFEST.in').read_text(), Path(rc.__file__)
        >>> assert f'{p.parent.name}/{p.name}' in text
        >>> for i in conf['distro']['exclude']['manifest']:
        ...     assert i not in text
        >>> for i in d.scripts:
        ...     if i in ls:
        ...         assert i in text

        Returns:
            None
        """
        (self.home / 'MANIFEST.in').write_text('\n'.join(
            [f'include {l}' for l in self.ls if l.parts[0] not in conf['distro']['exclude']['manifest']]))

    @property
    def modules(self):
        """
        Find Modules.

        >>> from icecream import ic
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> import rc
        >>> from rc import Distro, Pname
        >>>
        >>> with TemporaryDirectory() as tmp:
        ...     (Path(tmp) / Pname.SETUP.py()).touch()
        ...     test = Path(tmp) / 'test.py'
        ...     test.touch()
        ...     d = Distro(tmp)
        ...     assert d.modules['test'] == test
        ...     assert Pname.SETUP.py().name not in d.modules

        Returns:
            List of Module Names.
        """
        if not self._modules:
            self._modules = {i.stem: i / i for i in self.home.iterdir()
                             if i.suffix == Pname.PY.dot.name and i.name != Pname.SETUP.py().name}
        return self._modules

    @property
    def name(self):
        """
        Repository Name.

        >>> from pathlib import Path
        >>> import rc
        >>> from rc import Distro
        >>>
        >>> assert Distro().name == Path(rc.__file__).parent.parent.name

        Obtained From: Origin, File, Directory, Name (relative to ``HOME``) or '' for cwd.

        Returns:
            Name.
        """
        if self._name is None:
            self._name = Path(rv.path.segments[-1]).stem if (rv := self.url) and rv.url else self.home.name
        return self._name

    @property
    def packages(self):
        """
        Find Packages.

        >>> from pathlib import Path
        >>> import rc
        >>> from rc import Distro
        >>>
        >>> assert Distro().packages[rc.__name__] == Path(rc.__file__).parent

        Returns:
            List of Package Names.
        """
        if not self._packages:
            self._packages = {i: self.home / i for i in setuptools.find_packages(
                self.home, exclude=conf['distro']['exclude']['packages'])}
        return self._packages

    @property
    def piphttps(self):
        """
        PIP HTTPs Url (from repos).

        >>> from rc import Distro
        >>>
        >>> d = Distro()
        >>>
        >>> assert d.piphttps

        Returns:
            PIP HTTPs Url (from repos).
        """
        if not self._piphttps:
            self._piphttps = GitHub.pip(self.name, ssh=False)
        return self._piphttps

    @property
    def pipssh(self):
        """
        PIP SSH Url (from repos).

        >>> from rc import Distro, is_giturl
        >>>
        >>> d = Distro()
        >>>
        >>> assert is_giturl(d.pipssh)

        Returns:
            PIP SSH Url (from repos).
        """
        if not self._pipssh:
            self._pipssh = GitHub.pip(self.name)
        return self._pipssh

    @property
    def porcelain(self):
        """
        Untracked Files.

        >>> from pathlib import Path
        >>> import rc
        >>> from rc import Distro
        >>>
        >>> assert Distro().porcelain is not None

        Returns:
            Untracked Files Tuple.
        """
        if rv := self.git:
            return None if rv.head.is_detached else rv.git.status(porcelain=True, untracked_files=False)

    @property
    def py(self):
        """
        Is Py project (from repos)?.

        >>> from rc import Distro, Pname
        >>>
        >>> d = Distro()
        >>> assert d.py is True

        Returns:
            True if Py Distro.
        """
        if not self._py:
            self._py = GitHub.py(self.name) or (self.home / Pname.PYPROJECT.toml()).is_file() or (
                    self.home / Pname.SETUP.cfg()).is_file() or (self.home / Pname.SETUP.py()).is_file()
        return self._py

    @property
    def pypi(self):
        """
        PyPi User/Repository if Upload to PyPi (from repos).

        >>> from rc import Distro, env
        >>>
        >>> d = Distro()
        >>>
        >>> assert d.pypi == env('GITHUB_USER')

        Returns:
            PyPi User/Repository if Upload to PyPi or None (not upload).
        """
        if not self._pypi:
            self._pypi = GitHub.pypi(self.name)
        return self._pypi

    @property
    def pyproject_toml(self):
        """
        Reads pyproject.toml and updates build-system.

        >>> from rc import Distro, Pname
        >>>
        >>> d = Distro()
        >>> pyproject = d.pyproject_toml
        >>> assert 'build-system' in pyproject

        Returns:
            Updated pyproject.toml Dict.
        """
        file = self.configuration.pyproject_toml
        if not self._pyproject_toml and self.doproject:
            self._pyproject_toml = toml.load(file)
            # TODO: Version.
            if self._pyproject_toml.get('build-system') != conf['build-system']:
                self._pyproject_toml['build-system'] = conf['build-system']
                toml.dump(self._pyproject_toml, file)
        return self._pyproject_toml

    def remotediff(self, branch=None, name_only=True, remote=None, stat=False):
        """
        Git Diff.

        >>> from rc import Distro
        >>>
        >>> assert Distro().remotediff() is not None

        Args:
            branch: branch name. None or '' for default branch name.
            name_only: show only names of changed files.
            remote: '' for all remotes or remote name.
            stat: show diffstat instead of patch.

        Returns:
            Remote str diff if remote else dict with remote and diffs.
        """
        def value(name):
            b = branch if branch else self.branchdefault(remote=name)
            return g.git.diff(str(Path(name) / b), name_only=name_only, stat=stat)

        if g := self.git:
            if not remote:
                rv = {}
                for r in self.remotes:
                    if v := value(r):
                        rv |= {r: v}
                return rv
            return value(remote)

    @property
    def remotes(self):
        """
        Remotes Names.

        >>> from pathlib import Path
        >>> import rc
        >>> from rc import Distro
        >>>
        >>> assert Distro().remotes == ('origin', )

        Returns:
            Remotes Names Tuple.
        """
        if rv := self.git:
            return tuple(i.name for i in rv.remotes)

    @property
    def requirements(self):
        """
        Parse Requirements Files.

        >>> import jupyter
        >>> from pathlib import Path
        >>> import rc
        >>> from rc import Distro
        >>>
        >>> distro = Distro()
        >>> assert jupyter.__name__ in distro.requirements['optional-dependencies']['dev']

        Returns:
            List Script Paths Relative to Git dir.
        """
        def parse(_i):
            return tuple(r.requirement for r in parse_requirements(str(_i), session=session))

        home, name, session = self.home, Pname.REQUIREMENTS.name, PipSession()
        rv = {'dependencies': (), 'optional-dependencies': {}}
        for i in list(self.home.glob(f'*{name}*')) + list((home / name).glob(f'*{name}*')):
            if i.is_file():
                if (key := i.stem.rpartition('_')[2]) == name:
                    rv['dependencies'] = parse(i)
                else:
                    rv['optional-dependencies'][key] = parse(i)
        return rv

    def run(self):
        """
        Build, Publish.

        Returns:

        """
        if not self.doproject:
            print('Nothing to do here!')
        configuration = self.configuration
        pyproject = toml.load(configuration.pyproject_toml)

    @property
    def scripts(self):
        """
        Get Scripts Relative to Git Dir in path or parent directory.

        >>> from pathlib import Path
        >>> import rc
        >>> from rc import Distro
        >>>
        >>> distro = Distro()
        >>>
        >>> assert str(Path('scripts') / distro.name) in distro.scripts

        Returns:
            List Script Paths Relative to Git dir.
        """
        scripts = []
        if (rv := self.home / 'scripts').is_dir():
            scripts.append(rv)
        for i in self.packages:
            if (rv := self.home / i / 'scripts').is_dir():
                scripts.append(rv)
        return {str(s.relative_to(self.home)): s.resolve() for item in scripts for s in item.iterdir()}

    def setup(self):
        """
        Needs `pyproject.toml` file with [tool.rc] section and version key to read configuration.

        Updates: `pyproject.toml`, `setup.cfg` and `MANIFEST.in`.

            .. code-block:: toml

                [tool.rc]  # pyproject.toml (Read)
                cmdclass = {install = 'bashrc:Install'}
                entry-points = {rc = 'rc:app'}
                organization = true  # GitHub User: ``true`` organization & absent/anything else for personal.
                pypi = false  # PyPi Upload: ``true`` organization, ``false`` personal & absent/anything else only Git.
                version = '0.0.0'  # Required.

                [metadata]  # setup.cfg (Updates fields if empty)
                author = jose
                author_email = jose@google.com
                description = My package description
                long_description = file: README.md
                name = bashrc
                url = https://github.com/j5pu/bashrc.git
                version = 0.31.90

                [options]
                cmdclass =
                    install = rc.Install
                include_package_data = True
                install_requires =
                    requests
                packages = find:
                py_modules =  # if no packages
                    rc
                python_requires = >= 3.9
                scripts =
                    scripts/bashrc
                zip_safe = False

                [options.entry_points]
                console_scripts =
                    rc = rc:app

                [options.extras_require]
                beta =
                    bidict
                dev =
                    icecream
                    jupyter
                test =
                    pytest

                [options.packages.find]  # if packages
                exclude =
                    backup
                    doc
                    tests
                    tmp
                    venv

        Examples:
            >>> from rc import Distro, Pname
            >>>
            >>> d = Distro()
            >>> assert d.setup()

        Returns:
            Dict/kwargs for setup.py
        """
        if not self.doproject:
            return
        author = 'jose'
        author_email = 'jose@google'
        cmdclass = {'install': Install}
        entry_points = dict(console_scripts=['rc = bashrc:app'])
        extras_require = list(self.requirements['optional-dependencies'])
        install_requires = list(self.requirements['dependencies'])
        name = self.name
        packages = list(self.packages)
        python_requires = '>= 3.9'
        scripts = list(self.scripts)
        url = 'https://github.com',
        version = '0.31.90'
        zip_safe = False
        if conf['distro']['legacy']:
            toml.dump(
                dict(
                    project={  # https://www.python.org/dev/peps/pep-0621/
                        'authors': [dict(name=author, email=author_email)],
                        'entry-points': entry_points,
                        'optional-dependencies': extras_require,
                        'dependencies': install_requires,
                        'name': self.name,
                        'requires-python': python_requires,
                        'urls': dict(homepage=url),
                        'version': version,
                    },
                    **envtoml.load(self.configuration.pyproject_toml)
                ),
                self.configuration.pyproject_toml)
            self._setup = dict(
                cmdclass=cmdclass,
                packages=packages,
                scripts=list(self.scripts),
                zip_safe=zip_safe,
            )
        else:
            self._setup = dict(
                author=author,
                author_email=author_email,
                cmdclass=cmdclass,
                entry_points=dict(console_scripts=['rc = bashrc:app']),
                extras_require=extras_require,
                install_requires=install_requires,
                name=name,
                packages=packages,
                python_requires=python_requires,
                scripts=scripts,
                setup_requires=['setuptools', 'wheel'],
                url=url,
                version=version,
                zip_safe=zip_safe,
            )
        return self._setup

    # def setup(self):
    #     """
    #     Setup dict/kwargs for setup.py.
    #
    #     >>> from rc import Distro, Pname
    #     >>>
    #     >>> d = Distro()
    #     >>> assert d.setup()
    #
    #     Returns:
    #         Dict/kwargs for setup.py
    #     """
    #     if not self.doproject:
    #         return
    #     author = 'jose'
    #     author_email = 'jose@google'
    #     cmdclass = {'install': Install}
    #     entry_points = dict(console_scripts=['rc = bashrc:app'])
    #     extras_require = list(self.requirements['optional-dependencies'])
    #     install_requires = list(self.requirements['dependencies'])
    #     name = self.name
    #     packages = list(self.packages)
    #     python_requires = '>= 3.9'
    #     scripts = list(self.scripts)
    #     url = 'https://github.com',
    #     version = '0.31.90'
    #     zip_safe = False
    #     if conf['distro']['legacy']:
    #         toml.dump(
    #             dict(
    #                 project={  # https://www.python.org/dev/peps/pep-0621/
    #                     'authors': [dict(name=author, email=author_email)],
    #                     'entry-points': entry_points,
    #                     'optional-dependencies': extras_require,
    #                     'dependencies': install_requires,
    #                     'name': self.name,
    #                     'requires-python': python_requires,
    #                     # 'scripts': {'rc': 'bashrc:app'},
    #                     'urls': dict(homepage=url),
    #                     'version': version,
    #                 },
    #                 **envtoml.load(self.pyproject_toml)
    #             ),
    #             self.pyproject_toml)
    #         self._setup = dict(
    #             cmdclass=cmdclass,
    #             packages=packages,
    #             scripts=list(self.scripts),
    #             zip_safe=zip_safe,
    #         )
    #     else:
    #         self._setup = dict(
    #             author=author,
    #             author_email=author_email,
    #             cmdclass=cmdclass,
    #             entry_points=dict(console_scripts=['rc = bashrc:app']),
    #             extras_require=extras_require,
    #             install_requires=install_requires,
    #             name=name,
    #             packages=packages,
    #             python_requires=python_requires,
    #             scripts=scripts,
    #             setup_requires=['setuptools', 'wheel'],
    #             url=url,
    #             version=version,
    #             zip_safe=zip_safe,
    #         )
    #     return self._setup

    # def setup(self):
    #     if self.git:
    #         (self.project / MANIFEST).write_text('\n'.join([f'include {l}' for l in self.git.ls]))
    #
    #     self.setup_kwargs = dict(
    #         author=self.user.gecos, author_email=Url.email(), description=self.description,
    #         entry_points=dict(console_scripts=[f'{p} = {p}:{CLI}' for p in self.packages_upload]),
    #         include_package_data=True, install_requires=self.requirements.get('requirements', list()), name=self.repo,
    #         package_data={self.repo: [f'{p}/{d}/*' for p in self.packages_upload
    #                                   for d in (self.project / p).scan(PathOption.DIRS)
    #                                   if d not in self.exclude_dirs + tuple(self.packages + [DOCS])]},
    #         packages=self.packages_upload, python_requires=f'>={PYTHON_VERSIONS[0]}, <={PYTHON_VERSIONS[1]}',
    #         scripts=self.scripts_relative, setup_requires=self.requirements.get('requirements_setup', list()),
    #         tests_require=self.requirements.get('requirements_test', list()),
    #         url=Url.lumenbiomics(http=True, repo=self.repo).url,
    #         version=__version__, zip_safe=False
    #     )
    #     return self.setup_kwargs

    @property
    def setup_cfg(self):
        """
        setup.cfg update.

        >>> from rc import Distro, Pname
        >>>
        >>> d = Distro()
        >>> if (d.home / Pname.SETUP.cfg()).is_file():
        ...     assert d.setup_cfg

        Returns:
            setup.cfg Path.
        """
        # TODO: Aqui lo dejo. Ver que hago con author etc.. y si uso lo de icloud de python o no.
        file = self.configuration.setup_cfg
        if not self._setup_cfg and self.doproject:
            self._setup_cfg = toml.load(file)
            if self._pyproject_toml.get('build-system') != conf['build-system']:
                self._pyproject_toml['build-system'] = conf['build-system']
                toml.dump(self._pyproject_toml, file)
        c = read_configuration(str(file))
        return self._setup_cfg

    @property
    def ssh(self):
        """
        SSH Url (from repos).

        >>> from rc import Distro, is_giturl
        >>>
        >>> d = Distro()
        >>>
        >>> assert is_giturl(d.ssh)

        Returns:
            SSH Url (from repos).
        """
        if not self._ssh:
            self._ssh = GitHub.ssh(self.name)
        return self._ssh

    @property
    def untracked_files(self):
        """
        Untracked Files.

        >>> from pathlib import Path
        >>> import rc
        >>> from rc import Distro
        >>>
        >>> assert Distro().untracked_files is not None

        Returns:
            Untracked Files Tuple.
        """
        if rv := self.git:
            return tuple(rv.untracked_files)

    @property
    def url(self):
        """
        Origin Url.

        >>> from pathlib import Path
        >>> import rc
        >>> from rc import Distro
        >>>
        >>> assert Path(Distro().url.path.segments[-1]).stem == Path(rc.__file__).parent.parent.name

        Returns:
            Origin Url.
        """
        if self.git and (self._url is None):
            self._url = furl(shell(f'git config --get remote.origin.url', cwd=self.home, exc=False))
        return self._url

    @property
    def urls(self):
        """
        Remotes Urls.

        >>> from pathlib import Path
        >>> import rc
        >>> from rc import Distro
        >>>
        >>> assert Distro().urls['origin'][0].url

        Returns:
            Remotes Urls Dict.
        """
        if (rv := self.git) and (self._urls is None):
            self._urls = {i.name: tuple(map(furl, i.urls)) for i in rv.remotes}
        return self._urls


class Install(setuptools.command.install.install):
    """INSTALL ``cmdclass`` :class:`setuptools.command.install.install` Sub Class."""

    def run(self):
        setuptools.command.install.install.run(self)
        print('Hello!')


class Version(VersionInfo):
    """Version Helper Class."""
    _none = None
    __slots__ = ()

    def __init__(self, major, minor=0, patch=0, prerelease=None, build=None):
        super().__init__(major=major, minor=minor, patch=patch, prerelease=prerelease, build=build)

    @property
    def git(self): return self.vtext

    @classmethod
    def none(cls):
        if cls._none is None:
            cls._none = cls(0)
        return cls._none

    @classmethod
    def parse(cls, version=None):
        if version:
            return cls(*(super().parse((version.name if isinstance(version, TagReference)
                                        else version).removeprefix('v'))).to_tuple())

    @property
    def text(self): return str(self)

    @classmethod
    @cache
    def versions(cls, versions=None):
        if versions:
            return tuple(sorted([cls.parse(i) for i in versions]))

    @property
    def vtext(self): return f'v{str(self)}'
