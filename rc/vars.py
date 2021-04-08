# -*- coding: utf-8 -*-
"""Vars Module."""
__all__ = (
    'all_var',
    'main',
    'split_sep',
    'varname',
    'AUTHORIZED_KEYS',
    'Console',
    'FILE_DEFAULT',
    'FRAME_INDEX',
    'FRAME_SYS_INIT',
    'FUNCTION_MODULE',
    'FrameSysType',
    'GITCONFIG',
    'ID_RSA',
    'ID_RSA_PUB',
    'MODULE_MAIN',
    'NEWLINE',
    'PYTHON_SYS',
    'PYTHON_SITE',
    'SSH_DIR',
    'SUDO_USER',
    'SUDO',
    'SUDO_DEFAULT',
    'SYS_PATHS',
    'SYS_PATHS_EXCLUDE',
    'console',
    'debug',
    'fm',
    'fmic',
    'fmicc',
    'ic',
    'icc',
    'pp',
    'print_exception',
    'ModuleVars',
    'MODULE_VARS',
    'SSH_CONFIG',
    'SSH_CONFIG_TEXT',
    'PATHS_EXCLUDE',
    'ALL_COMPLETED',
    'Alias',
    'BYTECODE_SUFFIXES',
    'BaseModel',
    'Box',
    'CancelledError',
    'Condition',
    'Event',
    'FIRST_COMPLETED',
    'FIRST_EXCEPTION',
    'FrameInfo',
    'FrameType',
    'getsitepackages',
    'GitConfigParser',
    'Interval',
    'IntervalTree',
    'ModuleSpec',
    'ModuleType',
    'POST_INIT_NAME',
    'PathLike',
    'Queue',
    'QueueEmpty',
    'QueueFull',
    'Semaphore',
    'Template',
    'TemporaryDirectory',
    'Typer',
    'USER_SITE',
    'all_tasks',
    'as_completed',
    'ast',
    'asyncio',
    'attrgetter',
    'cache',
    'chdir',
    'contains',
    'contextmanager',
    'create_subprocess_exec',
    'create_subprocess_shell',
    'create_task',
    'current_task',
    'ensure_future',
    'find_packages',
    'first_true',
    'furl',
    'gather',
    'get_event_loop',
    'get_paths',
    'get_running_loop',
    'getargvalues',
    'getenv',
    'getmodulename',
    'importlib',
    'insp',
    'inspect',
    'isasyncgen',
    'isasyncgenfunction',
    'isawaitable',
    'iscoroutine',
    'iscoroutinefunction',
    'isfuture',
    'module_from_spec',
    'os',
    'pathlib',
    'pretty',
    'pytest',
    'quote',
    're',
    'recursive_repr',
    'rmtree',
    'run_coroutine_threadsafe',
    'site',
    'sleep',
    'spec_from_file_location',
    'subprocess',
    'suppress',
    'sys',
    'sysconfig',
    'textwrap',
    'to_thread',
    'tokenize',
    'urllib3',
    'wrap_future'
)

import ast
import asyncio
import importlib
import inspect
import os
import pathlib
import re
import subprocess
import sys
import sysconfig
import textwrap
import tokenize
import typing
import urllib3
from asyncio import ALL_COMPLETED
from asyncio import all_tasks
from asyncio import as_completed
from asyncio import CancelledError
from asyncio import Condition
from asyncio import create_subprocess_exec
from asyncio import create_subprocess_shell
from asyncio import create_task
from asyncio import current_task
from asyncio import ensure_future
from asyncio import Event
from asyncio import FIRST_COMPLETED
from asyncio import FIRST_EXCEPTION
from asyncio import gather
from asyncio import get_event_loop
from asyncio import iscoroutine
from asyncio import isfuture
from asyncio import Queue
from asyncio import QueueEmpty
from asyncio import QueueFull
from asyncio import run_coroutine_threadsafe
from asyncio import Semaphore
from asyncio import sleep
from asyncio import to_thread
from asyncio import wrap_future
from asyncio.events import get_running_loop
from collections import namedtuple
from contextlib import contextmanager
from contextlib import suppress
from dataclasses import _POST_INIT_NAME as POST_INIT_NAME
from functools import cache
from importlib.machinery import ModuleSpec
from importlib.machinery import BYTECODE_SUFFIXES
from importlib.util import module_from_spec
from importlib.util import spec_from_file_location
from inspect import FrameInfo
from inspect import getargvalues
from inspect import getmodulename
from inspect import isasyncgen
from inspect import isasyncgenfunction
from inspect import isawaitable
from inspect import iscoroutinefunction
from operator import attrgetter
from operator import contains
from os import chdir
from os import getenv
from os import PathLike
from pprint import pformat
from reprlib import recursive_repr
from shlex import quote
from shutil import rmtree
from site import getsitepackages
from site import USER_SITE
from sysconfig import get_paths
from tempfile import TemporaryDirectory
from types import ModuleType
from types import FrameType
from typing import Optional

import colorama
import pytest
import rich.console
import typer
from box import Box
from devtools import Debug
from furl import furl
from git import GitConfigParser
from icecream import IceCreamDebugger
from intervaltree import Interval
from intervaltree import IntervalTree
from jinja2 import Template
from more_itertools import first_true
from pydantic import BaseModel
from rich import pretty
from setuptools import find_packages
from typer import Typer


def all_var(file: str = None):
    file = pathlib.Path(file if file else inspect.stack()[1].filename)
    if file.exists():
        add = list()
        exclude = ['os.environ', '__all__', '@', 'import ', 'from ']
        lines = file.read_text().splitlines()
        for line in lines:
            if 'spec' in line:
                ic(line)
            if not any([re.search('^ ', line), *[v in line for v in exclude], line == str()]):
                found = False
                for word in ['async def ', 'def ', 'class ']:
                    if line.startswith(word):
                        add.append("    '" + line.replace(':', str()).split(word)[1].split('(')[0] + "'")
                        found = True
                        break
                if not found and ' = ' in line:
                    add.append("    '" + line.split(' = ')[0] + "'")

        print(f'__all__ = ({NEWLINE}' + f',{NEWLINE}'.join(add) + f'{NEWLINE})')


def main(file: str):
    """__all__ var"""
    all_var(file)


def split_sep(sep: str = '_') -> dict:
    return dict(sep=sep) if sep else dict()


def varname(index: int = 2, lower: bool = True, sep: str = '_') -> Optional[str]:
    """
    Caller var name.

    Examples:

        .. code-block:: python

            class A:

                def __init__(self):

                    self.instance = varname()

            a = A()

            var = varname(1)

    Args:
        index: index.
        lower: lower.
        sep: split.

    Returns:
        Optional[str]:
    """
    with suppress(IndexError, KeyError):
        _stack = inspect.stack()
        func = _stack[index - 1].function
        index = index + 1 if func == POST_INIT_NAME else index
        if line := textwrap.dedent(_stack[index].code_context[0]):
            if var := re.sub(f'(.| ){func}.*', str(), line.split(' = ')[0].replace('assert ', str()).split(' ')[0]):
                return (var.lower() if lower else var).split(**split_sep(sep))[0]


AUTHORIZED_KEYS = varname(1, sep=str())
Console = rich.console.Console
FILE_DEFAULT = True
FRAME_INDEX = 1
FRAME_SYS_INIT = sys._getframe(0)
FUNCTION_MODULE = '<module>'
FrameSysType = type(FRAME_SYS_INIT)
GITCONFIG = '.gitconfig'
ID_RSA = varname(1, sep=str())
ID_RSA_PUB = 'id_rsa.pub'
MODULE_MAIN = '__main__'
NEWLINE = '\n'
PYTHON_SYS = sys.executable
PYTHON_SITE = str(pathlib.Path(PYTHON_SYS).resolve())
SSH_DIR = '.ssh'
SUDO_USER = os.getenv('SUDO_USER')
SUDO = bool(SUDO_USER)
SUDO_DEFAULT = True
SYS_PATHS = {key: pathlib.Path(value) for key, value in sysconfig.get_paths().items()}
SYS_PATHS_EXCLUDE = [SYS_PATHS[item] for item in ['stdlib', 'purelib', 'include', 'platinclude', 'scripts']]
console = Console(color_system='256')
debug = Debug(highlight=True)
fm = pformat
fmic = IceCreamDebugger(prefix=str()).format
fmicc = IceCreamDebugger(prefix=str(), includeContext=True).format
ic = IceCreamDebugger(prefix=str())
icc = IceCreamDebugger(prefix=str(), includeContext=True)
pp = console.print
print_exception = console.print_exception

ModuleVars = namedtuple('ModuleVars', 'all, annotations, builtins, cached, doc, file, loader, name, package, repo, '
                                      'pypi, spec')
MODULE_VARS = ModuleVars(*[f'__{item}__' for item in ModuleVars._fields])
SSH_CONFIG = dict(AddressFamily='inet', BatchMode='yes', CheckHostIP='no', ControlMaster='auto',
                  ControlPath='/tmp/ssh-%h-%r-%p', ControlPersist='20m', IdentitiesOnly='yes', LogLevel='QUIET',
                  StrictHostKeyChecking='no', UserKnownHostsFile='/dev/null')
SSH_CONFIG_TEXT = ' '.join([f'-o {key}={value}' for key, value in SSH_CONFIG.items()])
PATHS_EXCLUDE = SYS_PATHS_EXCLUDE + ([plugins] if (
    plugins := pathlib.Path(os.getenv('PYCHARM_PLUGINS', '<None>'))).resolve().exists() else list())

ALL_COMPLETED = ALL_COMPLETED
Alias = typing._alias
BYTECODE_SUFFIXES = BYTECODE_SUFFIXES
BaseModel = BaseModel
Box = Box
CancelledError = CancelledError
Condition = Condition
Event = Event
FIRST_COMPLETED = FIRST_COMPLETED
FIRST_EXCEPTION = FIRST_EXCEPTION
FrameInfo = FrameInfo
FrameType = FrameType
getsitepackages = getsitepackages
GitConfigParser = GitConfigParser
Interval = Interval
IntervalTree = IntervalTree
ModuleSpec = ModuleSpec
ModuleType = ModuleType
POST_INIT_NAME = POST_INIT_NAME
PathLike = PathLike
Queue = Queue
QueueEmpty = QueueEmpty
QueueFull = QueueFull
Semaphore = Semaphore
Template = Template
TemporaryDirectory = TemporaryDirectory
Typer = Typer
USER_SITE = USER_SITE
all_tasks = all_tasks
as_completed = as_completed
ast = ast
asyncio = asyncio
attrgetter = attrgetter
cache = cache
chdir = chdir
contains = contains
contextmanager = contextmanager
create_subprocess_exec = create_subprocess_exec
create_subprocess_shell = create_subprocess_shell
create_task = create_task
current_task = current_task
ensure_future = ensure_future
find_packages = find_packages
first_true = first_true
furl = furl
gather = gather
get_event_loop = get_event_loop
get_paths = get_paths
get_running_loop = get_running_loop
getargvalues = getargvalues
getenv = getenv
getmodulename = getmodulename
importlib = importlib
insp = rich.inspect
inspect = inspect
isasyncgen = isasyncgen
isasyncgenfunction = isasyncgenfunction
isawaitable = isawaitable
iscoroutine = iscoroutine
iscoroutinefunction = iscoroutinefunction
isfuture = isfuture
module_from_spec = module_from_spec
os = os
pathlib = pathlib
pretty = pretty
pytest = pytest
quote = quote
re = re
recursive_repr = recursive_repr
rmtree = rmtree
run_coroutine_threadsafe = run_coroutine_threadsafe
site = getsitepackages
sleep = sleep
spec_from_file_location = spec_from_file_location
subprocess = subprocess
suppress = suppress
sys = sys
sysconfig = sysconfig
textwrap = textwrap
to_thread = to_thread
tokenize = tokenize
urllib3 = urllib3
wrap_future = wrap_future


colorama.init()
pretty.install(console=console, expand_all=True)
# rich.traceback.install(console=console, extra_lines=5, show_locals=True)
urllib3.disable_warnings()
os.environ['PYTHONWARNINGS'] = 'ignore'


if __name__ == '__main__':
    typer.run(main)
