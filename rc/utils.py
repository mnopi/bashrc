# -*- coding: utf-8 -*-
"""Utils Module."""
import ast
import dataclasses
import pathlib
import re
import subprocess
import sys
import textwrap
import tokenize
from asyncio import current_task
from asyncio import get_running_loop
from asyncio.events import _RunningLoop
from contextlib import suppress
from functools import cache
from inspect import findsource
from inspect import getmro
from inspect import stack
from os import getenv
from pprint import pformat
from reprlib import recursive_repr
from subprocess import CompletedProcess
from sysconfig import get_paths
from types import FrameType
from types import SimpleNamespace as Simple
from typing import Any
from typing import Callable
from typing import Iterable
from typing import NamedTuple
from typing import Optional
from typing import Union

from devtools import Debug
from icecream import IceCreamDebugger
from intervaltree import Interval
from intervaltree import IntervalTree
from pydantic import BaseModel

from .exceptions import *

__all__ = (
    'Simple',
    'POST_INIT_NAME',
    'PYTHON_SYS',
    'PYTHON_SITE',
    'SUDO_USER',
    'SUDO',
    'SYS_PATHS',
    'SYS_PATHS_EXCLUDE',

    'ddebug',
    'fm',
    'fmic',
    'fmicc',
    'ic',
    'icc',

    'aioloop',
    '_base',
    'basemodel',
    'cmd',
    'cmdname',
    'current_task_name',
    'include_file',
    'prefixed',
    'split_sep',
    'slots',
    'sudo',
    'to_iter',
    'varname',
)

POST_INIT_NAME = dataclasses._POST_INIT_NAME
PYTHON_SYS = sys.executable
PYTHON_SITE = str(pathlib.Path(PYTHON_SYS).resolve())
SUDO_USER = getenv('SUDO_USER')
SUDO = bool(SUDO_USER)
SYS_PATHS = get_paths()
SYS_PATHS_EXCLUDE = (SYS_PATHS['stdlib'], SYS_PATHS['purelib'], SYS_PATHS['include'], SYS_PATHS['platinclude'],
                     SYS_PATHS['scripts'])

ddebug = Debug(highlight=True)
fm = pformat
fmic = IceCreamDebugger(prefix=str()).format
fmicc = IceCreamDebugger(prefix=str(), includeContext=True).format
ic = IceCreamDebugger(prefix=str())
icc = IceCreamDebugger(prefix=str(), includeContext=True)


def aioloop() -> Optional[_RunningLoop]:
    try:
        return get_running_loop()
    except RuntimeError:
        return None


class _base:
    @classmethod
    @cache
    def clsname(cls, lower: bool = False) -> str:
        return cls.__name__.lower() if lower else cls.__name__

    @classmethod
    @cache
    def clsqual(cls, lower: bool = False) -> str:
        return cls.__qualname__.lower() if lower else cls.__qualname__

    def debug(self):
        ddebug(self)

    def fmic(self):
        fmic(self)

    def fmicc(self):
        fmicc(self)

    def ic(self):
        ic(self)

    def icc(self):
        icc(self)


class basemodel(BaseModel, _base):
    pass


def cmd(command: Iterable, exc: bool = False, lines: bool = True, shell: bool = True,
        py: bool = False, pysite: bool = True) -> Union[CompletedProcess, int, list, str]:
    """
    Runs a cmd.

    Examples:
        >>> cmd('ls a')
        Cmd(stdout=[], stderr=['ls: a: No such file or directory'], rc=1)
        >>> assert 'Requirement already satisfied' in cmd('pip install pip', py=True)[0][0]
        >>> cmd('ls a', shell=False, lines=False)  # Extra '\' added to avoid docstring error.
        Cmd(stdout='', stderr='ls: a: No such file or directory\\n', rc=1)
        >>> cmd('echo a', lines=False)  # Extra '\' added to avoid docstring error.
        Cmd(stdout='a\\n', stderr='', rc=0)

    Args:
        command: command.
        exc: raise exception.
        lines: split lines so ``\\n`` is removed from all lines (extra '\' added to avoid docstring error).
        py: runs with python executable.
        shell: expands shell variables and one line (shell True expands variables in shell).
        pysite: run on site python if running on a VENV.

    Raises:
        CmdError:

    Returns:
        Union[Cmd, int, list, str]:
    """
    if py:
        m = '-m'
        if isinstance(command, str) and command.startswith('/'):
            m = str()
        command = f'{PYTHON_SITE if pysite else PYTHON_SYS} {m} {command}'
    elif not shell:
        command = to_iter(command)

    if lines:
        text = False
    else:
        text = True

    proc = subprocess.run(command, shell=shell, capture_output=True, text=text)

    def std(out=True):
        if out:
            if lines:
                return proc.stdout.decode("utf-8").splitlines()
            else:
                # return proc.stdout.rstrip('.\n')
                return proc.stdout
        else:
            if lines:
                return proc.stderr.decode("utf-8").splitlines()
            else:
                # return proc.stderr.decode("utf-8").rstrip('.\n')
                return proc.stderr

    rv = CompletedProcess(proc.args, proc.returncode, std(), std(False))
    if rv.returncode != 0 and exc:
        raise CmdError(rv)
    return rv


def cmdname(func: Callable, sep: str = '_') -> str:
    """Command name."""
    return func.__name__.split(**split_sep(sep))[0]


def current_task_name() -> str:
    return current_task().get_name() if aioloop() else str()


def _file_to_tree_compute_interval(node):
    min_lineno = node.lineno
    max_lineno = node.lineno
    for node in ast.walk(node):
        if hasattr(node, "lineno"):
            min_lineno = min(min_lineno, node.lineno)
            max_lineno = max(max_lineno, node.lineno)
    return min_lineno, max_lineno + 1


def file_to_tree(filename) -> IntervalTree[Interval, ...]:
    with tokenize.open(filename) as f:
        parsed = ast.parse(f.read(), filename=filename)
    tree = IntervalTree()
    for node in ast.walk(parsed):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start, end = _file_to_tree_compute_interval(node)
            tree[start:end] = node
    return tree


FuncCodeSync = NamedTuple('FuncCodeSync', code=list, file=str, func=str, name=str, sync=bool)


def func_code(frame: FrameType, file: Any, func: str = None) -> FuncCodeSync:
    search = ('await', 'asyncio.run', 'async', 'as_completed', 'create_task', )
    lines, start = findsource(frame)
    tree = file_to_tree(file)
    names = list()
    for item in tree:
        names.append(item.data.name)
        if item.begin == frame.f_code.co_firstlineno:
            code = lines[start:item.end]
            if isinstance(item.data, ast.AsyncFunctionDef):
                sync = False
                rv = FuncCodeSync(code=code, file=file, func=func, name=item.data.name, sync=sync)
                icc(rv)
                return rv
            elif isinstance(item.data, ast.FunctionDef):
                sync = not any([item in line for line in code for item in search])
                rv = FuncCodeSync(code=code, file=file, func=func, name=item.data.name, sync=sync)
                icc(rv)
                return rv

    raise RuntimeError(f'Did not find a math in: {file}, for: {frame.f_code.co_firstlineno=}, in: {tree}, '
                       f'{names=}, {func=}')


def include_file(file: Any, scratches: bool = False) -> Optional[bool]:
    file = str(file)
    if scratches and 'scratches' in file:
        return True
    for f in SYS_PATHS_EXCLUDE:
        if file in f:
            return
    return True


def prefixed(name: str) -> str:
    try:
        return f'{name.upper()}_'
    except AttributeError:
        pass


class slots:
    """
    Slots Repr Helper Class
    """
    __slots__ = ()

    @recursive_repr()
    def __repr__(self):
        attrs = sorted({attr for item in getmro(self.__class__) for attr in getattr(item, "__slots__", list())})
        return f'{self.__class__.__name__}{", ".join(map(repr, map(self.__getattribute__,  attrs)))})'


def split_sep(sep: str = '_') -> dict:
    return dict(sep=sep) if sep else dict()


def sudo(command: str, su: bool = False) -> str:
    return command if SUDO or not su else f'sudo {command}'


def to_iter(data: Any) -> Iterable:
    if isinstance(data, str):
        return data.split(' ')
    elif isinstance(data, Iterable):
        return data
    else:
        return [data]


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
        _stack = stack()
        func = _stack[index - 1].function
        index = index + 1 if func == POST_INIT_NAME else index
        if line := textwrap.dedent(_stack[index].code_context[0]):
            if var := re.sub(f'(.| ){func}.*', str(), line.split(' = ')[0].replace('assert ', str()).split(' ')[0]):
                return (var.lower() if lower else var).split(**split_sep(sep))[0]
