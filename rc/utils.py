# -*- coding: utf-8 -*-
"""Utils Module."""
import dataclasses
import pathlib
import re
import subprocess
import sys
import textwrap
from asyncio import current_task
from asyncio import get_running_loop
from asyncio.events import _RunningLoop
from collections import OrderedDict
from contextlib import suppress
from functools import cache
from inspect import getmro
from inspect import stack
from os import getenv
from pprint import pformat
from reprlib import recursive_repr
from subprocess import CompletedProcess
from sysconfig import get_paths
from types import SimpleNamespace as Simple
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Optional
from typing import Union

from devtools import Debug
from icecream import IceCreamDebugger

from pydantic import BaseModel

from .exceptions import *

__all__ = (
    'Simple',
    'NEWLINE',
    'POST_INIT_NAME',
    'PYTHON_SYS',
    'PYTHON_SITE',
    'SUDO_USER',
    'SUDO',

    'debug',
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
    'del_key',
    'dict_sort',
    'file_include',
    'join_new',
    'prefixed',
    'repr_format',
    'slots',
    'split_sep',
    'sudo',
    'to_iter',
    'varname',
)

NEWLINE = '\n'
POST_INIT_NAME = dataclasses._POST_INIT_NAME
PYTHON_SYS = sys.executable
PYTHON_SITE = str(pathlib.Path(PYTHON_SYS).resolve())
SUDO_USER = getenv('SUDO_USER')
SUDO = bool(SUDO_USER)

debug = Debug(highlight=True)
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
        debug(self)

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


def del_key(data: Union[dict, list], key: Iterable = ('self', 'cls', )) -> Union[dict, list]:
    rv = data
    key = to_iter(key)
    if isinstance(data, dict):
        rv = data.copy()
        for item in key:
            with suppress(KeyError):
                del rv[item]
    elif isinstance(data, list):
        for item in key:
            with suppress(ValueError):
                data.remove(item)
        rv = data
    return rv


def dict_sort(data: dict, ordered: bool = False, reverse: bool = False) -> Union[dict, OrderedDict]:
    """
    Order a dict based on keys.

    Args:
        data: dict to be ordered.
        ordered: OrderedDict.
        reverse: reverse.

    Returns:
        Union[dict, collections.OrderedDict]:
    """
    rv = {key: data[key] for key in sorted(data.keys(), reverse=reverse)}
    if ordered:
        return OrderedDict(rv)
    return rv.copy()




def join_new(data: list) -> str:
    return NEWLINE.join(data)


def prefixed(name: str) -> str:
    try:
        return f'{name.upper()}_'
    except AttributeError:
        pass


def repr_format(obj: Any, attrs: Iterable, clear: bool = True, newline: bool = False):
    cls = obj.__class__
    if clear:
        for item in dir(cls):
            if (attr := getattr(cls, item, None)) and (c := getattr(attr, 'cache_clear', None)):
                # noinspection PyUnboundLocalVariable
                c()
    new = NEWLINE if newline else str()
    msg = f',{new if newline else " "}'.join([f"{arg}: {repr(getattr(obj, arg))}" for arg in to_iter(attrs)])
    return f'{cls.__name__}({new}{msg}{new})'


class slots:
    """
    Slots Repr Helper Class
    """
    __slots__ = ('_slots', )

    def __init__(self):
        self._slots = sorted({attr for item in getmro(self.__class__) for attr in getattr(item, "__slots__", list())
                              if attr != slots.__slots__[0]})
        for attr in self._slots:
            self.__setattr__(attr, None)

    @recursive_repr()
    def __repr__(self):
        values = {name: getattr(self, name) for name in self._slots}
        return f'{self.__class__.__name__}({", ".join([slot + ":" + repr(value) for slot, value in values.items()])})'


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
