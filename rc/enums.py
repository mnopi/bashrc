# -*- coding: utf-8 -*-
"""Enums Module."""
__all__ = (
    'Bump',
    'PathIs',
    'FrameID',
    'PathMode',
    'PathOption',
    'PathOutput',
    'PathSuffix',
)

import collections
import enum
from asyncio import get_running_loop
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from typing import Any
from typing import NamedTuple
from typing import Optional
from typing import Union

import box

from .enumdict import Enum


class Bump(str, enum.Enum):
    MAJOR = 'major'
    MINOR = 'minor'
    PATCH = 'patch'
    PRERELEASE = 'prerelease'
    BUILD = 'build'


class ChainRV(enum.Enum):
    ALL = enum.auto()
    FIRST = enum.auto()
    UNIQUE = enum.auto()


class Executor(enum.Enum):
    PROCESS = ProcessPoolExecutor
    THREAD = ThreadPoolExecutor
    NONE = None

    async def run(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Run in :lib:func:`loop.run_in_executor` with :class:`concurrent.futures.ThreadPoolExecutor`,
            :class:`concurrent.futures.ProcessPoolExecutor` or
            :lib:func:`asyncio.get_running_loop().loop.run_in_executor` or not poll.

        Args:
            func: func
            *args: args
            **kwargs: kwargs

        Raises:
            ValueError: ValueError

        Returns:
            Awaitable:
        """
        loop = get_running_loop()
        call = partial(func, *args, **kwargs)
        if not func:
            raise ValueError

        if self.value:
            with self.value() as p:
                return await loop.run_in_executor(p, call)
        return await loop.run_in_executor(self.value, call)


_FrameID = NamedTuple('_FrameID', code=str, decorator=Union[list, str], function=str, parts=Union[list, str],
                      real=Optional[int], sync=bool)


class FrameID(enum.Enum):
    ASYNCCONTEXTMANAGER = _FrameID(code=str(), decorator=str(),
                                   function='__aenter__', parts='contextlib.py', real=1, sync=False)
    RUN = _FrameID(code=str(), decorator=str(),
                   function='_run', parts='asyncio events.py', real=5, sync=False)
    TO_THREAD = _FrameID(code=str(), decorator=str(),
                         function='run', parts='concurrent futures thread.py', real=None, sync=True)
    # FUNCDISPATCH = ('return funcs[Call().sync]', 'wrapper bapy', 'core', 1)


class PathIs(Enum):
    DIR = 'is_dir'
    FILE = 'is_file'


class PathMode(Enum):
    DIR = 0o666
    FILE = 0o777
    X = 0o755


class PathOption(Enum):
    BOTH = enum.auto()
    DIRS = enum.auto()
    FILES = enum.auto()


class PathOutput(Enum):
    BOTH = 'both'
    BOX = box.Box
    DICT = dict
    LIST = list
    NAMED = collections.namedtuple
    TUPLE = tuple


class PathSuffix(Enum):
    NO = str()
    BASH = enum.auto()
    ENV = enum.auto()
    GIT = enum.auto()
    INI = enum.auto()
    J2 = enum.auto()
    JINJA2 = enum.auto()
    LOG = enum.auto()
    MONGO = enum.auto()
    OUT = enum.auto()
    PY = enum.auto()
    RLOG = enum.auto()
    SH = enum.auto()
    TOML = enum.auto()
    YAML = enum.auto()
    YML = enum.auto()
