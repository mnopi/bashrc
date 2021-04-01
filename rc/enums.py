# -*- coding: utf-8 -*-
"""Enums Module."""
__all__ = (
    'Bump',
    'CallerID',
    'ChainRV',
    'PathIs',
    'PathMode',
    'PathOption',
    'PathOutput',
    'PathSuffix',
)

import collections
import enum

import box

from .enumdict import Enum


class Bump(str, enum.Enum):
    MAJOR = 'major'
    MINOR = 'minor'
    PATCH = 'patch'
    PRERELEASE = 'prerelease'
    BUILD = 'build'


class CallerID(enum.Enum):
    TO_THREAD = ('result = self.fn', 'run', 'futures', 'thread', 4)  # No real.
    RUN = ('self._context.run', '_run', 'asyncio', 'events', 4)
    FUNCDISPATCH = ('return funcs[Call().sync]', 'wrapper', 'bapy', 'core', 1)


class ChainRV(enum.Enum):
    ALL = enum.auto()
    FIRST = enum.auto()
    UNIQUE = enum.auto()


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
