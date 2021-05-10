# -*- coding: utf-8 -*-
"""
Utils Module.

Examples:
    >>> from copy import deepcopy
    >>> import environs
    >>>
    >>> deepcopy(environs.Env()) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    RecursionError: maximum recursion depth exceeded
"""
__all__ = (
    # Imports: StdLib Modules
    'ast',
    'abc',
    'copy',
    'functools',
    'grp',
    'os',
    'pathlib',
    'pwd',
    're',
    'subprocess',
    'sys',
    'textwrap',
    'tokenize',
    'types',
    'warnings',

    # Imports: StdLib
    'ABCMeta',
    'abstractmethod',
    'AST',
    'AsyncFor',
    'AsyncFunctionDef',
    'AsyncWith',
    'Await',
    'ClassDef',
    'FunctionDef',
    'get_source_segment',
    'Import',
    'ImportFrom',
    'NodeVisitor',
    'walk',
    'all_tasks',
    'as_completed',
    'BaseEventLoop',
    'CancelledError',
    'Condition',
    'create_subprocess_exec',
    'create_subprocess_shell',
    'create_task',
    'current_task',
    'ensure_future',
    'Event',
    'gather',
    'get_event_loop',
    'get_running_loop',
    'Future',
    'iscoroutine',
    'isfuture',
    'AsyncLock',
    'AsyncQueue',
    'QueueEmpty',
    'QueueFull',
    'asyncrun',
    'run_coroutine_threadsafe',
    'Semaphore',
    'sleep',
    'AsyncTask',
    'to_thread',
    'ChainMap',
    'defaultdict',
    'namedtuple',
    'OrderedDict',
    'AsyncGenerator',
    'AsyncIterable',
    'AsyncIterator',
    'Awaitable',
    'ByteString',
    'Callable',
    'Collection',
    'Container',
    'Coroutine',
    'Generator',
    'Hashable',
    'ItemsView',
    'Iterable',
    'Iterator',
    'KeysView',
    'Mapping',
    'MappingView',
    'MutableMapping',
    'MutableSequence',
    'MutableSet',
    'Reversible',
    'Sequence',
    'Set',
    'Sized',
    'ValuesView',
    'ProcessPoolExecutor',
    'ThreadPoolExecutor',
    'contextmanager',
    'suppress',
    'dataclass',
    'DataField',
    'datafield',
    'datafields',
    'InitVar',
    'auto',
    'Enum',
    'EnumMeta',
    'cached_property',
    'partial',
    'singledispatch',
    'singledispatchmethod',
    'total_ordering',
    'wraps',
    'import_module',
    'module_from_spec',
    'spec_from_file_location',
    'classify_class_attrs',
    'FrameInfo',
    'getfile',
    'getsource',
    'getsourcefile',
    'getsourcelines',
    'isasyncgenfunction',
    'isawaitable',
    'iscoroutinefunction',
    'isgetsetdescriptor',
    'ismemberdescriptor',
    'ismethoddescriptor',
    'isroutine',
    'insstack',
    'BytesIO',
    'FileIO',
    'StringIO',
    'addLevelName',
    'basicConfig',
    'CRITICAL',
    'DEBUG',
    'ERROR',
    'FileHandler',
    'Formatter',
    'getLevelName',
    'getLogger',
    'getLoggerClass',
    'Handler',
    'INFO',
    'Logger',
    'LoggerAdapter',
    'NOTSET',
    'setLoggerClass',
    'StreamHandler',
    'WARNING',
    'RotatingFileHandler',
    'attrgetter',
    'chdir',
    'environ',
    'getenv',
    'PathLike',
    'system',
    'PathLib',
    'pickle_dump',
    'pickle_dumps',
    'pickle_load',
    'pickle_loads',
    'PicklingError',
    'shquote',
    'shsplit',
    'rmtree',
    'getsitepackages',
    'USER_SITE',
    'CompletedProcess',
    'subrun',
    'TemporaryDirectory',
    'CRLock',
    'Lock',
    'timeit',
    'AsyncGeneratorType',
    'BuiltinFunctionType',
    'ClassMethodDescriptorType',
    'CodeType',
    'CoroutineType',
    'FrameType',
    'FunctionType',
    'GeneratorType',
    'GenericAlias',
    'GetSetDescriptorType',
    'LambdaType',
    'MappingProxyType',
    'MemberDescriptorType',
    'MethodType',
    'MethodWrapperType',
    'ModuleType',
    'Simple',
    'TracebackType',
    'WrapperDescriptorType',
    'Alias',
    'catch_warnings',
    'filterwarnings',

    # Imports: PyPi - Modules
    'colorama',
    'executing',
    'pickle_np',
    'np',
    'paramiko',
    'pd',

    # Imports: PyPi
    'astprint',
    'astformat',
    'ASTTokens',
    'Box',
    'ObjectId',
    'secho',
    'Exit',
    'ColoredFormatter',
    'LevelFormatter',
    'decorator',
    'Debug',
    'dpathdelete',
    'dpathget',
    'dpathnew',
    'dpathset',
    'dpathsearch',
    'dpathvalues',
    'Environs',
    'Executing',
    'furl',
    'GitRepo',
    'GitSymbolicReference',
    'IceCreamDebugger',
    'Template',
    'Pickler',
    'Unpickler',
    'importable_name',
    'is_collections',
    'is_installed',
    'is_module_function',
    'is_noncomplex',
    'is_object',
    'is_primitive',
    'is_reducible',
    'is_reducible_sequence_subclass',
    'is_reducible_sequence_subclass',
    'is_unicode',
    'collapse',
    'consume',
    'first_true',
    'map_reduce',
    'side_effect',
    'nested_lookup',
    'AuthenticationException',
    'AutoAddPolicy',
    'BadAuthenticationType',
    'BadHostKeyException',
    'PasswordRequiredException',
    'SSHClient',
    'SSHConfig',
    'SSHException',
    'MACOS',
    'Console',
    'RichHandler',
    'pretty_install',
    'traceback_install',
    'SetUpToolsDistribution',
    'find_packages',
    'SetUpToolsInstall',
    'LazyCache',
    'memoize',
    'disable_warnings',
    'var_name',
    'NOTICE',
    'SPAM',
    'SUCCESS',
    'VERBOSE',
    'VerboseLogger',

    # Imports - Protected
    'RunningLoop',
    'DATACLASS_FIELDS',
    'MISSING_TYPE',
    'POST_INIT_NAME',
    'CRLock',
    'Alias',

    # Typing
    'DictStrAny',
    'ExceptionUnion',
    'LST',
    'SeqNoStr',
    'SeqUnion',
    'TupleStr',
    'TupleType',

    # Constants
    'AUTHORIZED_KEYS',
    'BUILTINS',
    'BUILTINS_CLASSES',
    'BUILTINS_FUNCTIONS',
    'BUILTINS_OTHER',
    'console',
    'cp',
    'DATACLASS_FIELDS',
    'debug',
    'FILE_DEFAULT',
    'fmic',
    'fmicc',
    'FRAME_SYS_INIT',
    'FUNCTION_MODULE',
    'GITCONFIG',
    'GITHUB_ORGANIZATION',
    'ID_RSA',
    'ID_RSA_PUB',
    'ic',
    'icc',
    'IgnoreAttr',
    'IgnoreCopy',
    'IgnoreStr',
    'NEWLINE',
    'SSH_CONFIG',
    'SSH_CONFIG_TEXT',
    'SSH_DIR',
    'STATE_ATTRS',
    'SUDO_USER',
    'SUDO',
    'SUDO_DEFAULT',
    'print_exception',
    'PYTHON_SYS',
    'PYTHON_SITE',

    # EnumBase
    'EnumBase',
    'EnumBaseAlias',

    # Classes: Enums, Named and No Deps
    'AccessMeta',
    'Access',
    'AccessEnumMembers',
    'Annotation',
    'Attribute',
    'BaseState',
    'BoxKeys',
    'CacheWrapperInfo',
    'ChainRV',
    'Chain',
    'Executor',
    'FindUp',
    'FrameBase',
    'Frame',
    'FrameSourceNode',
    'GitTop',
    'Is',
    'Kind',
    'ModuleBase',
    'Module',
    'NBase',
    'N',
    'PathGit',
    'PathInstallScript',
    'PathIs',
    'PathMode',
    'PathOption',
    'PathOutput',
    'PathSuffix',
    'Path',
    'Re',
    'Source',
    'SourceNode',
    'UserActual',
    'UserProcess',
    'User',

    # Classes: Deps
    'Attributes',
    'Es',
    'PathLikeStr',
    'StateEnv',

    # Functions
    'aioloop',
    'allin',
    'annotations',
    'annotations_init',
    'anyin',
    'cache',
    'cmd',
    'cmdname',
    'current_task_name',
    'delete',
    'delete_list',
    'dict_sort',
    'effect',
    'enumvalue',
    'get',
    'getnostr',
    'getset',
    'iseven',
    'in_dict',
    'join_newline',
    'map_reduce_even',
    'map_with_args',
    'newprop',
    'noexception',
    'prefixed',
    'repr_format',
    'runwarning',
    'split_sep',
    'startswith',
    'to_camel',
    'to_iter',
    'token_open',
    'varname',
    'yield_if',
    'yield_last',

    # Exceptions
    'CmdError',
    'CmdAioError',

    # Types
    'AnnotationsType',
    'AsDictMethodType',
    'AsDictPropertyType',
    'DataType',
    'DictType',
    'GetAttrNoBuiltinType',
    'GetAttrType',
    'GetSupportType',
    'GetType',
    'NamedType',
    'NamedAnnotationsType',
    'SlotsType',

    # Echo
    'black',
    'blue',
    'cyan',
    'green',
    'magenta',
    'red',
    'white',
    'yellow',
    'bblack',
    'bblue',
    'bcyan',
    'bgreen',
    'bmagenta',
    'bred',
    'bwhite',
    'byellow',

    # Test
    'TestAsync',
    'TestData',
    'TestDataDictMix',
    'TestDataDictSlotMix',

    # Instances
    'user',
)

import ast as ast
import collections.abc as abc
import copy as copy
import functools as functools
# noinspection PyCompatibility
import grp as grp
import os as os
import pathlib as pathlib
# noinspection PyCompatibility
import pwd as pwd
import re as re
import subprocess as subprocess
import sys as sys
import textwrap as textwrap
import tokenize as tokenize
import types as types
import typing
import warnings as warnings
from abc import ABCMeta as ABCMeta
from abc import abstractmethod as abstractmethod
from ast import AST as AST
from ast import AsyncFor as AsyncFor
from ast import AsyncFunctionDef as AsyncFunctionDef
from ast import AsyncWith as AsyncWith
from ast import Await as Await
from ast import ClassDef as ClassDef
from ast import FunctionDef as FunctionDef
from ast import get_source_segment as get_source_segment
from ast import Import as Import
from ast import ImportFrom as ImportFrom
from ast import NodeVisitor as NodeVisitor
from ast import walk as walk
from asyncio import all_tasks as all_tasks
from asyncio import as_completed as as_completed
from asyncio import BaseEventLoop as BaseEventLoop
from asyncio import CancelledError as CancelledError
from asyncio import Condition as Condition
from asyncio import create_subprocess_exec as create_subprocess_exec
from asyncio import create_subprocess_shell as create_subprocess_shell
from asyncio import create_task as create_task
from asyncio import current_task as current_task
from asyncio import ensure_future as ensure_future
from asyncio import Event as Event
from asyncio import gather as gather
from asyncio import get_event_loop as get_event_loop
from asyncio import get_running_loop as get_running_loop
from asyncio import Future as Future
from asyncio import iscoroutine as iscoroutine
from asyncio import isfuture as isfuture
from asyncio import Lock as AsyncLock
from asyncio import Queue as AsyncQueue
from asyncio import QueueEmpty as QueueEmpty
from asyncio import QueueFull as QueueFull
from asyncio import run as asyncrun
from asyncio import run_coroutine_threadsafe as run_coroutine_threadsafe
from asyncio import Semaphore as Semaphore
from asyncio import sleep as sleep
from asyncio import Task as AsyncTask
from asyncio import to_thread as to_thread
from asyncio.events import _RunningLoop
from collections import ChainMap as ChainMap
from collections import defaultdict as defaultdict
from collections import namedtuple as namedtuple
from collections import OrderedDict as OrderedDict
from collections.abc import AsyncGenerator as AsyncGenerator
from collections.abc import AsyncIterable as AsyncIterable
from collections.abc import AsyncIterator as AsyncIterator
from collections.abc import Awaitable as Awaitable
from collections.abc import ByteString as ByteString
from collections.abc import Callable as Callable
from collections.abc import Collection as Collection
from collections.abc import Container as Container
from collections.abc import Coroutine as Coroutine
from collections.abc import Generator as Generator
from collections.abc import Hashable as Hashable
from collections.abc import ItemsView as ItemsView
from collections.abc import Iterable as Iterable
from collections.abc import Iterator as Iterator
from collections.abc import KeysView as KeysView
from collections.abc import Mapping as Mapping
from collections.abc import MappingView as MappingView
from collections.abc import MutableMapping as MutableMapping
from collections.abc import MutableSequence as MutableSequence
from collections.abc import MutableSet as MutableSet
from collections.abc import Reversible as Reversible
from collections.abc import Sequence as Sequence
from collections.abc import Set as Set
from collections.abc import Sized as Sized
from collections.abc import ValuesView as ValuesView
from concurrent.futures.process import ProcessPoolExecutor as ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor as ThreadPoolExecutor
from contextlib import contextmanager as contextmanager
from contextlib import suppress as suppress
from dataclasses import _FIELDS
from dataclasses import _MISSING_TYPE
from dataclasses import _POST_INIT_NAME
from dataclasses import dataclass as dataclass
from dataclasses import Field as DataField
from dataclasses import field as datafield
from dataclasses import fields as datafields
from dataclasses import InitVar as InitVar
from enum import auto as auto
from enum import Enum as Enum
from enum import EnumMeta as EnumMeta
from functools import cached_property as cached_property
from functools import partial as partial
from functools import singledispatch as singledispatch
from functools import singledispatchmethod as singledispatchmethod
from functools import total_ordering as total_ordering
from functools import wraps as wraps
from importlib import import_module as import_module
from importlib.util import module_from_spec as module_from_spec
from importlib.util import spec_from_file_location as spec_from_file_location
from inspect import classify_class_attrs as classify_class_attrs
from inspect import FrameInfo as FrameInfo
from inspect import getfile as getfile
from inspect import getsource as getsource
from inspect import getsourcefile as getsourcefile
from inspect import getsourcelines as getsourcelines
from inspect import isasyncgenfunction as isasyncgenfunction
from inspect import isawaitable as isawaitable
from inspect import iscoroutinefunction as iscoroutinefunction
from inspect import isgetsetdescriptor as isgetsetdescriptor
from inspect import ismemberdescriptor as ismemberdescriptor
from inspect import ismethoddescriptor as ismethoddescriptor
from inspect import isroutine as isroutine
from inspect import stack as insstack
from io import BytesIO as BytesIO
from io import FileIO as FileIO
from io import StringIO as StringIO
from logging import addLevelName as addLevelName
from logging import basicConfig as basicConfig
from logging import CRITICAL as CRITICAL
from logging import DEBUG as DEBUG
from logging import ERROR as ERROR
from logging import FileHandler as FileHandler
from logging import Formatter as Formatter
from logging import getLevelName as getLevelName
from logging import getLogger as getLogger
from logging import getLoggerClass as getLoggerClass
from logging import Handler as Handler
from logging import INFO as INFO
from logging import Logger as Logger
from logging import LoggerAdapter as LoggerAdapter
from logging import NOTSET as NOTSET
from logging import setLoggerClass as setLoggerClass
from logging import StreamHandler as StreamHandler
from logging import WARNING as WARNING
from logging.handlers import RotatingFileHandler as RotatingFileHandler
from operator import attrgetter as attrgetter
from os import chdir as chdir
from os import environ as environ
from os import getenv as getenv
from os import PathLike as PathLike
from os import system as system
from pathlib import Path as PathLib
from pickle import dump as pickle_dump
from pickle import dumps as pickle_dumps
from pickle import load as pickle_load
from pickle import loads as pickle_loads
from pickle import PicklingError as PicklingError
from shlex import quote as shquote
from shlex import split as shsplit
from shutil import rmtree as rmtree
from site import getsitepackages as getsitepackages
from site import USER_SITE as USER_SITE
from subprocess import CompletedProcess as CompletedProcess
from subprocess import run as subrun
from tempfile import TemporaryDirectory as TemporaryDirectory
from threading import _CRLock
from threading import Lock as Lock
from timeit import timeit as timeit
from types import AsyncGeneratorType as AsyncGeneratorType
from types import BuiltinFunctionType as BuiltinFunctionType
from types import ClassMethodDescriptorType as ClassMethodDescriptorType
from types import CodeType as CodeType
from types import CoroutineType as CoroutineType
from types import DynamicClassAttribute as DynamicClassAttribute
from types import FrameType as FrameType
from types import FunctionType as FunctionType
from types import GeneratorType as GeneratorType
from types import GenericAlias as GenericAlias
from types import GetSetDescriptorType as GetSetDescriptorType
from types import LambdaType as LambdaType
from types import MappingProxyType as MappingProxyType
from types import MemberDescriptorType as MemberDescriptorType
from types import MethodType as MethodType
from types import MethodWrapperType as MethodWrapperType
from types import ModuleType as ModuleType
from types import SimpleNamespace as Simple
from types import TracebackType as TracebackType
from types import WrapperDescriptorType as WrapperDescriptorType
from typing import _alias
from typing import Any
from typing import cast
from typing import ClassVar
from typing import Final
from typing import get_args
from typing import get_origin
from typing import get_type_hints
from typing import Literal
from typing import NamedTuple
from typing import Optional
from typing import Protocol
from typing import runtime_checkable
from typing import Type
from typing import Union
from warnings import catch_warnings as catch_warnings
from warnings import filterwarnings as filterwarnings

import colorama as colorama
import executing as executing
import jsonpickle.ext.numpy as pickle_np
import numpy as np
import pandas as pd
import paramiko as paramiko
from astpretty import pformat as astformat
from astpretty import pprint as astprint
from asttokens import ASTTokens as ASTTokens
from box import Box as Box
from bson import ObjectId as ObjectId
from click import secho as secho
from click.exceptions import Exit as Exit
from colorlog import ColoredFormatter as ColoredFormatter
from colorlog import LevelFormatter as LevelFormatter
from decorator import decorator as decorator
from devtools import Debug as Debug
from dpath.util import delete as dpathdelete
from dpath.util import get as dpathget
from dpath.util import new as dpathnew
from dpath.util import search as dpathsearch
from dpath.util import set as dpathset
from dpath.util import values as dpathvalues
from environs import Env as Environs
from executing import Executing as Executing
from furl import furl as furl
from git import GitConfigParser as GitConfigParser
from git import Remote as Remote
from git import Repo as GitRepo
from git.refs import SymbolicReference as GitSymbolicReference
from icecream import IceCreamDebugger as IceCreamDebugger
from jinja2 import Template as Template
from jsonpickle.pickler import Pickler as Pickler
from jsonpickle.unpickler import Unpickler as Unpickler
from jsonpickle.util import importable_name as importable_name
from jsonpickle.util import is_collections as is_collections
from jsonpickle.util import is_installed as is_installed
from jsonpickle.util import is_module_function as is_module_function
from jsonpickle.util import is_noncomplex as is_noncomplex
from jsonpickle.util import is_object as is_object
from jsonpickle.util import is_primitive as is_primitive
from jsonpickle.util import is_reducible as is_reducible
from jsonpickle.util import is_reducible_sequence_subclass as is_reducible_sequence_subclass
from jsonpickle.util import is_unicode as is_unicode
from more_itertools import collapse as collapse
from more_itertools import consume as consume
from more_itertools import first_true as first_true
from more_itertools import map_reduce as map_reduce
from more_itertools import side_effect as side_effect
from nested_lookup import nested_lookup as nested_lookup
from nested_lookup.nested_lookup import _nested_lookup
from paramiko import AuthenticationException as AuthenticationException
from paramiko import AutoAddPolicy as AutoAddPolicy
from paramiko import BadAuthenticationType as BadAuthenticationType
from paramiko import BadHostKeyException as BadHostKeyException
from paramiko import PasswordRequiredException as PasswordRequiredException
from paramiko import SSHClient as SSHClient
from paramiko import SSHConfig as SSHConfig
from paramiko import SSHException as SSHException
from psutil import MACOS as MACOS
from rich.console import Console as Console
from rich.logging import RichHandler as RichHandler
from rich.pretty import install as pretty_install
from rich.traceback import install as traceback_install
from setuptools import find_packages as find_packages
from setuptools import Distribution as SetUpToolsDistribution
from setuptools.command.install import install as SetUpToolsInstall
from thefuck.utils import Cache as LazyCache
from thefuck.utils import memoize as memoize
from urllib3 import disable_warnings as disable_warnings
from varname import varname as var_name
from verboselogs import NOTICE as NOTICE
from verboselogs import SPAM as SPAM
from verboselogs import SUCCESS as SUCCESS
from verboselogs import VERBOSE as VERBOSE
from verboselogs import VerboseLogger as VerboseLogger

# <editor-fold desc="Protected">
RunningLoop = _RunningLoop
DATACLASS_FIELDS = _FIELDS
MISSING_TYPE = _MISSING_TYPE
POST_INIT_NAME = _POST_INIT_NAME
CRLock = _CRLock
Alias = _alias

# </editor-fold>
# <editor-fold desc="Typing">
DictStrAny = dict[str, Any]
ExceptionUnion = Union[tuple[Type[Exception]], Type[Exception]]
LST = Union[typing.MutableSequence, typing.MutableSet, tuple]
SeqNoStr = Union[typing.Iterator, typing.KeysView, typing.MutableSequence, typing.MutableSet, tuple, typing.ValuesView]
SeqUnion = Union[typing.AnyStr, typing.ByteString, typing.Iterator, typing.KeysView, typing.MutableSequence,
                 typing.MutableSet, typing.Sequence, tuple, typing.ValuesView]
TupleStr = tuple[str, ...]
TupleType = tuple[Type, ...]

# </editor-fold>
# <editor-fold desc="Constants">
AUTHORIZED_KEYS = 'authorized_keys'
BUILTINS = (_bltns if isinstance(_bltns := globals()['__builtins__'], dict) else vars(_bltns)).copy()
BUILTINS_CLASSES = tuple(filter(lambda x: isinstance(x, type), BUILTINS.values()))
BUILTINS_FUNCTIONS = tuple(filter(lambda x: isinstance(x, (BuiltinFunctionType, FunctionType,)), BUILTINS.values()))
BUILTINS_OTHER = tuple(map(BUILTINS.get, ('__doc__', '__import__', '__spec__', 'copyright', 'credits', 'exit',
                                          'help', 'license', 'quit',)))
console = Console(color_system='256')
cp = console.print
debug = Debug(highlight=True)
FILE_DEFAULT = True
fmic = IceCreamDebugger(prefix=str()).format
fmicc = IceCreamDebugger(prefix=str(), includeContext=True).format
FRAME_SYS_INIT = sys._getframe(0)
FUNCTION_MODULE = '<module>'
GITCONFIG = '.gitconfig'
GITHUB_ORGANIZATION = getenv('GITHUB_ORGANIZATION')
ID_RSA = 'id_rsa'
ID_RSA_PUB = 'id_rsa.pub'
ic = IceCreamDebugger(prefix=str())
icc = IceCreamDebugger(prefix=str(), includeContext=True)
IgnoreAttr = Literal['asdict', 'attrs', 'defaults', 'keys', 'kwargs', 'kwargs_dict', 'public', 'values', 'values_dict']
"""Exclude instance attribute."""
IgnoreCopy = Union[CRLock, Environs, FrameType, GitConfigParser, GitSymbolicReference, Remote]
"""True or class for repr instead of nested asdict and deepcopy. No deepcopy (default: (:class:`rc.CRLock`,
:class:`rc.Environs`, :class:`types.FrameType`, :class:`git.GitConfigParser`, :class:`rc.GitSymbolicReference`,
:class:`git.Remote`, ))."""
IgnoreStr = Union[GitConfigParser, GitRepo, ObjectId, PathLib]
"""Use str value for object (default: (:class:`git.GitConfigParser`, :class:`rc.GitRepo`, :class:`bson.ObjectId`,
:class:`rc.PathLib`, ))."""
LockClass = type(Lock())
NEWLINE = '\n'
print_exception = console.print_exception
PYTHON_SYS = PathLib(sys.executable)
PYTHON_SITE = PathLib(PYTHON_SYS).resolve()
SSH_CONFIG = dict(AddressFamily='inet', BatchMode='yes', CheckHostIP='no', ControlMaster='auto',
                  ControlPath='/tmp/ssh-%h-%r-%p', ControlPersist='20m', IdentitiesOnly='yes', LogLevel='QUIET',
                  StrictHostKeyChecking='no', UserKnownHostsFile='/dev/null')
SSH_CONFIG_TEXT = ' '.join([f'-o {key}={value}' for key, value in SSH_CONFIG.items()])
SSH_DIR = '.ssh'
STATE_ATTRS = {
    Environs: ('__custom_parsers__', '_errors', '_fields', '_prefix', '_sealed', '_values', 'eager', 'expand_vars',),
}
SUDO_USER = getenv('SUDO_USER')
SUDO = bool(SUDO_USER)
SUDO_DEFAULT = True


# </editor-fold>
# <editor-fold desc="EnumBase">
@total_ordering
class EnumBase(Enum):
    """Enum Base Class."""
    def __eq__(self, other):
        """
        Equal Using Enum Key and Enum Instance.

        Examples:
            >>> from rc import Access, pretty_install
            >>> pretty_install()
            >>>
            >>> Access['field'] == Access.PUBLIC
            True
            >>> Access['field'] is Access.PUBLIC
            True
            >>> Access['field'] == '_field'
            False
            >>> Access['_field'] == '_field'
            True
            >>> Access['_field'] is '_field'
            False

        Returns:
            True if Enum Key or Enum Instance are equal to self.
        """
        if rv := type(self)[other]:
            return self._name_ == rv._name_ and self._value_ == rv._value_
        return NotImplemented

    def __gt__(self, other):
        """
        Greater Than Using Enum Key and Enum Instance.

        Examples:
            >>> from rc import Access, pretty_install
            >>> pretty_install()
            >>>
            >>> Access['field']
            <Access.PUBLIC: re.compile('^(?!_)..*$')>
            >>> Access['field'] > '_field'
            True
            >>> Access['field'] > Access.ALL
            True
            >>> Access['field'] > 'field'
            False
            >>> Access['field'] > Access.PUBLIC
            False
            >>>
            >>> Access['field'] < Access.ALL
            False
            >>>
            >>> Access.PROTECTED >= Access.ALL
            True
            >>> Access['field'] >= Access.ALL
            True
            >>> Access['field'] >= '_field'
            True

        Returns:
            True self (index/int) is greater than other Enum Key or Enum Instance.
        """

        if rv := type(self)[other]:
            return self.__int__() > rv.__int__()
        return NotImplemented

    def __hash__(self): return hash(self._name_)

    def __int__(self):
        """
        int based on index to compare.

        Examples:
            >>> from rc import Access, pretty_install
            >>> pretty_install()
            >>>
            >>> int(Access.PROTECTED)
            2

        Returns:
            Index.
        """
        return list(Access.__members__.values()).index(self)

    def _generate_next_value_(self, start, count, last_values):
        return self.lower()

    @classmethod
    def asdict(cls):
        return {key: value._value_ for key, value in cls.__members__.items()}

    @classmethod
    def attrs(cls):
        return list(cls.__members__)

    @classmethod
    def default(cls):
        return cls._member_map_[cls._member_names_[0]]

    @classmethod
    def default_attr(cls):
        return cls.attrs()[0]

    @classmethod
    def default_dict(cls):
        return {cls.default_attr(): cls.default_value()}

    @classmethod
    def default_value(cls):
        return cls[cls.default_attr()]

    @property
    def describe(self):
        """
        Returns:
            tuple:
        """
        # self is the member here
        return self.name, self.value

    @property
    def lower(self):
        return self.name.lower()

    @classmethod
    def values(cls):
        return list(cls.asdict().values())


EnumBaseAlias = Alias(EnumBase, 1, name=EnumBase.__name__)


# </editor-fold>
# <editor-fold desc="Classes: Enums, Named and No Deps">
class AccessMeta(EnumMeta):
    def __getitem__(cls, item):
        """
        Access Instance Value:
            - If str and is enum key: returns value.
            - If str and not enum key: returns value base on re.compile.
            - If Access Instance: returns item.

        Examples:
            >>> from rc import Access, pretty_install
            >>> pretty_install()
            >>> Access[str()], Access['__name__'], Access['_name__'], Access['name__'], Access[Access.PROTECTED]
            (
                None,
                <Access.PRIVATE: re.compile('^__.*')>,
                <Access.PROTECTED: re.compile('^_(?!_).*$')>,
                <Access.PUBLIC: re.compile('^(?!_)..*$')>,
                <Access.PROTECTED: re.compile('^_(?!_).*$')>
            )
            >>> Access['PROTECTED']
            <Access.PROTECTED: re.compile('^_(?!_).*$')>
            >>> Access[dict()] # doctest: +IGNORE_EXCEPTION_DETAIL, +ELLIPSIS
            Traceback (most recent call last):
            KeyError: "{} not in ...

        Raises:
            KeyError: item not in cls.__members__.

        Args:
            item: Access key, string to run re.compile or Access Instance.

        Returns:
            Access Instance.
        """
        if isinstance(item, str):
            if item in cls._member_map_:
                return cls._member_map_[item]
            for key in list(cls._member_map_.keys())[1:]:
                value = cls._member_map_[key]
                if bool(value.value.search(item)):
                    return value
            return
        elif isinstance(item, Enum):
            return item
        else:
            for value in cls._member_map_.values():
                if value.value == item:
                    return item
        raise KeyError(f'{item} not in {cls._member_map_}')

    __class_getitem__ = __getitem__


class Access(EnumBase, metaclass=AccessMeta):
    """Access Attributes Enum Class."""
    ALL = re.compile('.')
    PRIVATE = re.compile('^__.*')
    PROTECTED = re.compile('^_(?!_).*$')
    PUBLIC = re.compile('^(?!_)..*$')

    @classmethod
    @functools.cache
    def classify(cls, *args, keys=False, **kwargs):
        """
        Classify args or kwargs based on data access.

        Examples:
            >>> a = Access.classify(**dict(map(lambda x: (x.name, x, ), classify_class_attrs(TestAsync))))
            >>> n = '__class__'
            >>> assert n in a.all and n in a.private and n not in a.protected and n not in a.public
            >>> n = TestAsync._staticmethod
            >>> assert n in a.all and n not in a.private and n in a.protected and n not in a.public
            >>> n = TestAsync.staticmethod.__name__
            >>> assert n in a.all and n not in a.private and n not in a.protected and n in a.public
            >>>
            >>> a = Access.classify(**dict(map(lambda x: (x.name, x, ), classify_class_attrs(TestAsync))), keys=True)
            >>> n = '__class__'
            >>> assert n in a.all and n in a.private and n not in a.protected and n not in a.public
            >>> assert a.private[n].name == n
            >>> n = TestAsync._staticmethod
            >>> assert n in a.all and n not in a.private and n in a.protected and n not in a.public
            >>> assert a.protected[n].name == n
            >>> n = TestAsync.staticmethod.__name__
            >>> assert n in a.all and n not in a.private and n not in a.protected and n in a.public
            >>> assert a.public[n].name == n

        Args:
            *args: str iterable.
            keys: include keys if kwargs so return dict or list with kwargs.values().
            **kwargs: dict with keys to check and values

        Raises:
            TypeError('args or kwargs not both')

        Returns:
            AccessEnumMembers.
        """
        if args and kwargs:
            raise TypeError('args or kwargs not both')
        rv: defaultdict[Access, Union[dict, list]] = defaultdict(dict if keys and kwargs else list, dict())
        for name in args or kwargs:
            if value := cls[name]:
                rv[value].update({name: kwargs.get(name)}) if keys and kwargs else rv[value].append(name)
                rv[cls.ALL].update({name: kwargs.get(name)}) if keys and kwargs else rv[cls.ALL].append(name)
        return AccessEnumMembers(all=rv[cls.ALL], private=rv[cls.PRIVATE], protected=rv[cls.PROTECTED],
                                 public=rv[cls.PUBLIC])

    @functools.cache
    def include(self, name):
        """
        Include Key.

        Examples:
            >>> pretty_install()
            >>> Access.ALL.include(str()), Access.PRIVATE.include(str()), Access.PROTECTED.include(str()), \
Access.PUBLIC.include(str())
            (None, None, None, None)
            >>> Access.ALL.include('__name__'), Access.PRIVATE.include('__name__'), \
            Access.PROTECTED.include('__name__'), Access.PUBLIC.include('__name__')
            (True, True, False, False)
            >>> Access.ALL.include('_name__'), Access.PRIVATE.include('_name__'), Access.PROTECTED.include('_name__'), \
Access.PUBLIC.include('_name__')
            (True, True, True, False)
            >>> Access.ALL.include('name__'), Access.PRIVATE.include('name__'), Access.PROTECTED.include('name__'), \
Access.PUBLIC.include('name__')
            (True, True, True, True)

        Args:
            name: name.

        Returns:
            True if key to be included.
        """
        if name:
            return type(self)[name] >= self


AccessEnumMembers = namedtuple('AccessEnumMembers', 'all private protected public')


Annotation = namedtuple('Annotation', 'any args classvar cls default final hint initvar literal name optional '
                                      'origin union')


Attribute = namedtuple('Attribute', 'access annotation classvar copy coro dataclass default defaultfactory defining '
                                    'dict dictslot es field hash ignore '
                                    'init initvar kind name named object '
                                    'publicprop qual repr slot state str type value')


class BaseState:
    """
    Deepcopy and Pickle State Base Class.

    Examples:
        >>> from copy import deepcopy
        >>>
        >>> class Test(BaseState):
        ...     __slots__ = ('attribute', )
        ...     __state__ = ('attribute', )
        ...     def __init__(self): self.attribute = dict(a=1)
        >>>
        >>> test = Test()
        >>> test_copy = test
        >>> test_deepcopy = deepcopy(test)
        >>> assert id(test) == id(test_copy)
        >>> assert id(test) != id(test_deepcopy)
        >>> test.attribute['a'] = 2
        >>> assert id(test.attribute) == id(test_copy.attribute)
        >>> assert id(test.attribute) != id(test_deepcopy.attribute)
        >>> assert test_copy.attribute['a'] == 2
        >>> assert test_deepcopy.attribute['a'] == 1
    """
    __slots__ = ()
    __state__ = ()

    def __getstate__(self): return Es(self).state()

    def __setstate__(self, state): Es(self).state(state)


class BoxKeys(Box):
    """
    Creates a Box with values from keys.
    """

    def __init__(self, keys, value='lower'):
        """
        Creates Box instance.

        Examples:
            >>> pretty_install()
            >>>
            >>> BoxKeys('a b', value=None)
            <Box: {'a': 'a', 'b': 'b'}>
            >>> BoxKeys('A B')
            <Box: {'A': 'a', 'B': 'b'}>
            >>> BoxKeys('A B', value=list)
            <Box: {'A': [], 'B': []}>

        Args:
            keys: keys to use for keys and values.
            value: Type or function to use to init the Box.

        Returns:
            Initialize box from keys.
        """
        es = Es(value)
        super().__init__({item: getattr(item, value)() if es.str else item if es.none else value()
                          for item in to_iter(keys)})


CacheWrapperInfo = namedtuple('CacheWrapperInfo', 'hit passed total')


class ChainRV(EnumBase):
    ALL = auto()
    FIRST = auto()
    UNIQUE = auto()


class Chain(ChainMap):
    """Variant of chain that allows direct updates to inner scopes and returns more than one value,
    not the first one."""
    rv = ChainRV.UNIQUE
    default = None
    maps = list()

    def __init__(self, *maps, rv=ChainRV.UNIQUE, default=None):
        super().__init__(*maps)
        self.rv = rv
        self.default = default

    def __getitem__(self, key):
        rv = []
        for mapping in self.maps:
            if Es(mapping).namedtype:
                mapping = mapping._asdict()
            elif hasattr(mapping, 'asdict'):
                to_dict = getattr(mapping.__class__, 'asdict')
                if isinstance(to_dict, property):
                    mapping = mapping.asdict
                elif callable(to_dict):
                    mapping = mapping.asdict()
            if hasattr(mapping, '__getitem__'):
                try:
                    value = mapping[key]
                    if self.rv is ChainRV.FIRST:
                        return value
                    if (self.rv is ChainRV.UNIQUE and value not in rv) or self.rv is ChainRV.ALL:
                        rv.append(value)
                except KeyError:
                    pass
            elif hasattr(mapping, '__getattribute__') and isinstance(key, str) and \
                    not isinstance(mapping, (tuple, bool, int, str, bytes)):
                try:
                    value = getattr(mapping, key)
                    if self.rv is ChainRV.FIRST:
                        return value
                    if (self.rv is ChainRV.UNIQUE and value not in rv) or self.rv is ChainRV.ALL:
                        rv.append(value)
                except AttributeError:
                    pass
        return self.default if self.rv is ChainRV.FIRST else rv

    def __delitem__(self, key):
        index = 0
        deleted = []
        found = False
        for mapping in self.maps:
            if mapping:
                if not isinstance(mapping, (tuple, bool, int, str, bytes)):
                    if hasattr(mapping, '__delitem__'):
                        if key in mapping:
                            del mapping[key]
                            if self.rv is ChainRV.FIRST:
                                found = True
                    elif hasattr(mapping, '__delattr__') and hasattr(mapping, key) and isinstance(key, str):
                        delattr(mapping.__class__, key) if key in dir(mapping.__class__) else delattr(mapping, key)
                        if self.rv is ChainRV.FIRST:
                            found = True
                if not mapping:
                    deleted.append(index)
                if found:
                    break
            index += 1
        for index in reversed(deleted):
            del self.maps[index]
        return self

    def delete(self, key):
        del self[key]
        return self

    def __setitem__(self, key, value):
        found = False
        for mapping in self.maps:
            if mapping:
                if not isinstance(mapping, (tuple, bool, int, str, bytes)):
                    if hasattr(mapping, '__setitem__'):
                        if key in mapping:
                            mapping[key] = value
                            if self.rv is ChainRV.FIRST:
                                found = True
                    elif hasattr(mapping, '__setattr__') and hasattr(mapping, key) and isinstance(key, str):
                        setattr(mapping, key, value)
                        if self.rv is ChainRV.FIRST:
                            found = True
                if found:
                    break
        if not found and not isinstance(self.maps[0], (tuple, bool, int, str, bytes)):
            if hasattr(self.maps[0], '__setitem__'):
                self.maps[0][key] = value
            elif hasattr(self.maps[0], '__setattr__') and isinstance(key, str):
                setattr(self.maps[0], key, value)
        return self

    def set(self, key, value):
        return self.__setitem__(key, value)


class Executor(EnumBase):
    PROCESS = ProcessPoolExecutor
    THREAD = ThreadPoolExecutor
    NONE = None

    async def run(self, func, *args, **kwargs):
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


FindUp = namedtuple('FindUp', 'path previous')
FrameBase = namedtuple('FrameBase', 'back code frame function globals lineno locals name package path vars')


class Frame(FrameBase):
    """Frame Class."""
    __slots__ = ()
    def __new__(cls, *args, **kwargs): return super().__new__(cls, *args, **kwargs)

    @property
    def asttokens(self):
        """
        Returns :meth:`executing.executing.Executing.source.asttokens()`.

        Examples:
            >>> from ast import unparse
            >>> from inspect import getmodulename
            >>> import rc.utils
            >>> from rc import N
            >>>
            >>> pretty_install()
            >>>

        Returns:
            Frame source.
        """
        return self.executing.source.asttokens()

    @property
    def executing(self):
        """
        Returns :class:`executing.executing.Executing`.

        Examples:
            >>> from ast import unparse
            >>> from inspect import getmodulename
            >>> import rc.utils
            >>> from rc import N
            >>>
            >>> pretty_install()
            >>>

        Returns:
            Frame source.
        """
        return Source.executing(self.frame)


FrameSourceNode = namedtuple('FrameSourceNode', 'framebase sourcenode')


GitTop = namedtuple('GitTop', 'name origin path')


class Is:
    """
    Is Instance, Subclass Helper Class

    Examples:
        >>> from ast import unparse
        >>> from inspect import getmodulename
        >>> from rc import Is, pretty_install, TestAsync
        >>> import rc.utils
        >>>
        >>> pretty_install()
        >>>
        >>> Is(2).int
        True
        >>> Is(2).bool
        False
        >>> Is(2).instance(dict, tuple)
        False
        >>> Is(2).instance(dict, tuple)
        False
        >>> def func(): pass
        >>> Is(func).coro
        False
        >>> async def async_func(): pass
        >>> i = Is(async_func)
        >>> i.coro, i.coroutinefunction, i.asyncgen, i.asyncgenfunction, i.awaitable, i.coroutine
        (True, True, False, False, False, False)
        >>> rv = dict(map(lambda x: (x.name, Is(x.object)), classify_class_attrs(TestAsync)))
        >>> rv['async_classmethod'].coro, rv['async_classmethod'].coroutinefunction, \
        rv['async_classmethod'].asyncgen, rv['async_classmethod'].asyncgenfunction, \
        rv['async_classmethod'].awaitable, rv['async_classmethod'].coroutine
        (True, True, False, False, False, False)
        >>> rv['async_method'].coro, rv['async_method'].coroutinefunction, \
        rv['async_method'].asyncgen, rv['async_method'].asyncgenfunction, \
        rv['async_method'].awaitable, rv['async_method'].coroutine
        (True, True, False, False, False, False)
        >>> rv['async_prop'].coro, rv['async_prop'].coroutinefunction, \
        rv['async_prop'].asyncgen, rv['async_prop'].asyncgenfunction, \
        rv['async_prop'].awaitable, rv['async_prop'].coroutine
        (True, True, False, False, False, False)
        >>> rv['async_staticmethod'].coro, rv['async_staticmethod'].coroutinefunction, \
        rv['async_staticmethod'].asyncgen, rv['async_staticmethod'].asyncgenfunction, \
        rv['async_staticmethod'].awaitable, rv['async_staticmethod'].coroutine
        (True, True, False, False, False, False)
        >>> assert rv['classmethod'].coro == False
        >>> assert rv['cprop'].coro == False
        >>> assert rv['method'].coro == False
        >>> assert rv['prop'].coro == False
        >>> assert rv['staticmethod'].coro == False

    Attributes:
    -----------
        data: Any
            object to provide information (default: None)
    """
    __slots__ = ('data', )

    def __init__(self, data=None): self.data = data

    def __getstate__(self): return dict(data=self.data)

    def __hash__(self): return hash((type(self.data), self.__str__()))

    def __reduce__(self): return self.__class__, tuple(self.data)

    def __reduce_ex__(self, *args, **kwargs): return self.__class__, tuple(self.data)

    def __repr__(self): return f'{self.__class__.__name__}({self.data})'

    def __setstate__(self, state): self.data = state['data']

    def __str__(self): return str(self.data)

    @property
    def _cls(self): return self.data if isinstance(self.data, type) else type(self.data)

    @property
    def _func(self): return self.data.fget if isinstance(self.data, property) else self.data.__func__ \
        if isinstance(self.data, (classmethod, staticmethod)) else self.data

    @property
    def annotation(self): return isinstance(self.data, Annotation)

    @property
    def annotationstype(self): return isinstance(self.data, AnnotationsType)

    @property
    def annotationstype_any(self): return self.annotationstype or self.annotationstype_sub

    @property
    def annotationstype_sub(self): return isinstance(self.data, type) and issubclass(self.data, AnnotationsType)

    @property
    def asdictmethod(self): return isinstance(self.data, AsDictMethodType)

    @property
    def asdictmethod_sub(self): return isinstance(self.data, type) and issubclass(self.data, AsDictMethodType)

    @property
    def asdictproperty(self): return isinstance(self.data, AsDictPropertyType)

    @property
    def asdictproperty_sub(self): return isinstance(self.data, type) and issubclass(self.data, AsDictPropertyType)

    @property
    def ast(self): return isinstance(self.data, AST)

    @property
    def asyncfor(self): return isinstance(self._func, AsyncFor)

    @property
    def asyncfunctiondef(self): return isinstance(self._func, AsyncFunctionDef)

    @property
    def asyncgen(self): return isinstance(self._func, AsyncGeneratorType)

    @property
    def asyncgenfunction(self): return isasyncgenfunction(self._func)

    @property
    def asyncwith(self): return isinstance(self._func, AsyncWith)

    @property
    def attribute(self): return isinstance(self.data, Attribute)

    @property
    def await_ast(self): return isinstance(self._func, Await)

    @property
    def awaitable(self): return isawaitable(self._func)

    @property
    def basestate(self): return isinstance(self.data, BaseState)

    @property
    def bool(self): return isinstance(self.data, int) and isinstance(self.data, bool)

    @property
    def builtin(self): return any([in_dict(BUILTINS, self.data), self.builtinclass, self.builtinfunction])

    @property
    def builtinclass(self): return self.data in BUILTINS_CLASSES

    @property
    def builtinfunction(self): return self.data in BUILTINS_FUNCTIONS

    @property
    def builtinfunctiontype(self): return isinstance(self.data, BuiltinFunctionType)

    @property
    def bytesio(self): return isinstance(self.data, BytesIO)  # :class:`typing.BinaryIO`

    @property
    def cached_property(self): return isinstance(self.data, cached_property)

    @property
    def callable(self): return isinstance(self.data, Callable)

    @property
    def chain(self): return isinstance(self.data, Chain)

    @property
    def chainmap(self): return isinstance(self.data, ChainMap)

    @property
    def classdef(self): return isinstance(self.data, ClassDef)

    @property
    def classmethoddescriptortype(self): return isinstance(self.data, ClassMethodDescriptorType)

    @property
    def classvar(self): return (self.datafield and get_origin(self.data.type) == ClassVar) or get_origin(
        self.data) == ClassVar

    @property
    def clsmethod(self): return isinstance(self.data, classmethod)

    @property
    def codetype(self): return isinstance(self.data, CodeType)

    @property
    def collections(self): return is_collections(self.data)

    @property
    def container(self): return isinstance(self.data, Container)

    @property
    def coro(self): return any(
        [self.asyncfor, self.asyncfunctiondef, self.asyncwith, self.await_ast] if self.ast else
        [self.asyncgen, self.asyncgenfunction, self.awaitable, self.coroutine, self.coroutinefunction])

    @property
    def coroutine(self): return iscoroutine(self._func) or isinstance(self._func, CoroutineType)

    @property
    def coroutinefunction(self): return iscoroutinefunction(self._func)

    @property
    def default_factory(self): return self.datafield and type(self)(
        self.data.default).missing and N.default_factory.has(self.data)

    @property
    def datafield(self): return isinstance(self.data, DataField)

    @property
    def datatype(self): return isinstance(self.data, DataType)

    @property
    def datatype_any(self): return self.datatype or self.datatype_sub

    @property
    def datatype_sub(self): return isinstance(self.data, type) and issubclass(self.data, DataType)

    @property
    def defaultdict(self): return isinstance(self.data, defaultdict)

    @property
    def deleter(self): return self.property_any and self.data.fdel is not None

    @property
    def dict(self): return isinstance(self.data, dict)

    @property
    def dicttype(self): return isinstance(self.data, DictType)

    @property
    def dicttype_sub(self): return isinstance(self.data, type) and issubclass(self.data, DictType)

    @property
    def dynamicclassattribute(self): return isinstance(self.data, DynamicClassAttribute)

    @property
    def dlst(self): return isinstance(self.data, (dict, list, set, tuple))

    @property
    def enum(self): return isinstance(self.data, Enum)

    @property
    def enum_sub(self): return isinstance(self.data, type) and issubclass(self.data, Enum)

    @property
    def enumbase(self): return isinstance(self.data, EnumBase)

    @property
    def enumbase_sub(self): return isinstance(self.data, type) and issubclass(self.data, EnumBase)

    @property
    def even(self): return not self.data % 2

    @property
    def fileio(self): return isinstance(self.data, FileIO)

    @property
    def float(self): return isinstance(self.data, float)

    @property
    def frameinfo(self): return isinstance(self.data, FrameInfo)

    @property
    def frametype(self): return isinstance(self.data, FrameType)

    @property
    def functiondef(self): return isinstance(self.data, FunctionDef)

    @property
    def functiontype(self): return isinstance(self.data, FunctionType)

    @property
    def generator(self): return isinstance(self.data, Generator)

    @property
    def generatortype(self): return isinstance(self.data, GeneratorType)

    @property
    def genericalias(self): return isinstance(self.data, types.GenericAlias)

    @property
    def getattrnobuiltintype(self): return isinstance(self.data, GetAttrNoBuiltinType)

    @property
    def getattrnobuiltintype_sub(self): return isinstance(self.data, type) and issubclass(
        self.data, GetAttrNoBuiltinType)

    @property
    def getattrtype(self): return isinstance(self.data, GetAttrType)

    @property
    def getattrtype_sub(self): return isinstance(self.data, type) and issubclass(self.data, GetAttrType)

    @property
    def getsetdescriptortype(self): return isinstance(self.data, GetSetDescriptorType)

    @property
    def gettype(self): return isinstance(self.data, GetType)

    @property
    def gettype_sub(self): return isinstance(self.data, type) and issubclass(self.data, GetType)

    @property
    def hashable(self): return bool(noexception(TypeError, hash, self.data))

    @property
    def import_ast(self): return isinstance(self.data, Import)

    @property
    def importfrom(self): return isinstance(self.data, ImportFrom)

    @property
    def initvar(self): return (self.datafield and isinstance(self.data.type, InitVar)) or isinstance(self.data, InitVar)

    @property
    def installed(self): return is_installed(self.data)

    def instance(self, *args): return isinstance(self.data, args)

    @property
    def int(self): return isinstance(self.data, int)

    @property
    def io(self): return self.bytesio and self.stringio  # :class:`typing.IO`

    @property
    def iterable(self): return isinstance(self.data, Iterable)

    @property
    def iterator(self): return isinstance(self.data, Iterator)

    @property
    def lambdatype(self): return isinstance(self.data, LambdaType)

    @property
    def list(self): return isinstance(self.data, list)

    @property
    def lst(self): return isinstance(self.data, (list, set, tuple))

    @property
    def mappingproxytype(self): return isinstance(self.data, MappingProxyType)

    @property
    def mappingproxytype_sub(self): return isinstance(self.data, type) and issubclass(self.data, MappingProxyType)

    @property
    def memberdescriptor(self): return ismemberdescriptor(self.data)

    @property
    def memberdescriptortype(self): return isinstance(self.data, MemberDescriptorType)

    @property
    def meta(self): return  # TODO: meta

    @property
    def meta_sub(self): return  # TODO: meta_sub

    @property
    def metatype(self): return  # TODO: metatype

    @property
    def metatype_sub(self): return  # TODO: metatype_sub

    @property
    def method(self): return self.methodtype and not isinstance(self.data, (classmethod, property, staticmethod))

    @property
    def methoddescriptor(self): return ismethoddescriptor(self.data)

    @property
    def methoddescriptortype(self): return isinstance(self.data, types.MethodDescriptorType)

    @property
    def methodtype(self): return isinstance(self.data, MethodType)  # True if it is an instance method!.

    @property
    def methodwrappertype(self): return isinstance(self.data, MethodWrapperType)

    @property
    def methodwrappertype_sub(self): return isinstance(self.data, type) and issubclass(self.data, MethodWrapperType)

    @property
    def missing(self): return isinstance(self.data, MISSING_TYPE)

    @property
    def mlst(self): return isinstance(self.data, (MutableMapping, list, set, tuple))

    @property
    def mm(self): return isinstance(self.data, MutableMapping)

    @property
    def moduletype(self): return isinstance(self.data, ModuleType)

    @property
    def module_function(self): return is_module_function(self.data)

    @property
    def noncomplex(self): return is_noncomplex(self.data)

    @property
    def namedtype(self): return isinstance(self.data, NamedType)

    @property
    def namedtype_any(self): return self.namedtype or self.namedtype_sub

    @property
    def namedtype_sub(self): return isinstance(self.data, type) and issubclass(self.data, NamedType)

    @property
    def named_annotationstype(self): return isinstance(self.data, NamedAnnotationsType)

    @property
    def named_annotationstype_sub(self): return isinstance(self.data, type) and issubclass(
        self.data, NamedAnnotationsType)

    @property
    def none(self): return isinstance(self.data, type(None))

    @property
    def object(self): return is_object(self.data)

    @property
    def pathlib(self): return isinstance(self.data, PathLib)

    @property
    def picklable(self): return True if noexception(pickle_dumps, self.data) else False

    @property
    def primitive(self): return is_primitive(self.data)

    @property
    def prop(self): return isinstance(self.data, property)

    @property
    def property_any(self): return self.prop or self.cached_property

    @property
    def reducible(self): return is_reducible(self.data)

    @property
    def reducible_sequence_subclass(self): return is_reducible_sequence_subclass(self.data)

    @property
    def routine(self): return isroutine(self.data)

    @property
    def sequence(self): return isinstance(self.data, Sequence)

    @property
    def sequence_sub(self): return isinstance(self.data, type) and issubclass(self.data, Sequence)

    @property
    def _set(self): return isinstance(self.data, set)

    @property
    def setter(self): return isinstance(self.data, property) and self.data.fset is not None

    @property
    def simple(self): return isinstance(self.data, Simple)

    @property
    def sized(self): return isinstance(self.data, Sized)

    @property
    def slotstype(self): return isinstance(self.data, SlotsType)

    @property
    def slotstype_sub(self): return isinstance(self.data, type) and issubclass(self.data, SlotsType)

    @property
    def static(self): return isinstance(self.data, staticmethod)

    @property
    def str(self): return isinstance(self.data, str)

    def subclass(self, *args): return isinstance(self.data, type) and issubclass(self.data, args)

    @property
    def stringio(self): return isinstance(self.data, StringIO)  # :class:`typing.TextIO`

    @property
    def tracebacktype(self): return isinstance(self.data, TracebackType)

    @property
    def tuple(self): return isinstance(self.data, tuple)

    @property
    def type(self): return isinstance(self.data, type)

    @property
    def unicode(self): return is_unicode(self.data)

    @property
    def wrapperdescriptortype(self): return isinstance(self.data, WrapperDescriptorType)


class Kind(EnumBase):
    CLASS = 'class method'
    DATA = 'data'
    METHOD = 'method'
    PROPERTY = 'property'
    STATIC = 'static method'


class ModuleBase(EnumBase):
    @property
    @functools.cache
    def get(self): return globals().get(self.value)

    @property
    @functools.cache
    def load(self): return import_module(self.value)


class Module(ModuleBase):
    """Module Enum Class."""
    FUNCTOOLS = auto()
    TYPING = auto()


class NBase(EnumBase):
    """Access Types Base Enum Class."""

    def _generate_next_value_(self, start, count, last_values):
        return f'__{self.lower()}__' if self.isupper() else self.removesuffix('_')

    def get(self, obj, default=None, setvalue=False):
        """
        Get key/attr value.

        Args:
            obj: object.
            default: None.
            setvalue: set value if not found.

        Returns:
            Value.
        """
        return Es(obj).get(name=self.value, default=default, setvalue=setvalue)

    def getf(self, obj, default=None):
        # noinspection PyUnresolvedReferences
        """
        Get value from: FrameInfo, FrameType, TracebackType, MutableMapping abd GetAttr.

        Use :class:`rc.N.get` for real names.


        Examples:
            >>> from ast import unparse
            >>> from inspect import getmodulename
            >>> from types import FrameType
            >>> import rc.utils
            >>> from rc import N
            >>> from rc import insstack
            >>> pretty_install()
            >>>
            #
            # FrameInfo
            #
            >>> f = insstack()[0]
            >>> assert f == FrameInfo(N.frame.getf(f), str(N.filename.getf(f)), N.lineno.getf(f),\
            N.function.getf(f), N.code_context.getf(f), N.index.getf(f))
            >>> assert N.filename.getf(f) == N.FILE.getf(f)
            >>> assert str(N.FILE.getf(f)) == str(N.filename.getf(f))
            >>> assert N.NAME.getf(f) == N.co_name.getf(f) == N.function.getf(f)
            >>> assert N.lineno.getf(f) == N.f_lineno.getf(f) == N.tb_lineno.getf(f)
            >>> assert N.vars.getf(f) == (f.frame.f_globals | f.frame.f_locals).copy()
            >>> assert N.vars.getf(f) == (N.f_globals.getf(f) | N.f_locals.getf(f)).copy()
            >>> assert N.vars.getf(f) == (N.globals.getf(f) | N.locals.getf(f)).copy()
            >>> assert N.SPEC.getf(f).origin == __file__
            >>>
            #
            # FrameType
            #
            >>> frameinfo = insstack()[0]
            >>> from rc import N
            >>> f: FrameType = frameinfo.frame
            >>> assert N.filename.getf(f) == N.filename.getf(frameinfo)
            >>> assert N.frame.getf(f) == N.frame.getf(frameinfo)
            >>> assert N.lineno.getf(f) == N.lineno.getf(frameinfo)
            >>> assert N.function.getf(f) == N.function.getf(frameinfo)
            >>> assert frameinfo == FrameInfo(N.frame.getf(f), str(N.filename.getf(f)), N.lineno.getf(f),\
            N.function.getf(f), frameinfo.code_context, frameinfo.index)
            >>> assert N.filename.getf(f) == N.FILE.getf(f)
            >>> assert str(N.FILE.getf(f)) == str(N.filename.getf(f))
            >>> assert N.NAME.getf(f) == N.co_name.getf(f) == N.function.getf(f)
            >>> assert N.lineno.getf(f) == N.f_lineno.getf(f) == N.tb_lineno.getf(f)
            >>> assert N.vars.getf(f) == (f.f_globals | f.f_locals).copy()
            >>> assert N.vars.getf(f) == (N.f_globals.getf(f) | N.f_locals.getf(f)).copy()
            >>> assert N.vars.getf(f) == (N.globals.getf(f) | N.locals.getf(f)).copy()
            >>> assert N.SPEC.getf(f).origin == __file__
            >>>
            #
            # MutaleMapping
            #
            >>> f = insstack()[0]
            >>> globs_locs = (f.frame.f_globals | f.frame.f_locals).copy()
            >>> N.filename.getf(f), N.function.getf(f)  # doctest: +ELLIPSIS
            (PosixPath('<doctest ...>'), '<module>')
            >>> N.FILE.getf(globs_locs)  # doctest: +ELLIPSIS
            PosixPath('/Users/jose/....py')
            >>> assert N.FILE.getf(globs_locs) == PathLib(__file__) == PathLib(N.SPEC.getf(globs_locs).origin)
            >>> assert N.NAME.getf(globs_locs) == getmodulename(__file__) == N.SPEC.getf(globs_locs).name

            #
            # GetAttr
            #
            >>> f = insstack()[0]
            >>> globs_locs = (f.frame.f_globals | f.frame.f_locals).copy()

            >>> N.FILE.getf(globs_locs)  # doctest: +ELLIPSIS
            PosixPath('/Users/jose/....py')
            >>> N.FILE.getf(rc.utils)  # doctest: +ELLIPSIS
            PosixPath('/Users/jose/....py')
            >>> globs_locs.get('__spec__'), N.SPEC.getf(globs_locs)  # doctest: +ELLIPSIS
            (
                ModuleSpec(name='...', loader=<_frozen_importlib_external.SourceFileLoader ...>, origin='....py'),
                ModuleSpec(name='...', loader=<_frozen_importlib_external.SourceFileLoader ...>, origin='....py')
            )
            >>> PathLib(N.SPEC.getf(globs_locs).origin)  # doctest: +ELLIPSIS
            PosixPath('/Users/jose/....py')
            >>> PathLib(N.SPEC.getf(rc.utils).origin)  # doctest: +ELLIPSIS
            PosixPath('/Users/jose/....py')
            >>> N.SPEC.getf(rc.utils).name == rc.utils.__name__
            True
            >>> N.SPEC.getf(rc.utils).name.split('.')[0] == rc.utils.__package__
            True
            >>> N.NAME.getf(rc.utils) == rc.utils.__name__
            True
            >>> N.PACKAGE.getf(rc.utils) == rc.utils.__package__
            True

        Args:
            obj: object
            default: None

        Returns:
            Value from get() method
        """
        return Es(obj).getf(name=self, default=default)

    @property
    @functools.cache
    def getter(self):
        """
        Attr Getter.

        Examples:
            >>> import rc.utils
            >>> from rc import N
            >>>
            >>> N.MODULE.getter(tuple)
            'builtins'
            >>> N.NAME.getter(tuple)
            'tuple'
            >>> N.FILE.getter(rc.utils)  # doctest: +ELLIPSIS
            '/Users/jose/....py'
        """
        return attrgetter(self.value)

    def has(self, obj):
        """
        Checks if Object has attr.

        Examples:
            >>> from rc import Es, N, pretty_install
            >>> pretty_install()
            >>> import rc.utils
            >>>
            >>> class Test:
            ...     __repr_newline__ = True
            >>>
            >>> N.REPR_NEWLINE.has(Test())
            True
            >>> N.REPR_EXCLUDE.has(Test())
            False
            >>> N.MODULE.has(tuple)
            True
            >>> N.NAME.has(tuple)
            True
            >>> N.FILE.has(tuple)
            False
            >>> N.FILE.has(rc.utils)
            True

        Args:
            obj: object.

        Returns:
            True if object has attribute.
        """
        return hasattr(obj, self.value)

    def mro_first_data(self, obj):
        """
        First value of attr found in mro and instance if obj is instance.

        Examples:
            >>> pretty_install()
            >>>
            >>> class Test:
            ...     __repr_newline__ = True
            >>>
            >>> test = Test()
            >>> class Test2(Test):
            ...     def __init__(self):
            ...         self.__repr_newline__ = False
            >>>
            >>> N.REPR_NEWLINE.mro_first_data(Test())
            True
            >>> N.REPR_NEWLINE.mro_first_data(Test2())
            False
            >>> N.REPR_NEWLINE.mro_first_data(int())
            >>> N.REPR_PPROPERTY.mro_first_data(Test())

        Returns:
            First value of attr found in mro and instance if obj is instance.
        """
        return Es(obj).mro_first_data(self)

    def mro_first_dict(self, obj, mro=None):
        """
        First value of attr in obj.__class__.__dict__ found in mro.

        Examples:
            >>> pretty_install()
            >>>
            >>> class Test:
            ...     __repr_newline__ = False
            >>>
            >>> test = Test()
            >>> class Test2(Test):
            ...     def __init__(self):
            ...         self.__repr_newline__ = False
            >>>
            >>> N.REPR_NEWLINE.mro_first_dict(Test())
            False
            >>> N.REPR_NEWLINE.mro_first_dict(Test2())
            False
            >>> N.REPR_NEWLINE.mro_first_dict(int())
            NotImplemented
            >>> N.REPR_PPROPERTY.mro_first_dict(Test())
            NotImplemented
            >>> A = namedtuple('A', 'a')
            >>> N.SLOTS.mro_first_dict(A)
            ()

        Args:
            obj: object.
            mro: mro or search in dict (default: self.mro)

        Returns:
            First value of attr in obj.__class__.__dict__ found in mro.
        """
        return Es(obj).mro_first_dict(self, mro=mro)

    def mro_first_dict_no_object(self, obj):
        """
        First value of attr in obj.__class__.__dict__ found in mro excluding object.

        Examples:
            >>> pretty_install()
            >>>
            >>> class Test:
            ...     __repr_newline__ = False
            >>>
            >>> test = Test()
            >>> class Test2(Test):
            ...     def __init__(self):
            ...         self.__repr_newline__ = False
            >>>
            >>> N.REPR_NEWLINE.mro_first_dict_no_object(Test())
            False
            >>> N.REPR_NEWLINE.mro_first_dict_no_object(Test2())
            False
            >>> N.REPR_NEWLINE.mro_first_dict_no_object(int())
            NotImplemented
            >>> N.REPR_PPROPERTY.mro_first_dict_no_object(Test())
            NotImplemented
            >>> A = namedtuple('A', 'a')
            >>> N.SLOTS.mro_first_dict_no_object(A)
            ()
            >>> N.SLOTS.mro_first_dict_no_object(dict)
            NotImplemented
            >>> N.SLOTS.mro_first_dict_no_object(dict())
            NotImplemented

        Returns:
            First value of attr in obj.__class__.__dict__ found in mro excluding object.
        """
        return Es(obj).mro_first_dict_no_object(self)

    def mro_values(self, obj):
        """
        All/accumulated values of attr in mro and obj if instance.

        Examples:
            >>> pretty_install()
            >>>
            >>> class First:
            ...     __slots__ = ('_hash', '_repr')
            ...     __ignore_copy__ = (tuple, )
            ...     __repr_exclude__ = ('_repr', )
            >>>
            >>> class Test(First):
            ...     __slots__ = ('_prop', '_slot', )
            ...     __hash_exclude__ = ('_slot', )
            ...     __ignore_attr__ = ('attr', )
            ...     __ignore_kwarg__ = ('kwarg', )
            ...     __ignore_str__ = (tuple, )
            >>>
            >>> test = Test()
            >>> N.HASH_EXCLUDE.mro_values(test)
            ('_slot',)
            >>> N.IGNORE_ATTR.mro_values(test)  # doctest: +ELLIPSIS
            (
                'asdict',
                'attr',
                ...
            )
            >>> set(N.IGNORE_COPY.mro_values(test)).difference(N.IGNORE_COPY.mro_values_default(test))
            {<class 'tuple'>}
            >>> N.IGNORE_KWARG.mro_values(test)
            ('kwarg',)
            >>> set(N.IGNORE_STR.mro_values(test)).difference(N.IGNORE_STR.mro_values_default(test))
            {<class 'tuple'>}
            >>> N.REPR_EXCLUDE.mro_values(test)
            ('_repr',)
            >>> N.SLOTS.mro_values(test)
            ('_hash', '_prop', '_repr', '_slot')
            >>> assert sorted(N.STATE.mro_values(Environs())) == sorted(STATE_ATTRS[Environs])

        Returns:
            All/accumulated values of attr in mro and obj if instance.
        """
        return Es(obj).mro_values(self)

    def mro_values_default(self, obj):
        """
        Default values for attr in mro and instance.

        Examples:
            >>> from rc import Es, pretty_install
            >>> pretty_install()
            >>>
            >>> class First:
            ...     __slots__ = ('_hash', '_repr')
            ...     __ignore_copy__ = (tuple, )
            ...     __repr_exclude__ = ('_repr', )
            >>>
            >>> class Test(First):
            ...     __slots__ = ('_prop', '_slot', )
            ...     __hash_exclude__ = ('_slot', )
            ...     __ignore_attr__ = ('attr', )
            ...     __ignore_kwarg__ = ('kwarg', )
            ...     __ignore_str__ = (tuple, )
            >>>
            >>> test = Test()
            >>> assert N.HASH_EXCLUDE.mro_values_default(test) == tuple()
            >>> assert N.IGNORE_ATTR.mro_values_default(test) == IgnoreAttr.__args__
            >>> assert N.IGNORE_COPY.mro_values_default(test) == IgnoreCopy.__args__
            >>> assert N.IGNORE_KWARG.mro_values_default(test) == tuple()
            >>> assert N.IGNORE_STR.mro_values_default(test) == IgnoreStr.__args__
            >>> assert N.REPR_EXCLUDE.mro_values_default(test) == tuple()
            >>> assert N.SLOTS.mro_values_default(test) == tuple()
            >>> assert sorted(N.STATE.mro_values_default(Environs())) == sorted(STATE_ATTRS[Environs])

        Returns:
           Default values for attr in mro and instance.
        """
        return Es(obj).mro_values_default(self)

    def slot(self, obj):
        """
        Is attribute in slots?

        Examples:
            >>> pretty_install()
            >>>
            >>> class First:
            ...     __slots__ = ('_data', )
            >>>
            >>> class Test(First):
            ...     __slots__ = ('_id', )
            >>>
            >>> N._data.slot(Test())
            True
            >>> N._id.slot(Test())
            True
            >>> N.ip.slot(Test())
            False

        Args:
            obj: object.

        Returns:
            True if attribute in slots
        """
        return Es(obj).slot(self)

    def slots_include(self, obj):
        """
        Accumulated values from slots - Accumulated values from mro attr name.

        Examples:
            >>> pretty_install()
            >>>
            >>> class First:
            ...     __slots__ = ('_hash', )
            >>>
            >>> class Test(First):
            ...     __slots__ = ('_prop', '_repr', '_slot', )
            ...     __hash_exclude__ = ('_slot', )
            ...     __repr_exclude__ = ('_repr', )
            >>>
            >>> test = Test()
            >>> slots = N.SLOTS.mro_values(test)
            >>> slots
            ('_hash', '_prop', '_repr', '_slot')
            >>> hash_attrs = N.HASH_EXCLUDE.slots_include(test)
            >>> hash_attrs
            ('_hash', '_prop', '_repr')
            >>> sorted(hash_attrs + N.HASH_EXCLUDE.mro_values(test)) == sorted(slots)
            True
            >>> repr_attrs = N.REPR_EXCLUDE.slots_include(test)
            >>> repr_attrs
            ('_hash', '_prop', '_slot')
            >>> sorted(repr_attrs + N.REPR_EXCLUDE.mro_values(test)) == sorted(slots)
            True

        Returns:
            Accumulated values from slots - Accumulated values from mro attr name.
        """
        return Es(obj).slots_include(self)


class N(NBase):
    """N Access Attributes Enum Class.."""
    ABOUT = auto()
    ABSTRACTMETHODS = auto()
    AUTHOR = auto()
    ADAPT = auto()
    ALL = auto()
    ALLOC = auto()
    ANNOTATIONS = auto()
    ARGS = auto()
    ASDICT = auto()
    ATTRIBUTES = auto()
    BASE = auto()
    BASICSIZE = auto()
    BUILD_CLASS = auto()
    BUILTINS = auto()
    CACHE_CLEAR = auto()
    CACHE_INFO = auto()
    CACHED = auto()
    CLASS = auto()
    CODE = auto()
    CONFORM = auto()
    CONTAINS = auto()
    CREDITS = auto()
    COPY = auto()
    COPYRIGHT = auto()
    CVSID = auto()
    DATACLASS_FIELDS = auto()
    DATACLASS_PARAMS = auto()
    DATE = auto()
    DECIMAL_CONTEXT = auto()
    DEEPCOPY = auto()
    DELATTR = auto()
    DICT = auto()
    DICTOFFSET = auto()
    DIR = auto()
    DOC = auto()
    DOCFORMAT = auto()
    EMAIL = auto()
    EQ = auto()
    EXCEPTION = auto()
    FILE = auto()
    FLAGS = auto()
    GET = auto()
    GETATTRIBUTE = auto()
    GETFORMAT = auto()
    GETINITARGS = auto()
    GETITEM = auto()
    GETNEWARGS = auto()
    GETSTATE = auto()
    HASH = auto()
    HASH_EXCLUDE = auto()
    IGNORE_ATTR = auto()
    IGNORE_COPY = auto()
    IGNORE_HASH = auto()
    IGNORE_INIT = auto()
    IGNORE_KWARG = auto()
    IGNORE_REPR = auto()
    IGNORE_STR = auto()
    INIT = auto()
    INIT_SUBCLASS = auto()
    INITIALIZING = auto()
    ISABSTRACTMETHOD = auto()
    ITEMSIZE = auto()
    LEN = auto()
    LIBMPDEC_VERSION = auto()
    LOADER = auto()
    LTRACE = auto()
    MEMBERS = auto()
    METHODS = auto()
    MODULE = auto()
    MP_MAIN = auto()
    MRO = auto()
    NAME = auto()
    NEW_MEMBER = auto()
    NEW_OBJ = auto()
    NEW_OBJ_EX = auto()
    OBJ_CLASS = auto()
    PACKAGE = auto()
    POST_INIT = auto()
    PREPARE = auto()
    PYCACHE = auto()
    QUALNAME = auto()
    REDUCE = auto()
    REDUCE_EX = auto()
    REPR = auto()
    REPR_EXCLUDE = auto()
    REPR_NEWLINE = auto()
    REPR_PPROPERTY = auto()
    RETURN = auto()
    SELF_CLASS = auto()
    SETATTR = auto()
    SETFORMAT = auto()
    SETSTATE = auto()
    SIGNATURE = auto()
    SIZEOF = auto()
    SLOTNAMES = auto()
    SLOTS = auto()
    SPEC = auto()
    STATE = auto()
    STATUS = auto()
    STR = auto()
    SUBCLASSHOOK = auto()
    TEST = auto()
    TEXT_SIGNATURE = auto()
    THIS_CLASS = auto()
    TRUNC = auto()
    VERSION = auto()
    WARNING_REGISTRY = auto()
    WEAKREF = auto()
    WEAKREFOFFSET = auto()
    WRAPPED = auto()
    _asdict = auto()
    _cls = auto()
    _copy = auto()
    _count = auto()
    _data = auto()
    _extend = auto()
    _external = auto()
    _field_defaults = auto()
    _fields = auto()
    _file = auto()
    _filename = auto()
    _frame = auto()
    _func = auto()
    _function = auto()
    _get = auto()
    _globals = auto()
    _id = auto()
    _index = auto()
    _ip = auto()
    _item = auto()
    _items = auto()
    _key = auto()
    _keys = auto()
    _kind = auto()
    _locals = auto()
    _name = auto()
    _node = auto()
    _origin = auto()
    _obj = auto()
    _object = auto()
    _path = auto()
    _repo = auto()
    _RV = auto()
    _pypi = auto()
    _remove = auto()
    _reverse = auto()
    _sort = auto()
    _source = auto()
    _update = auto()
    _value = auto()
    _values = auto()
    _vars = auto()
    add = auto()
    append = auto()
    asdict = auto()
    cls = auto()
    clear = auto()
    co_name = auto()
    code_context = auto()
    copy = auto()
    count = auto()
    data = auto()
    default_factory = auto()
    endswith = auto()
    extend = auto()
    external = auto()
    f_back = auto()
    f_code = auto()
    f_globals = auto()
    f_lineno = auto()
    f_locals = auto()
    file = auto()
    filename = auto()
    frame = auto()
    func = auto()
    function = auto()
    get_ = auto()  # value: get: To avoid conflict with N.get()
    globals = auto()
    id = auto()
    index = auto()
    ip = auto()
    item = auto()
    items = auto()
    key = auto()
    keys = auto()
    kind = auto()
    lineno = auto()
    locals = auto()
    name = auto()
    node = auto()
    origin = auto()
    obj = auto()
    object = auto()
    path = auto()
    repo = auto()
    rv = auto()
    pop = auto()
    popitem = auto()
    pypi = auto()
    remove = auto()
    reverse = auto()
    self_ = auto()  # value: 'self': To avoid conflict with Enum
    sort = auto()
    source = auto()
    startswith = auto()
    tb_frame = auto()
    tb_lineno = auto()
    tb_next = auto()
    update = auto()
    value = auto()
    values = auto()
    vars = auto()


class PathGit(EnumBase):
    PATH = 'git rev-parse --show-toplevel'
    ORIGIN = 'git config --get remote.origin.url'
    ORGANIZATION = f'git config --get remote.{GITHUB_ORGANIZATION}.url'

    def cmd(self, path=None):
        rv = None
        if (path and ((path := Path(path).resolved).exists() or (path := Path.cwd() / path).resolve().exists())) \
                or (path := Path.cwd().resolved):
            with Path(path).cd:
                if path := subrun(shsplit(self.value), capture_output=True, text=True).stdout.removesuffix('\n'):
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


class PathInstallScript(SetUpToolsInstall):
    def run(self):
        # does not call install.run() by design
        # noinspection PyUnresolvedReferences
        self.distribution.install_scripts = self.install_scripts

    @classmethod
    def path(cls):
        dist = SetUpToolsDistribution({'cmdclass': {'install': cls}})
        dist.dry_run = True  # not sure if necessary, but to be safe
        dist.parse_config_files()
        command = dist.get_command_obj('install')
        command.ensure_finalized()
        command.run()
        return dist.install_scripts


class PathIs(EnumBase):
    DIR = 'is_dir'
    FILE = 'is_file'


class PathMode(EnumBase):
    DIR = 0o666
    FILE = 0o777
    X = 0o755


class PathOption(EnumBase):
    BOTH = auto()
    DIRS = auto()
    FILES = auto()


class PathOutput(EnumBase):
    BOTH = 'both'
    BOX = Box
    DICT = dict
    LIST = list
    NAMED = namedtuple
    TUPLE = tuple


class PathSuffix(EnumBase):
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


class Path(PathLib, pathlib.PurePosixPath):
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
               f'{shquote(self.resolved.text)}')
        return self

    def chown(self, group=None, u=None):
        system(f'{user.sudo("chown", SUDO_DEFAULT)} {u or user.name}:{group or user.gname} '
               f'{shquote(self.resolved.text)}')
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
            system(f'{user.sudo("mkdir", su)} -p -m {mode or 755} {shquote(p.text)}')
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

            system(f'{user.sudo("touch", su)} {shquote(p.text)}')
        if file:
            raise NotADirectoryError(f'{file=} is file and not dir', f'{(self / (name or str())).resolved}')
        p.chmod(mode=mode)
        p.chown(group=group, u=u)
        return p



class Re(EnumBase):
    ALL = Access.ALL.value
    ASYNCDEF = re.compile(r'^(\s*async\s+def\s)')
    BLOCK = re.compile(r'^(\s*def\s)|(\s*async\s+def\s)|(.*(?<!\w)lambda(:|\s))|^(\s*@)')
    DEF = re.compile(r'^(\s*def\s)')
    DECORATOR = re.compile(r'^(\s*@)')
    LAMBDA = re.compile(r'^(.*(?<!\w)lambda(:|\s))')
    PRIVATE = Access.PRIVATE.value
    PROTECTED = Access.PROTECTED.value
    PUBLIC = Access.PUBLIC.value


class Source(executing.Source):
    """Source Class."""
    __slots__ = ('_nodes_by_line', '_qualnames', 'filename', 'lines', 'text', 'tree',)
    def __init__(self, filename, lines): super().__init__(filename, lines)


class SourceNode:
    """Source Node Class."""
    __slots__ = ('code', 'code_line', 'complete', 'complete_line', 'context', 'es', 'index',)

    def __init__(self, es):
        self.es = es
        self.code = self.get()
        self.code_line = self.get(line=True)
        self.complete = self.get(complete=True)
        self.complete_line = self.get(complete=True, line=True)
        self.context = tuple(self.es.data.code_context) if self.es.frameinfo else tuple()
        self.index = self.es.data.index if self.es.frameinfo else int()

    def __repr__(self): return f'{self.__class__.__name__}({self.es.data})'

    @functools.cache
    def get(self, complete=False, line=False):
        """
        Get source of object.

        Args:
            complete: return complete source file (always for module and frame corresponding to module)
                or object source (default=False)
            line: return line

        Returns:
            Source str.
        """
        if any([self.es.moduletype, (self.es.frameinfo and self.es.data.function == FUNCTION_MODULE),
                (self.es.frametype and self.es.data.f_code.co_name == FUNCTION_MODULE) or
                (self.es.tracebacktype and self.es.data.tb_frame.f_code.co_name == FUNCTION_MODULE) or
                complete]):
            if source := self.open(line):
                return source

        try:
            if line:
                lines, lnum = getsourcelines(self.es.data.frame if self.es.frameinfo else self.es.data)
                return ''.join(lines), lnum
            return getsource(self.es.data.frame if self.es.frameinfo else self.es.data)
        except (OSError, TypeError):
            if source := self.open(line):
                return source
            if line:
                return str(self.es.data), 1
            return str(self.es.data)

    @functools.cache
    def node(self, complete=False, line=False):
        """
        Get node of object.


        Args:
            complete: return complete node for file (always for module and frame corresponding to module)
                or object node (default=False)
            line: return line

        Returns:
            Node.
        """
        return ast.parse(self.get(complete=complete, line=line) or str(self.es.data))

    @functools.cache
    def open(self, line=False):
        f = self.es.getf(N.FILE.value) or self.es.data
        try:
            if (p := PathLib(f)).is_file():
                if s := token_open(p):
                    if line:
                        return s, 1
                    return s
        except OSError:
            if line:
                return str(self.es.data), 1
            return str(self.es.data)


class UserActual:
    """User Actual Class."""
    ROOT = None
    SUDO = None
    SUDO_USER = None

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
    id = os.getuid()
    gecos = pwd.getpwuid(id).pw_gecos
    gid = os.getgid()
    gname = grp.getgrgid(gid).gr_name
    home = Path(pwd.getpwuid(id).pw_dir).resolve()
    name = pwd.getpwuid(id).pw_name
    passwd: pwd.struct_passwd = pwd.getpwuid(id)
    ROOT = not id
    shell = Path(pwd.getpwuid(id).pw_shell).resolve()
    ssh = home / SSH_DIR
    SUDO = SUDO
    SUDO_USER = SUDO_USER
    auth_keys = ssh / AUTHORIZED_KEYS
    id_rsa = ssh / ID_RSA
    id_rsa_pub = ssh / ID_RSA_PUB
    git_config_path = home / GITCONFIG
    git_config: GitConfigParser = GitConfigParser(str(git_config_path))
    github_username = git_config.get_value(section='user', option='username', default=str())


class User:
    """User Class."""
    actual: UserActual = UserActual()
    process: UserProcess = UserProcess()
    gecos = process.gecos if SUDO else actual.gecos
    gid = process.gid if SUDO else actual.gid
    gname = process.gname if SUDO else actual.gname
    home = process.home if SUDO else actual.home
    id = process.id if SUDO else actual.id
    name = process.name if SUDO else actual.name
    passwd: pwd.struct_passwd = process.passwd if SUDO else actual.passwd
    ROOT = UserProcess.ROOT
    shell = process.shell if SUDO else actual.shell
    ssh = process.ssh if SUDO else actual.ssh
    SUDO = UserProcess.SUDO
    SUDO_USER = UserProcess.SUDO_USER
    auth_keys = process.auth_keys if SUDO else actual.auth_keys
    id_rsa = process.id_rsa if SUDO else actual.id_rsa
    id_rsa_pub = process.id_rsa_pub if SUDO else actual.id_rsa_pub
    git_config_path = process.git_config_path if SUDO else actual.git_config_path
    git_config: GitConfigParser = process.git_config if SUDO else actual.github_username
    github_username = process.github_username if SUDO else actual.github_username
    GIT_SSH_COMMAND = f'ssh -i {str(id_rsa)} {SSH_CONFIG_TEXT}'
    os.environ['GIT_SSH_COMMAND'] = GIT_SSH_COMMAND
    __contains__ = lambda self, item: item in self.name
    __eq__ = lambda self, other: self.name == other.name
    __hash__ = lambda self: hash(self.name)
    sudo = staticmethod(lambda command, su=False: command if SUDO or not su else f'sudo {command}')


# </editor-fold>
# <editor-fold desc="Classes: Deps">
class Attributes(dict, MutableMapping):
    """Attributes :class:`rc.utils.Attribute` Dict Class."""
    STATE = (N.ALLOC.value, N.ANNOTATIONS.value, N.ARGS.value, N.ARGS.value, N.BUILD_CLASS.value, N.BUILTINS.value,
             N.CLASS.value, N.DICT.value, N.DOC.value, N.HASH.value,
             N.MODULE.value, N.SLOTS.value, N.WEAKREF.value)
    __slots__ = ()
    def __new__(cls, *args, **kwargs): return super().__new__(cls, *args, **kwargs)
    def __hash__(self): return hash(tuple(self.keys()))
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __repr__(self): return f'{self.__class__.__name__}{super().__repr__()}'

    @property
    def data_all(self):
        """
        Data Access.ALL Attributes.

        Examples:
            >>> from rc import Es, TestDataDictSlotMix, pretty_install
            >>>
            >>> pretty_install()
            >>>
            >>> rv = Es(TestDataDictSlotMix).attributes().data_all
            >>> list(rv) # doctest: +ELLIPSIS
            [
                '_TestData__data',
                '_TestData__dataclass_classvar',
                '_TestData__dataclass_default_factory',
                '_TestData__dataclass_default_factory_init',
                '__annotations__',
                '__class__',
                '__dataclass_classvar__',
                '__dataclass_fields__',
                '__dataclass_params__',
                '__dict__',
                '__doc__',
                '__hash__',
                '__module__',
                '__slots__',
                '__weakref__',
                '_slot_property',
                'dataclass_classvar',
                'dataclass_default',
                'dataclass_default_factory',
                'dataclass_default_factory_init',
                'dataclass_default_init',
                'dataclass_initvar',
                'dataclass_str',
                'slot',
                'subclass_annotated_str',
                'subclass_classvar',
                'subclass_str'
            ]

        Returns:
            Data Access.ALL Attributes.
        """
        return self.filter(include=Access.ALL, value=lambda x: x.kind == Kind.DATA)

    @property
    def data_protected(self):
        """
        Data Access.PROTECTED Attributes.

        Examples:
            >>> from rc import Es, TestDataDictSlotMix, pretty_install
            >>>
            >>> pretty_install()
            >>>
            >>> rv = Es(TestDataDictSlotMix).attributes().data_protected
            >>> list(rv) # doctest: +ELLIPSIS
            [
                '_TestData__data',
                '_TestData__dataclass_classvar',
                '_TestData__dataclass_default_factory',
                '_TestData__dataclass_default_factory_init',
                '_slot_property',
                'dataclass_classvar',
                'dataclass_default',
                'dataclass_default_factory',
                'dataclass_default_factory_init',
                'dataclass_default_init',
                'dataclass_initvar',
                'dataclass_str',
                'slot',
                'subclass_annotated_str',
                'subclass_classvar',
                'subclass_str'
            ]

        Returns:
            Data Access.ALL Attributes.
        """
        return self.filter(value=lambda x: x.kind == Kind.DATA)

    @functools.cache
    def filter(self, include=Access.PROTECTED, key=lambda x: True, value=lambda x: True):
        """
        Class Methods.

        Examples:
            >>> from rc import Es, TestDataDictSlotMix, pretty_install
            >>>
            >>> pretty_install()
            >>>
            >>> test = Es(TestDataDictSlotMix)
            >>> attributes = test.attributes()
            >>> attributes # doctest: +ELLIPSIS
            Attributes{'_TestData__data': Attribute(...
            >>> list(attributes.filter(include=Access.PUBLIC)) # doctest: +ELLIPSIS
            [
                'dataclass_classvar',
                ...
            ]
            >>> list(attributes.filter()) # doctest: +ELLIPSIS
            [
                '_TestData__data',
                ...
            ]
            >>> list(attributes.filter(include=Access.ALL)) # doctest: +ELLIPSIS
            [
                ...,
                '__class__',
                ...
            ]

        Args:
            include: include attributes (default: Access.PROTECTED).
            key: predicate for keys.
            value: predicate for values.

        Returns:
            Filtered Attributes.
        """
        return type(self)(filter(lambda x: x[1].access >= include and key(x[0]) and value(x[1]), self.items()))

    @property
    def state(self):
        """
        State Access.ALL Attributes.

        Examples:
            >>> from rc import Es, TestDataDictSlotMix, pretty_install, Environs
            >>> from inspect import classify_class_attrs
            >>>
            >>> pretty_install()
            >>>
            >>> rv = Es(TestDataDictSlotMix()).attributes().state
            >>> list(rv) # doctest: +ELLIPSIS
            [
                '_TestData__dataclass_default_factory',
                '_TestData__dataclass_default_factory_init',
                '_slot_property',
                'dataclass_default_factory',
                'dataclass_default_factory_init',
                'dataclass_default_init',
                'dataclass_str',
                'slot',
                'subclass_dynamic'
            ]
            >>> rv = Es(TestDataDictSlotMix()).attributes().state
            >>> es = Es(Environs())
            >>> state = es.attributes().state
            >>> assert sorted(state.keys()) == sorted(es.state().keys())

        Returns:
            State Access.ALL Attributes.

        """
        return self.filter(include=Access.ALL, value=lambda x: x.state is True)


class Es(Is):
    # noinspection PyUnresolvedReferences
    """
    Is Instance, Subclass, etc. Helper Class

    Examples:
        >>> from ast import unparse
        >>> from inspect import getmodulename
        >>> from rc import *
        >>> import rc.utils
        >>>
        >>> pretty_install()
        >>>
        >>> frameinfo = insstack()[0]
        >>> globs_locs = frameinfo.frame.f_globals | frameinfo.frame.f_locals
        >>> '__file__' in globs_locs  # doctest in properties do not have globals.
        True
        >>> sourcenode_file = Es(rc.utils.__file__).sourcenode
        >>> sourcenode_info = Es(frameinfo).sourcenode
        >>> sourcenode_vars = Es(globs_locs).sourcenode
        >>> sourcenode_file.code  # doctest: +ELLIPSIS
        '# -*- coding: utf-8 -*-\\n...
        >>> sourcenode_vars.code  # doctest: +ELLIPSIS
        '# -*- coding: utf-8 -*-\\n...
        >>> assert sourcenode_file.code == sourcenode_vars.code
        >>> assert unparse(sourcenode_file.node()) == unparse(sourcenode_vars.node()) \
        # Unparse does not have encoding line
        >>> assert sourcenode_info.code in unparse(sourcenode_file.node())
        >>> assert sourcenode_info.code in unparse(sourcenode_vars.node())
        >>>
        >>> unparse(Es('pass').sourcenode.node())
        'pass'
        >>>
        #
        # Frame
        #
        >>> frametype = insstack()[0].frame
        >>> es = Es(frametype)
        >>> frame = es.frame
        >>>
        # >>> frame.asttokens
        # >>> frame.executing

    Attributes:
    -----------
        data: Any
            object to provide information (default: None)
    """
    __slots__ = ('_annotations', '_attributes', '_classified', '_classified_dict', '_stack_index', 'data')

    def __init__(self, data=None): super().__init__(data)

    def __call__(self, *args): return isinstance(self.data, args)

    def __getstate__(self): return dict(data=self.data)

    def __hash__(self): return hash((type(self.data), self.__str__()))

    def __reduce__(self): return self.__class__, tuple(self.data)

    def __reduce_ex__(self, *args, **kwargs): return self.__class__, tuple(self.data)

    def __repr__(self): return f'{self.__class__.__name__}({self.data})'

    def __setstate__(self, state): self.data = state['data']

    def __str__(self): return str(self.data)

    # <editor-fold desc="Es - Bool">
    def getsetdescriptor(self, n=None): return isgetsetdescriptor(self.es_cls.get(n) if n else self.data)

    def has(self, name):
        """
        Has Attribute/Key.

        Examples:
            >>> from rc import Es, N, pretty_install
            >>> pretty_install()
            >>>
            >>> class Test:
            ...     __repr_newline__ = True
            >>> assert Es(Test()).has(N.REPR_NEWLINE.value) is True
            >>> assert Es(Test()).has(N.REPR_EXCLUDE.value) is False
            >>> assert Es(dict(a=1)).has('a') is True
            >>> assert Es(dict(a=1)).has('b') is False

        Args:
            name: attribute or key name.

        Returns:
            True object/dict has attribute/key.
        """
        return name in self.data if self.mm else hasattr(self.data, name)

    def in_prop(self, name, exclude_coro=True):
        """
        Is attribute in self.data.__class__.__dict__ and is a property and is not coro?

        Examples:
            >>> pretty_install()
            >>>
            >>> class First:
            ...     __slots__ = ('_data', )
            ...     @property
            ...     def data(self):
            ...         return self
            >>>
            >>> class Test(First):
            ...     __slots__ = ('_id', '_coro')
            ...     @property
            ...     async def coro(self):
            ...         return self
            >>>
            >>> Es(Test()).in_prop(N.data)
            True
            >>> Es(Test()).in_prop('coro')
            False
            >>> Es(Test()).in_prop('coro', exclude_coro=False)
            True
            >>> Es(Test()).in_prop(N._id)
            False
            >>> Es(Test()).in_prop(N._ip)
            False

        Args:
            name: attribute name.
            exclude_coro: consider coro properties as no property.

        Returns:
            True if attribute in self.data.__class__.__dict__ and is a property.
        """
        name = enumvalue(name)
        if (value := self.mro_first_dict(name)) is not NotImplemented:
            es = type(self)(value)
            return es.property_any and (not es.coro if exclude_coro else True)
        return False

    def public_prop(self, name, exclude_coro=True):
        """
        When name is :class:`rc.Access.Protected` checks if it has a property removing prefix '_'.

        Examples:
            >>> pretty_install()
            >>>
            >>> class First:
            ...     __slots__ = ('_data', )
            ...     @property
            ...     def data(self):
            ...         return self
            >>>
            >>> class Test(First):
            ...     __slots__ = ('_id', '_coro')
            ...     @property
            ...     async def coro(self):
            ...         return self
            >>>
            >>> Es(Test()).public_prop(N._data)
            True
            >>> Es(Test()).public_prop('_coro')
            False
            >>> Es(Test()).public_prop('_coro', exclude_coro=False)
            True
            >>> Es(Test()).public_prop(N._id)
            False
            >>> Es(Test()).public_prop(N._ip)
            False

        Args:
            name: attribute name.
            exclude_coro: consider coro properties as no property.

        Returns:
            True if protected and has a prop removing prefix '_'.
        """
        name = enumvalue(name)
        access = Access[name]
        return access is Access.PROTECTED and self.in_prop(name.removeprefix('_'), exclude_coro=exclude_coro)

    def readonly(self, name='__readonly__'):
        """
        Is readonly object?

        Examples:
            >>> from rc import Es, pretty_install
            >>> pretty_install()
            >>>
            >>> class First:
            ...     __slots__ = ('_data', )
            >>>
            >>> class Test(First):
            ...     __slots__ = ('_id', )
            >>>
            >>> Es(Test()).readonly()
            AttributeError("'Test' object has no attribute '__readonly__'")

        Returns:
            True if readonly.
        """
        name = enumvalue(name)
        value = None
        try:
            if self.has(name):
                value = object.__getattribute__(self.data, name)
            object.__setattr__(self.data, name, value)
        except Exception as exception:
            return exception
        if value is not None:
            object.__delattr__(self.data, name)
        return False

    def slot(self, name):
        """
        Is attribute in slots?

        Examples:
            >>> pretty_install()
            >>>
            >>> class First:
            ...     __slots__ = ('_data', )
            >>>
            >>> class Test(First):
            ...     __slots__ = ('_id', )
            >>>
            >>> Es(Test()).slot(N._data)
            True
            >>> Es(Test()).slot(N._id)
            True
            >>> Es(Test()).slot(N._ip)
            False

        Returns:
            True if attribute in slots
        """
        name = enumvalue(name)
        return name in self.slots

    def writeable(self, name):
        # noinspection PyUnresolvedReferences
        """
        Checks if an attr is writeable in object.

        Examples:
            >>> from rc import Es, pretty_install
            >>> pretty_install()
            >>>
            >>> class First:
            ...     __slots__ = ('_data', )
            >>>
            >>> class Test(First):
            ...     __slots__ = ('_id', )
            >>>
            >>> test = Test()
            >>> es = Es(test)
            >>> es.writeable('_id')
            True
            >>> es.writeable('test')
            False
            >>> Es(test.__class__).writeable('test')
            True
            >>> test.__class__.test = 2
            >>> assert test.test == 2
            >>> Es(test.__class__).writeable('cls')
            True
            >>> object.__setattr__(test.__class__, 'cls', 2) # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            TypeError: can't apply this __setattr__ to type object

        Args:
            name: attribute name.

        Returns:
            True if can be written.
        """
        return self.dicttype or self.slotstype and self.slot(name) or not self.readonly(name)
    # </editor-fold>

    # <editor-fold desc="Es - Inspect">
    def annotations(self, stack=2):
        if hasattr(self, '_annotations') and getset(self, '_stack_index', setvalue=False) == stack:
            return self._annotations
        getset(self, '_stack_index', stack)
        getset(self, '_annotations', annotations(self.data, stack=stack))
        return self._annotations

    def attributes(self, stack=2, prop=True, repr_=Access.PROTECTED, exclude_coro=True):
        # TODO: Examples and classify attrs props etc. defaults, backup/infocls1.py
        """
        Attribute Information for Object Class or Class.

        Examples:
            >>> pretty_install()
            >>>
            >>> test = Es(TestDataDictSlotMix)
            >>> attr = test.attributes()['_TestData__dataclass_default_factory']
            >>> attr.default, attr.defining, attr.field.default_factory, attr.object, \
            attr.field.data.type == attr.annotation.hint, attr.object == attr.annotation.default  # doctest: +ELLIPSIS
            (NotImplemented, <class '....TestData'>, True, {}, True, True)
            >>> attr.object, attr.annotation.default
            ({}, {})

        Args:
            stack: stack index to extract globals and locals (default: 1) or frame.
            prop: get value of not coro properties.
            repr_: repr include.
            exclude_coro: consider coro properties as no property.

        Returns:
            Dict Attribute Information for Object Class or Class.
        """
        def _rv(_defining: Type, _name: str, _object: Any, _kind: str = 'data') -> dict[str, Attribute]:
            _access = Access[_name]
            _es = cls(_object)
            _coro = _es.coro
            _dict = _name in instance
            _slot = _es.memberdescriptortype
            _dictslot = _dict or _slot
            _classvar = _kind is Kind.DATA and not _dictslot
            _field = fields.pop(_name, None)
            _es_field = cls(_field)
            _kind = Kind[_kind.split(' ')[0].upper()]
            _default = defaults.get(_name) if _name in defaults else _object \
                if ((_field is None and _kind is Kind.DATA) or (_field and _field.init)) else NotImplemented
            _init = _default is not NotImplemented and _name not in ignore_kwarg
            _copy = _hash = _ignore_attr = _publicprop = _repr = _state = _str = _value = TypeError
            if not t:
                _ignore_attr = _name in ignore_attr
                _hash = _kind is Kind.DATA and _name not in hash_exclude
                _publicprop = self.public_prop(_name, exclude_coro=exclude_coro)
                _repr = _access >= repr_ and _kind is Kind.DATA and _name not in repr_exclude
                _state = _kind is Kind.DATA and _dictslot
                if _ignore_attr:
                    _value = NotImplemented
                else:
                    if _name in instance:
                        _value = instance[_name]
                    elif (_kind is Kind.PROPERTY and prop and not _coro) or (
                            _kind is not Kind.PROPERTY and hasattr(self.data, _name)):
                        _value = getattr(self.data, _name)
                    else:
                        _value = AttributeError
                if _value is not NotImplemented and _value is not AttributeError:
                    _value_mro = cls(_value).mro
                    _copy = not bool(anyin(_value_mro, ignore_copy))
                    _str = not bool(anyin(_value_mro, ignore_str))

            return {_name: Attribute(
                access=_access,
                annotation=hints.get(_name),
                classvar=_classvar,
                copy=_copy,
                coro=_coro,
                dataclass=d,
                default=_default,
                defaultfactory=_es_field.default_factory,
                defining=_defining,
                dict=_dict,  # in instance __dict__.
                dictslot=_dictslot,  # in instance __dict__ or __slots__.
                es=_es,
                field=_es_field,
                hash=_hash,
                ignore=_ignore_attr,
                init=_init,  # do not rely !!
                initvar=_es_field.initvar,
                kind=_kind,
                name=k,
                named=n,
                object=_object,
                publicprop=_publicprop,  # protected with property.
                qual=cls(_es._func).get(N.QUALNAME.value, default=_name),
                repr=_repr,
                slot=_slot,  # in __slots__.
                state=_state,  # include in __getstate__
                str=_str,
                type=t,
                value=_value,  # instance value
            )}

        if hasattr(self, '_attributes') and getset(self, '_stack_index', setvalue=False) == stack:
            return self._attributes
        cls = type(self)
        d = self.datatype_any
        n = self.namedtype or self.namedtype_sub
        t = self.type
        classified = self.classified_dict
        defaults = self.data._field_defaults if n else {}
        fields = self.fields.copy() if d else {}
        hints = self.annotations(stack=stack) if self.annotationstype_any else dict()
        hash_exclude = ignore_attr = ignore_copy = ignore_kwarg = ignore_str = repr_exclude = ()
        instance = {}
        if not t:
            hash_exclude = self.mro_values(N.HASH_EXCLUDE)
            ignore_attr = self.mro_values(N.IGNORE_ATTR)
            ignore_copy = self.mro_values(N.IGNORE_COPY)
            ignore_kwarg = self.mro_values(N.IGNORE_KWARG)
            ignore_str = self.mro_values(N.IGNORE_STR)
            repr_exclude = self.mro_values(N.REPR_EXCLUDE)
            instance = vars(self.data) if self.dicttype else {}
        names = sorted({*classified, *fields, *instance})
        rv = Attributes()
        for k in names:
            if k in classified:
                v = classified[k]
                rv |= _rv(_defining=v.defining_class, _name=k, _object=v.object, _kind=v.kind)
            elif k in fields:
                v = fields[k]
                defining = self.cls
                for C in self.mro:
                    es_C = cls(C)
                    if not es_C.datatype_sub or (es_C.datatype_sub and k not in es_C.fields):
                        break
                    defining = C
                obj = v.default_factory() if cls(v).default_factory else cls(defining).get(k)
                rv |= _rv(_defining=defining, _name=k, _object=obj)
            else:
                rv |= _rv(_defining=NotImplemented, _name=k, _object=instance[k])
        getset(self, '_stack_index', stack)
        getset(self, '_attributes', rv)
        return self._attributes

    @property
    def classified(self): return self._classified if hasattr(self, '_classified') else \
        getset(self, '_classified', tuple(classify_class_attrs(self.cls)))

    @property
    def classified_dict(self): return self._classified_dict if hasattr(self, '_classified_dict') else \
        getset(self, '_classified_dict', {item.name: item for item in self.classified})

    @property
    def cls(self): return self.data if self.type else self.data.__class__

    @property
    def clsname(self): return self.cls.__name__

    @property
    def clsqual(self): return self.cls.__qualname__

    @property
    def es_cls(self): return type(self)(self.cls)

    @property
    def fields(self): return self.data.__dataclass_fields__ if self.datatype_any else {}

    @property
    def first_builtin(self): return anyin(self.mro, BUILTINS_CLASSES)

    @property
    def importable_name_cls(self): return noexception(importable_name, self.cls)

    @property
    def importable_name_data(self): return noexception(importable_name, self.data)

    @property
    def modname(self):
        return self.__name__ if self.moduletype else self.__module__ if self.has(N.MODULE.value) else None

    @property
    def mro(self): return self.cls.__mro__

    @property
    def mro_and_data(self):
        """
        Tuple with Class MRO and instance (self.data).

        Examples:
            >>> from rc import Es, pretty_install
            >>> pretty_install()
            >>>
            >>> Es(dict(a=1)).mro_and_data
            ({'a': 1}, <class 'dict'>, <class 'object'>)

        Returns:
            Tuple with Class MRO and instance (self.data).
        """
        return (() if self.type else (self.data, )) + self.mro

    def mro_first_data(self, name):
        """
        First value of attr found in mro and instance if obj is instance.

        Examples:
            >>> from rc import Es, N, pretty_install
            >>> pretty_install()
            >>>
            >>> class Test:
            ...     __repr_newline__ = True
            >>>
            >>> test = Test()
            >>> class Test2(Test):
            ...     def __init__(self):
            ...         self.__repr_newline__ = False
            >>>
            >>> Es(Test()).mro_first_data(N.REPR_NEWLINE)
            True
            >>> Es(Test2()).mro_first_data(N.REPR_NEWLINE)
            False
            >>> Es(int()).mro_first_data(N.REPR_NEWLINE)
            >>> Es(Test()).mro_first_data(N.REPR_PPROPERTY)

        Returns:
            First value of attr found in mro and instance if obj is instance.
        """
        name = enumvalue(name)
        for item in self.mro_and_data:
            if type(self)(item).has(name):
                return item.__getattribute__(name)

    def mro_first_dict(self, name, mro=None):
        """
        First value of attr in obj.__class__.__dict__ found in mro.

        Examples:
            >>> from rc import Es, N, pretty_install
            >>> pretty_install()
            >>>
            >>> class Test:
            ...     __repr_newline__ = False
            >>>
            >>> test = Test()
            >>> class Test2(Test):
            ...     def __init__(self):
            ...         self.__repr_newline__ = False
            >>>
            >>> Es(Test()).mro_first_dict(N.REPR_NEWLINE)
            False
            >>> Es(Test2()).mro_first_dict(N.REPR_NEWLINE)
            False
            >>> Es(int()).mro_first_dict(N.REPR_NEWLINE)
            NotImplemented
            >>> Es(Test()).mro_first_dict(N.REPR_PPROPERTY)
            NotImplemented
            >>> A = namedtuple('A', 'a')
            >>> Es(A).mro_first_dict(N.SLOTS)
            ()

        Args:
            name: attribute name.
            mro: mro or search in dict (default: self.mro)

        Returns:
            First value of attr in obj.__class__.__dict__ found in mro.
        """
        name = enumvalue(name)
        for C in mro or self.mro:
            if name in C.__dict__:
                return C.__dict__[name]
        return NotImplemented

    def mro_first_dict_no_object(self, name):
        """
        First value of attr in obj.__class__.__dict__ found in mro excluding object.

        Examples:
            >>> from rc import Es, N, pretty_install
            >>> pretty_install()
            >>>
            >>> class Test:
            ...     __repr_newline__ = False
            >>>
            >>> test = Test()
            >>> class Test2(Test):
            ...     def __init__(self):
            ...         self.__repr_newline__ = False
            >>>
            >>> Es(Test()).mro_first_dict_no_object(N.REPR_NEWLINE)
            False
            >>> Es(Test2()).mro_first_dict_no_object(N.REPR_NEWLINE)
            False
            >>> Es(int()).mro_first_dict_no_object(N.REPR_NEWLINE)
            NotImplemented
            >>> Es(Test()).mro_first_dict_no_object(N.REPR_PPROPERTY)
            NotImplemented
            >>> A = namedtuple('A', 'a')
            >>> Es(A).mro_first_dict_no_object(N.SLOTS)
            ()
            >>> Es(dict).mro_first_dict_no_object(N.SLOTS)
            NotImplemented
            >>> Es(dict()).mro_first_dict_no_object(N.SLOTS)
            NotImplemented

        Returns:
            First value of attr in obj.__class__.__dict__ found in mro excluding object.
        """
        mro = list(self.mro)
        mro.remove(object)
        return self.mro_first_dict(enumvalue(name), tuple(mro))

    def mro_values(self, name):
        """
        All/accumulated values of attr in mro and obj if instance.

        Examples:
            >>> from rc import Es, pretty_install
            >>> pretty_install()
            >>>
            >>> class First:
            ...     __slots__ = ('_hash', '_repr')
            ...     __ignore_copy__ = (tuple, )
            ...     __repr_exclude__ = ('_repr', )
            >>>
            >>> class Test(First):
            ...     __slots__ = ('_prop', '_slot', )
            ...     __hash_exclude__ = ('_slot', )
            ...     __ignore_attr__ = ('attr', )
            ...     __ignore_kwarg__ = ('kwarg', )
            ...     __ignore_str__ = (tuple, )
            >>>
            >>> test = Test()
            >>> Es(test).mro_values(N.HASH_EXCLUDE)
            ('_slot',)
            >>> Es(test).mro_values(N.IGNORE_ATTR)  # doctest: +ELLIPSIS
            (
                'asdict',
                'attr',
                ...
            )
            >>> set(Es(test).mro_values(N.IGNORE_COPY)).difference(Es().mro_values_default(\
N.IGNORE_COPY))
            {<class 'tuple'>}
            >>> Es(test).mro_values(N.IGNORE_KWARG)
            ('kwarg',)
            >>> set(Es(test).mro_values(N.IGNORE_STR)).difference(Es().mro_values_default(\
N.IGNORE_STR))
            {<class 'tuple'>}
            >>> Es(test).mro_values(N.REPR_EXCLUDE)
            ('_repr',)
            >>> Es(test).mro_values(N.SLOTS)
            ('_hash', '_prop', '_repr', '_slot')
            >>> assert sorted(Es(Environs()).mro_values(N.STATE)) == sorted(STATE_ATTRS[Environs])

        Returns:
            All/accumulated values of attr in mro and obj if instance.
        """
        name = enumvalue(name)
        values = tuple({*(value for item in self.mro_and_data
                          for value in type(self)(item).get(name, default=tuple())), *self.mro_values_default(name)})
        return tuple(sorted(values)) if values and not type(self)(tuple(values)[0]).type else values

    def mro_values_default(self, name):
        """
        Default values for attr in mro and instance.

        Examples:
            >>> from rc import Es, pretty_install
            >>> pretty_install()
            >>>
            >>> class First:
            ...     __slots__ = ('_hash', '_repr')
            ...     __ignore_copy__ = (tuple, )
            ...     __repr_exclude__ = ('_repr', )
            >>>
            >>> class Test(First):
            ...     __slots__ = ('_prop', '_slot', )
            ...     __hash_exclude__ = ('_slot', )
            ...     __ignore_attr__ = ('attr', )
            ...     __ignore_kwarg__ = ('kwarg', )
            ...     __ignore_str__ = (tuple, )
            >>>
            >>> test = Test()
            >>> assert Es(test).mro_values_default(N.HASH_EXCLUDE) == tuple()
            >>> assert Es(test).mro_values_default(N.IGNORE_ATTR) == IgnoreAttr.__args__
            >>> assert Es(test).mro_values_default(N.IGNORE_COPY) == IgnoreCopy.__args__
            >>> assert Es(test).mro_values_default(N.IGNORE_KWARG) == tuple()
            >>> assert Es(test).mro_values_default(N.IGNORE_STR) == IgnoreStr.__args__
            >>> assert Es(test).mro_values_default(N.REPR_EXCLUDE) == tuple()
            >>> assert Es(test).mro_values_default(N.SLOTS) == tuple()
            >>> assert sorted(Es(Environs()).mro_values_default(N.STATE)) == sorted(STATE_ATTRS[Environs])

        Returns:
           Default values for attr in mro and instance.
        """
        if enumvalue(name) == N.STATE.value:
            default = set()
            for C in self.mro:
                if C in STATE_ATTRS:
                    default.update(STATE_ATTRS[C])
        else:
            es = Es(name)
            default = rv.__args__ if (rv := globals().get(to_camel(name.name if es.enum else name))) else tuple()
        return tuple(sorted(default)) if default and not type(self)(tuple(default)[0]).type else tuple(default)

    @property
    def name(self): return self.data.__name__ if self.has(N.NAME.value) else None

    @classmethod
    def prop_getter(cls, prop):
        es = cls(prop)
        if all([es.property_any, (func := es._func), (name := cls(func).name), name in es.cls.__dict__]):
            return attrgetter(name)
        return NotImplemented

    @property
    def slots(self):
        """
        Slots values in mro.

        Examples:
            >>> from rc import Es, pretty_install
            >>> pretty_install()
            >>>
            >>> class First:
            ...     __slots__ = ('_data', )
            >>>
            >>> class Test(First):
            ...     __slots__ = ('_id', )
            >>>
            >>> assert Es(Test()).slots == First.__slots__ + Test.__slots__

        Returns:
            Slots values in mro Tuple.
        """
        return self.mro_values(N.SLOTS)

    def slots_include(self, name):
        """
        Accumulated values from slots - Accumulated values from mro attr name.

        Examples:
            >>> pretty_install()
            >>>
            >>> class First:
            ...     __slots__ = ('_hash', )
            >>>
            >>> class Test(First):
            ...     __slots__ = ('_prop', '_repr', '_slot', )
            ...     __hash_exclude__ = ('_slot', )
            ...     __repr_exclude__ = ('_repr', )
            >>>
            >>> es = Es(Test())
            >>> es.slots
            ('_hash', '_prop', '_repr', '_slot')
            >>> hash_attrs = es.slots_include(N.HASH_EXCLUDE)
            >>> hash_attrs
            ('_hash', '_prop', '_repr')
            >>> sorted(hash_attrs + es.mro_values(N.HASH_EXCLUDE)) == sorted(es.slots)
            True
            >>> repr_attrs = es.slots_include(N.REPR_EXCLUDE)
            >>> repr_attrs
            ('_hash', '_prop', '_slot')
            >>> sorted(repr_attrs + es.mro_values(N.REPR_EXCLUDE)) == sorted(es.slots)
            True

        Returns:
            Accumulated values from slots - Accumulated values from mro attr name.
        """
        return tuple(sorted(set(self.slots).difference(self.mro_values(enumvalue(name)))))

    @property
    def super(self): return self.mro[1]
    # </editor-fold>

    # <editor-fold desc="Es - Utils">
    @property
    def deepcopy(self):
        """
        Deep copy object

        Tries again after calling :class:`rc.Es.state_methods` in case of PicklingError RecursionError.

        Examples:
            >>> from copy import deepcopy
            >>> from rc import Environs, Es
            >>>
            >>> deepcopy(Environs()) # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            RecursionError: maximum recursion depth exceeded
            >>> env = Environs()
            >>> term = env('TERM')
            >>> env
            <Env {'TERM': 'xterm-256color'}>
            >>> env_copy = env
            >>> assert id(env_copy) == id(env)
            >>> env_deepcopy = Es(env).deepcopy
            >>> env_deepcopy
            <Env {'TERM': 'xterm-256color'}>
            >>> assert id(env_deepcopy) != id(env)
            >>> assert id(env_deepcopy._values) != id(env._values)

        Returns:
            Deep copied object.
        """
        try:
            return copy.deepcopy(self.data)
        except (PicklingError, RecursionError):
            return copy.deepcopy(self.state_methods)

    @property
    def frame(self):
        """
        :class:`rc.utils.Frame`.

        Examples:
            >>> from ast import unparse
            >>> from inspect import getmodulename
            >>> import rc.utils
            >>> from rc import *
            >>>
            >>> pretty_install()
            >>>
            >>> frameinfo = insstack()[0]
            >>> finfo = Es(frameinfo).framebase
            >>> ftype = Es(frameinfo.frame).framebase
            >>> assert frameinfo.frame.f_code == finfo.code
            >>> assert frameinfo.frame == finfo.frame
            >>> assert frameinfo.filename == str(finfo.path)
            >>> assert frameinfo.lineno == finfo.lineno
            >>> fields_frame = list(FrameBase._fields)
            >>> fields_frame.remove('vars')
            >>> for attr in fields_frame:
            ...     assert getattr(finfo, attr) == getattr(ftype, attr)

        Returns:
            :class:`rc.utils.Frame`.
        """
        return Frame(**self.framebase._asdict())

    @property
    def framebase(self):
        """
        :class:`rc.utils.FrameBase`.

        Examples:
            >>> from ast import unparse
            >>> from inspect import getmodulename
            >>> import rc.utils
            >>> from rc import *
            >>>
            >>> pretty_install()
            >>>
            >>> frameinfo = insstack()[0]
            >>> finfo = Es(frameinfo).framebase
            >>> ftype = Es(frameinfo.frame).framebase
            >>> assert frameinfo.frame.f_code == finfo.code
            >>> assert frameinfo.frame == finfo.frame
            >>> assert frameinfo.filename == str(finfo.path)
            >>> assert frameinfo.lineno == finfo.lineno
            >>> fields_frame = list(FrameBase._fields)
            >>> fields_frame.remove('vars')
            >>> for attr in fields_frame:
            ...     assert getattr(finfo, attr) == getattr(ftype, attr)

        Returns:
            :class:`rc.utils.FrameBase`.
        """
        if not any([self.frameinfo, self.frametype, self.tracebacktype]):
            return
        if self.frameinfo:
            frame = self.data.frame
            back = frame.f_back
            lineno = self.data.lineno
        elif self.frametype:
            frame = self.data
            back = self.data.f_back
            lineno = self.data.f_lineno
        else:
            frame = self.data.tb_frame
            back = self.data.tb_next
            lineno = self.data.tb_lineno

        code = frame.f_code
        f_globals = frame.f_globals
        f_locals = frame.f_locals
        function = code.co_name
        v = f_globals | f_locals
        name = v.get(N.NAME.value) or function

        return FrameBase(back=back, code=code, frame=frame, function=function, globals=f_globals, lineno=lineno,
                         locals=f_locals, name=name, package=v.get(N.PACKAGE.value) or name.split('.')[0],
                         path=self.path, vars=v)

    @property
    def framesourcenode(self):
        """
        :class:`rc.utils.FrameSourceNode`.

        Examples:
            >>> from ast import unparse
            >>> from inspect import getmodulename
            >>> import rc.utils
            >>> from rc import *
            >>>
            >>> pretty_install()
            >>>
            >>> frameinfo = insstack()[0]
            >>> finfo = Es(frameinfo).framesourcenode
            >>> ftype = Es(frameinfo.frame).framesourcenode
            >>> assert frameinfo.frame == finfo.framebase.frame
            >>> assert frameinfo.filename == str(finfo.framebase.path)
            >>> assert frameinfo.lineno == finfo.framebase.lineno
            >>> assert frameinfo.code_context == list(finfo.sourcenode.context)
            >>> assert frameinfo.index == finfo.sourcenode.index
            >>> assert frameinfo == FrameInfo(finfo.framebase.frame, str(finfo.framebase.path), finfo.framebase.lineno,\
            finfo.framebase.function, list(finfo.sourcenode.context), finfo.sourcenode.index)
            >>> finfo.sourcenode.code
            'frameinfo = insstack()[0]\\n'
            >>> finfo.sourcenode.code_line
            ('frameinfo = insstack()[0]\\n', 0)
            >>> finfo.sourcenode.complete
            'frameinfo = insstack()[0]\\n'
            >>> finfo.sourcenode.complete_line
            ('frameinfo = insstack()[0]\\n', 0)
            >>> fields_frame = list(FrameBase._fields)
            >>> fields_frame.remove('vars')
            >>> for attr in fields_frame:
            ...     assert getattr(finfo.framebase, attr) == getattr(ftype.framebase, attr)
            >>> fields_source = list(SourceNode.__slots__)
            >>> fields_source.remove('context')
            >>> fields_source.remove('es')
            >>> for attr in fields_source:
            ...     assert getattr(finfo.sourcenode, attr) == getattr(ftype.sourcenode, attr)

        Returns:
            :class:`rc.utils.FrameSourceNode`.
        """
        if any([self.frameinfo, self.frametype, self.tracebacktype]):
            return FrameSourceNode(framebase=self.framebase, sourcenode=self.sourcenode)

    def get(self, name, default=None, setvalue=False):
        """
        Get key/attr value.

        Examples:
            >>> from rc import Es, pretty_install, N, insstack
            >>> import rc.utils
            >>> from inspect import getmodulename
            >>> from ast import unparse
            >>> pretty_install()
            >>> d = Es(dict(a=1))
            >>> simple = Es(Simple(b=2))
            >>> assert d.get('a') == 1
            >>> assert simple.get('b') == 2
            >>> assert d.get(N.data.value) is None and d.has(N.data.value) is False
            >>> assert simple.get(N.data.value) is None and simple.has(N.data.value) is False
            >>> assert d.get(N.data.value, setvalue=True) is None and d.has(N.data.value) is True
            >>> assert simple.get(N.data.value, setvalue=True) is None and simple.has(N.data.value) is True
            >>>
            >>> f = insstack()[0]
            >>> globs_locs = f.frame.f_globals | f.frame.f_locals
            >>> es = Es(globs_locs)
            >>> es.data['__file__']  # doctest: +ELLIPSIS
            '/Users/jose/....py'
            >>>
            >>> file = PathLib(es.get(N.FILE.value))
            >>> n = es.get(N.NAME.value)
            >>> spec = es.get(N.SPEC.value)
            >>>
            >>> file  # doctest: +ELLIPSIS
            PosixPath('/Users/jose/....py')
            >>> assert file == PathLib(rc.utils.__file__) == PathLib(spec.origin)
            >>> assert n == getmodulename(rc.utils.__file__) == spec.name

        Args:
            name: attribute or key name.
            default: default.
            setvalue: setvalue if not found.

        Returns:
            Value.
        """
        if self.has(name):
            return self.data.get(name, default) if self.mm else getattr(self.data, cast(str, name), default)
        if setvalue:
            return self.set(name=name, value=default)
        return default

    def getf(self, name, default=None):
        """
        Get value from: FrameInfo, FrameType, MutableMapping, TracebackType and object.

        Use :class:`rc.Es.get` for real names.

        Args:
            name: attribute or key name.
            default: default.

        Returns:
            Value.
        """
        name = enumvalue(name)
        if name in [N.FILE.value, N.filename.value]:
            return self.path
        if self.frameinfo:
            if name in [N.NAME.value, N.co_name.value, N.function.value]:
                return self.data.function
            if name in [N.lineno.value, N.f_lineno.value, N.tb_lineno.value]:
                return self.data.lineno
            if name in [N.f_globals.value, N.globals.value]:
                return self.data.frame.f_globals
            if name in [N.f_locals.value, N.locals.value]:
                return self.data.frame.f_locals
            if name in [N.frame.value, N.tb_frame.value]:
                return self.data.frame
            if name == N.vars.value:
                return self.data.frame.f_globals | self.data.frame.f_locals
            if name in [N.CODE.value, N.f_code.value]:
                return self.data.frame.f_code
            if name in [N.f_back.value, N.tb_next.value]:
                return self.data.frame.f_back
            if name == N.index.value:
                return self.data.index
            if name == N.code_context.value:
                return self.data.code_context
            if name in [N.FILE.value, N.filename.value]:
                return self.path
            return type(self)(self.data.frame.f_globals | self.data.frame.f_locals).get(name=name, default=default)
        elif self.frametype:
            if name in [N.NAME.value, N.co_name.value, N.function.value]:
                return self.data.f_code.co_name
            if name in [N.lineno.value, N.f_lineno.value, N.tb_lineno.value]:
                return self.data.f_lineno
            if name in [N.f_globals.value, N.globals.value]:
                return self.data.f_globals
            if name in [N.f_locals.value, N.locals.value]:
                return self.data.f_locals
            if name in [N.frame.value, N.tb_frame.value]:
                return self.data
            if name == N.vars.value:
                return self.data.f_globals | self.data.f_locals
            if name in [N.CODE.value, N.f_code.value]:
                return self.data.f_code
            if name in [N.f_back.value, N.tb_next.value]:
                return self.data.f_back
            if name in [N.FILE.value, N.filename.value]:
                return self.path
            return type(self)(self.data.f_globals | self.data.f_locals).get(name=name, default=default)
        elif self.tracebacktype:
            if name in [N.NAME.value, N.co_name.value, N.function.value]:
                return self.data.tb_frame.f_code.co_name
            if name in [N.lineno.value, N.f_lineno.value, N.tb_lineno.value]:
                return self.data.tb_lineno
            if name in [N.f_globals.value, N.globals.value]:
                return self.data.tb_frame.f_globals
            if name in [N.f_locals.value, N.locals.value]:
                return self.data.tb_frame.f_locals
            if name in [N.frame.value, N.tb_frame.value]:
                return self.data.tb_frame
            if name == N.vars.value:
                return self.data.tb_frame.f_globals | self.data.tb_frame.f_locals
            if name in [N.CODE.value, N.f_code.value]:
                return self.data.tb_frame.f_code
            if name in [N.f_back.value, N.tb_next.value]:
                return self.data.tb_next
            if name in [N.FILE.value, N.filename.value]:
                return self.path
            return type(self)(self.data.tb_frame.f_globals | self.data.tb_frame.f_locals).get(
                name=name, default=default)
        # MutableMapping and object
        return self.get(name=name, default=default)

    @property
    def path(self):
        """
        Get path of object.

        Examples:
            >>> from ast import unparse
            >>> from inspect import getmodulename
            >>> from rc import N, pretty_install, insstack, Es, allin
            >>> import rc.utils
            >>>
            >>> pretty_install()
            >>> frameinfo = insstack()[0]
            >>> globs_locs = (frameinfo.frame.f_globals | frameinfo.frame.f_locals).copy()
            >>> Es(N.getf).path  # doctest: +ELLIPSIS
            PosixPath('/Users/jose/....py')
            >>> Es(rc.utils.__file__).path  # doctest: +ELLIPSIS
            PosixPath('/Users/jose/....py')
            >>> Es(allin).path  # doctest: +ELLIPSIS
            PosixPath('/Users/jose/....py')
            >>> Es(dict(a=1)).path
            PosixPath("{'a': 1}")

        Returns:
            Path.
        """
        if self.mm:
            f = self.data.get(N.FILE.value)
        elif self.frameinfo:
            f = self.data.filename
        else:
            try:
                f = getsourcefile(self.data) or getfile(self.data)
            except TypeError:
                f = None
        return PathLib(f or str(self.data))

    @property
    def pickles(self):
        """
        Pickles object (dumps).

        Tries again after calling :class:`rc.Es.state_methods` in case of PicklingError RecursionError.

        Returns:
            Pickle object (bytes) or None if methods added to self.data so it can be called again.
        """
        try:
            return pickle_dumps(self.data)
        except (PicklingError, RecursionError):
            return pickle_dumps(self.state_methods)

    def set(self, name, value=None):
        """
        Get key/attr value.

        Examples:
            >>> from rc import Es, pretty_install, N
            >>> pretty_install()
            >>> dct = Es(dict(a=1))
            >>> simple = Es(Simple(b=2))
            >>> assert dct.set(N.data.value) is None and dct.has(N.data.value) is True
            >>> assert simple.set(N.data.value) is None and simple.has(N.data.value) is True


        Args:
            name: name.
            value: default.

        Returns:
            Value.
        """
        self.data.__setitem__(name, value) if self.mm else self.data.__setattr__(name, value)
        return value

    @property
    def sourcenode(self):
        """
        Get :class:`rc.utils.SourceNode` for self.data.

        Examples:
            >>> from ast import unparse
            >>> from inspect import getmodulename
            >>> from rc import Es, pretty_install, TestAsync, insstack, allin, N, PathLib
            >>> import rc.utils
            >>>
            >>> pretty_install()
            >>>

            #
            # Class Method
            #
            >>> sourcenode = Es(TestAsync.async_classmethod).sourcenode
            >>> sourcenode.code
            '    @classmethod\\n    async def async_classmethod(cls): return cls._async_classmethod\\n'
            >>> sourcenode.complete  # doctest: +ELLIPSIS
            '# -*- coding: utf-8 -*-\\n...
            >>> assert sourcenode.code.splitlines()[0] in sourcenode.complete
            >>>

            #
            # Dict
            #
            >>> sourcenode = Es(dict(a=1)).sourcenode
            >>> sourcenode.code
            "{'a': 1}"
            >>> sourcenode.complete  # doctest: +ELLIPSIS
            "{'a': 1}"
            >>> assert sourcenode.code.splitlines()[0] in sourcenode.complete
            >>>

            #
            # File
            #
            >>> sourcenode = Es(rc.utils.__file__).sourcenode
            >>> sourcenode.code  # doctest: +ELLIPSIS
            '# -*- coding: utf-8 -*-\\n...
            >>> sourcenode.complete  # doctest: +ELLIPSIS
            '# -*- coding: utf-8 -*-\\n...
            >>> assert sourcenode.code.splitlines()[0] in sourcenode.complete
            >>>

            #
            # FrameInfo
            #
            >>> frameinfo = insstack()[0]
            >>> sourcenode = Es(frameinfo).sourcenode
            >>> sourcenode.code, frameinfo.function
            ('frameinfo = insstack()[0]\\n', '<module>')
            >>> sourcenode.complete, frameinfo.function
            ('frameinfo = insstack()[0]\\n', '<module>')
            >>> assert sourcenode.code.splitlines()[0] in sourcenode.complete
            >>>
            >>> N.filename.getf(frameinfo), N.function.getf(frameinfo), N.code_context.getf(frameinfo)[0], \
            sourcenode.code # doctest: +ELLIPSIS
            (
                PosixPath('<doctest ...Es.sourcenode[...]>'),
                '<module>',
                'frameinfo = insstack()[0]\\n',
                'frameinfo = insstack()[0]\\n'
            )

            #
            # FrameType
            #
            >>> frametype = frameinfo.frame
            >>> sourcenode = Es(frametype).sourcenode
            >>> sourcenode.code, frameinfo.function
            ('frameinfo = insstack()[0]\\n', '<module>')
            >>> sourcenode.complete, frameinfo.function
            ('frameinfo = insstack()[0]\\n', '<module>')
            >>> assert sourcenode.code.splitlines()[0] in sourcenode.complete
            >>>

            #
            # Function
            #
            >>> sourcenode = Es(allin).sourcenode
            >>> sourcenode.code  # doctest: +ELLIPSIS
            'def allin(origin, destination):\\n...'
            >>> sourcenode.code_line  # doctest: +ELLIPSIS
            (
                'def allin(origin, destination):\\n...',
                ...
            )
            >>> sourcenode.complete  # doctest: +ELLIPSIS
            '# -*- coding: utf-8 -*-\\n...
            >>> sourcenode.complete_line  # doctest: +ELLIPSIS
            (
                '# -*- coding: utf-8 -*-\\n...,
                ...
            )
            >>> assert sourcenode.code.splitlines()[0] in sourcenode.complete
            >>>

            #
            # Module
            #
            >>> sourcenode = Es(rc.utils).sourcenode
            >>> sourcenode.code  # doctest: +ELLIPSIS
            '# -*- coding: utf-8 -*-\\n...
            >>> sourcenode.complete  # doctest: +ELLIPSIS
            '# -*- coding: utf-8 -*-\\n...
            >>> assert sourcenode.code.splitlines()[0] in sourcenode.complete
            >>>

            #
            # Other
            #
            >>> Es().sourcenode.code  # doctest: +ELLIPSIS
            'None'

            #
            # Globals - Locals
            #
            >>> info = insstack()[0]
            >>> globs_locs = frameinfo.frame.f_globals | frameinfo.frame.f_locals
            >>> '__file__' in globs_locs  # doctest in properties do not have globals.
            True

        Returns:
            :class:`rc.utils.SourceNode` for self.data.
        """
        return SourceNode(self)

    def state(self, data=None):
        """
        Values for :class:`rc.BaseState` methods:
            - :meth:`rc.BaseState.__getstate__`: (pickle) when state=None.
            - :meth:`rc.BaseState.__setstate__`: (unpickle) when state.

        Examples:
            >>> class Test:
            ...     __slots__ = ('attribute', )
            ...     __state__ = ('attribute', )
            >>>
            >>> test = Test()
            >>> e = Es(test)
            >>> e.slots + e.mro_values(N.STATE)
            ('attribute', 'attribute')
            >>> status = e.state()
            >>> status
            {}
            >>> reconstruct = e.state(status)
            >>> reconstruct  # doctest: +ELLIPSIS
            <utils.Test object at 0x...>
            >>> reconstruct == e.state(status)
            True
            >>>
            >>> test.attribute = 1
            >>> new = Es(test)
            >>> status = new.state()
            >>> status
            {'attribute': 1}
            >>> reconstruct == new.state(status)
            True
            >>> reconstruct.attribute
            1

        Args:
            data: dict to restore object.

        Returns:
            State dict (pickle) or restored object from state dict (unpickle).
        """
        if data is None:
            return {attr: self.data.__getattribute__(attr) for attr in self.mro_values(N.STATE.value)
                    if self.has(attr)}
        for key, value in data.items():
            self.data.__setattr__(key, value)
        return self.data

    @property
    def state_methods(self):
        """
        Add :class:`rc.BaseState` methods to object if PicklingError or RecursionError if class in mro is
        in :data `rc.STATE_ATTRS`:
            - :meth:`rc.BaseState.__getstate__`: (pickle) when state=None.
            - :meth:`rc.BaseState.__setstate__`: (unpickle) when state.

        Raises:
            PicklingError: Object has one or both state methods.
            AttributeError: Read-only.
            NotImplementedError: No mro items in STATE_ATTRS.

        Returns:
            Object with __getstate__ and __setstate__ methods added.
        """

        def _state(*args):
            if args and len(args) == 1:
                return type(self)(args[0]).state()
            elif args and len(args) == 2:
                type(self)(args[0]).state(args[1])

        setstate = None
        if (getstate := self.has(N.GETSTATE.value)) or (setstate := self.has(N.SETSTATE.value)):
            raise PicklingError(f'Object {self.data}, has one or both state methods: {getstate=}, {setstate=}')
        found = False
        for C in self.mro:
            if C in STATE_ATTRS:
                for attr in (N.GETSTATE.value, N.SETSTATE.value,):
                    es = type(self)(self.data.__class__)
                    if not es.writeable(attr):
                        exc = es.readonly(attr)
                        raise AttributeError(f'Read-only: {self.data=}') from exc
                self.data.__class__.__getstate__ = _state
                self.data.__class__.__setstate__ = _state
                found = True
        if found:
            return self.data
        else:
            raise NotImplementedError(f'No mro: {self.mro} items in {STATE_ATTRS.keys()=} for {self.data=}')

    @property
    def unpickles(self):
        """Unpickles object (loads)."""
        return pickle_loads(self.data)
    # </editor-fold>

    __class_getitem__ = classmethod(GenericAlias)


PathLikeStr = Union[PathLike, Path, str]


class StateEnv(Environs, BaseState):
    """
    Env Class with Deepcopy and Pickle.

    Examples:
        >>> from copy import deepcopy
        >>> from rc import StateEnv
        >>> import environs
        >>>
        >>> env = StateEnv()
        >>> term = env('TERM')
        >>> env
        <StateEnv {'TERM': 'xterm-256color'}>
        >>> state = env.__getstate__()
        >>> env_deepcopy = deepcopy(env)
        >>> env_deepcopy
        <StateEnv {'TERM': 'xterm-256color'}>
        >>> assert id(env_deepcopy) != id(env)
        >>> assert id(env_deepcopy._values) != id(env._values)
    """

    def __init__(self, *, eager=True, expand_vars=False):
        super().__init__(eager=eager, expand_vars=expand_vars)


# </editor-fold>
# <editor-fold desc="Functions">
def aioloop(): return noexception(RuntimeError, get_running_loop)


def allin(origin, destination):
    """
    Checks all items in origin are in destination iterable.

    Examples:
        >>> class Int(int):
        ...     pass
        >>> allin(tuple.__mro__, BUILTINS_CLASSES)
        True
        >>> allin(Int.__mro__, BUILTINS_CLASSES)
        False
        >>> allin('tuple int', 'bool dict int')
        False
        >>> allin('bool int', ['bool', 'dict', 'int'])
        True
        >>> allin(['bool', 'int'], ['bool', 'dict', 'int'])
        True

    Args:
        origin: origin iterable.
        destination: destination iterable to check if origin items are in.

    Returns:
        True if all items in origin are in destination.
    """
    origin = to_iter(origin)
    destination = to_iter(destination)
    return all(map(lambda x: x in destination, origin))


def annotations(obj, stack=1, sequence=False):
    """
    Formats obj annotations.

    Examples:
        >>> from dataclasses import dataclass
        >>>
        >>> pretty_install()
        >>>
        >>> @dataclass
        ... class Test:
        ...     any: Any = 'any'
        ...     classvar: ClassVar[str] = 'classvar'
        ...     classvar_optional: ClassVar[Optional[str]] = 'classvar_optional'
        ...     classvar_optional_union: ClassVar[Optional[Union[str, int]]] = 'classvar_optional_union'
        ...     classvar_union: ClassVar[Union[str, int]] = 'classvar_union'
        ...     final: Final= 'final'
        ...     final_str: Final[str] = 'final_str'
        ...     integer: int = 1
        ...     initvar: InitVar[str] = 'initvar'
        ...     initvar_optional: InitVar[Optional[str]] = 'initvar_optional'
        ...     literal: Literal['literal', 'literal2'] = 'literal2'
        ...     literal_optional: Optional[Literal['literal_optional', 'literal_optional2']] = 'literal_optional2'
        ...     optional: Optional[str] = 'optional'
        ...     union: Union[str, int] = 1
        ...     optional_union: Optional[Union[str, bool]] = True
        ...     def __post_init__(self, initvar: int, initvar_optional: Optional[int]):
        ...         self.a = initvar
        >>>
        >>> ann = annotations(Test)
        >>> ann['any'].any, ann['any'].cls, ann['any'].default
        (True, typing.Any, None)
        >>> ann['classvar'].classvar, ann['classvar'].cls, ann['classvar'].default
        (True, <class 'str'>, '')
        >>> ann['classvar_optional'].classvar, ann['classvar_optional'].cls, ann['classvar_optional'].default
        (True, <class 'str'>, '')
        >>> ann['classvar_optional_union'].classvar, ann['classvar_optional_union'].cls, \
        ann['classvar_optional_union'].default
        (True, <class 'str'>, '')
        >>> ann['classvar_union'].classvar, ann['classvar_union'].cls, ann['classvar_union'].default
        (True, <class 'str'>, '')
        >>> ann['final'].final, ann['final'].cls, ann['final'].default  # TODO: 'final'
        (True, typing.Final, None)
        >>> ann['final_str'].final, ann['final_str'].cls, ann['final_str'].default  # TODO: 'final_str'
        (True, <class 'str'>, '')
        >>> ann['integer'].cls, ann['integer'].default
        (<class 'int'>, 0)
        >>> ann['initvar'].initvar, ann['initvar'].cls, ann['initvar'].default
        (True, <class 'str'>, '')
        >>> ann['initvar_optional'].initvar, ann['initvar_optional'].cls, ann['initvar_optional'].default
        (True, <class 'str'>, '')
        >>> ann['literal'].literal, ann['literal'].cls, ann['literal'].default
        (True, <class 'str'>, 'literal')
        >>> ann['literal_optional'].literal, ann['literal_optional'].cls, ann['literal_optional'].default
        (True, <class 'str'>, 'literal_optional')
        >>> ann['optional'].optional, ann['optional'].cls, ann['optional'].default
        (True, <class 'str'>, '')
        >>> ann['union'].union, ann['union'].cls, ann['union'].default
        (True, <class 'str'>, '')
        >>> ann['optional_union'].optional, ann['optional_union'].union, ann['optional_union'].cls, \
        ann['optional_union'].default
        (True, True, <class 'str'>, '')

    Args:
        obj: object.
        stack: stack index to extract globals and locals (default: 1) or frame.
        sequence: return sequence instead of dict (default: False).

    Returns:
        Annotation: obj annotations. Default are filled with annotation not with class default.
    """

    def value(_cls):
        # TODO: 1) default from annotations, 2) value from kwargs or class defaults.
        return noexception(_cls)

    def inner(_hint):
        cls = _hint
        default = None
        args = list(get_args(_hint))
        _annotations = list()
        origin = get_origin(_hint)
        literal = origin == Literal
        final = origin == Final or _hint == Final
        _any = _hint == Any
        union = origin == Union
        classvar = origin == ClassVar
        # TODO: Look because origin must be InitVar and then  ...
        initvar = Es(cls).initvar
        optional = type(None) in args
        if initvar:
            if isinstance(_hint.type, type):
                cls = _hint.type
                default = value(cls)
            else:
                _hint = _hint.type
                _a = inner(_hint)
                _annotations.append(_a)
                default = _a.default
                cls = _a.cls
        elif origin is None:
            cls = _hint
            # TODO: final (now: None) -> default: 'final'  # hint == Final and origin is None
            default = None if _any or final else value(cls)
        elif literal and args:
            default = args[0]  # TODO: o default or kwarg or None if Optional(?)
            cls = type(default)
        elif final and args:  # origin == Final
            cls = args[0]
            # TODO: final (now: '') -> default: 'final_str'
            default = cls()
        elif args:
            literal = Literal._name in repr(_hint)
            for arg in args:
                if Es(arg).none:
                    _annotations.append(None)
                else:
                    _a = inner(arg)
                    _annotations.append(_a)
            data = _annotations[1] if _annotations[0] is None else _annotations[0]
            default = data.default
            cls = data.cls
        return Annotation(any=_any, args=_annotations or args, classvar=classvar, cls=cls, default=default,
                          final=final, hint=_hint, initvar=initvar, literal=literal, name=name,
                          optional=optional, origin=origin, union=union)

    frame = stack if Es(stack).frametype else insstack()[stack].frame
    rv = dict()
    if a := noexception(get_type_hints, obj, globalns=frame.f_globals, localns=frame.f_locals):
        for name, hint in a.items():
            rv |= {name: inner(hint)}
    rv = dict_sort(rv)
    return list(rv.values()) if sequence else rv


def annotations_init(cls, stack=2, optional=True, **kwargs):
    """
    Init with defaults or kwargs an annotated class.

    Examples:
        >>> NoInitValue = NamedTuple('NoInitValue', var=str)

        >>> A = NamedTuple('A', module=str, path=Optional[PathLib], test=Optional[NoInitValue])
        >>> annotations_init(A, optional=False)
        A(module='', path=None, test=None)
        >>> annotations_init(A)
        A(module='', path=PosixPath('.'), test=None)
        >>> annotations_init(A, test=NoInitValue('test'))
        A(module='', path=PosixPath('.'), test=NoInitValue(var='test'))
        >>> annotations_init(A, optional=False, test=NoInitValue('test'))
        A(module='', path=None, test=NoInitValue(var='test'))

    Args:
        cls: NamedTuple cls.
        stack: stack index to extract globals and locals (default: 2) or frame.
        optional: True to use args[0] instead of None as default for Optional fallback to None if exception.
        **kwargs:

    Returns:
        cls: cls instance with default values.
    """
    values = dict()
    for name, a in annotations(cls, stack=stack).items():
        if v := kwargs.get(name):
            value = v
        elif a.origin == Union and not optional:
            value = None
        else:
            value = a.default
        values[name] = value
    # for name, a in annotations(cls).items():
    #     value = None
    #     if v := kwargs.get(name):
    #         value = v
    #     elif a.origin == Literal:
    #         value = a.args[0]
    #     elif a.origin == Union and not optional:
    #         pass
    #     else:
    #         with suppress(Exception):
    #             value = (a.cls if a.origin is None else a.args[1] if a.args[0] is None else a.args[0])()
    #     rv[name] = value
    with suppress(Exception):
        return cls(**values)


def anyin(origin, destination):
    """
    Checks any item in origin are in destination iterable and return the first found.

    Examples:
        >>> class Int(int):
        ...     pass
        >>> anyin(tuple.__mro__, BUILTINS_CLASSES)
        <class 'tuple'>
        >>> assert anyin('tuple int', BUILTINS_CLASSES) is None
        >>> anyin('tuple int', 'bool dict int')
        'int'
        >>> anyin('tuple int', ['bool', 'dict', 'int'])
        'int'
        >>> anyin(['tuple', 'int'], ['bool', 'dict', 'int'])
        'int'

    Args:
        origin: origin iterable.
        destination: destination iterable to check if any of origin items are in.

    Returns:
        First found if any item in origin are in destination.
    """
    origin = to_iter(origin)
    destination = to_iter(destination)
    return first_true(origin, pred=lambda x: x in destination)


def cache(func):
    """
    Caches previous calls to the function if object is pickable.

    Examples:
        >>>
    """
    cache.enabled = True
    cache.memo = dict()
    if func not in cache.memo:
        cache.memo[func] = dict()

    def cache_info():
        """
        Cache Wrapper Info.

        Returns:
            Cache Wrapper Info.
        """
        return CacheWrapperInfo(wrapper.cache_hit, wrapper.cache_passed, wrapper.cache_total)

    def cache_clear():
        """Clear Cache."""
        wrapper.cache_hit = 0
        wrapper.cache_passed = 0
        wrapper.cache_total = 0
        del cache.memo[func]
        cache.memo[func] = dict()

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Cache Wrapper."""
        wrapper.cache_total += 1
        if cache.enabled:
            try:
                key = Es((args, kwargs,)).pickles
                if key not in cache.memo[func]:
                    wrapper.cache_hit += 1
                    cache.memo[func][key] = func(*args, **kwargs)
                else:
                    wrapper.cache_passed += 1
                value = cache.memo[func][key]
            except Exception as exception:
                red(f'{func=}({args}, {kwargs}) not cached: {exception}')
                wrapper.cache_hit += 1
                value = func(*args, **kwargs)
        else:
            wrapper.cache_hit += 1
            value = func(*args, **kwargs)
        return value

    wrapper.cache_hit = 0
    wrapper.cache_passed = 0
    wrapper.cache_total = 0
    wrapper.cache_clear = cache_clear
    wrapper.cache_info = cache_info

    return wrapper


def cmd(command, exc=False, lines=True, shell=True, py=False, pysite=True):
    """
    Runs a cmd.

    Examples:
        >>> cmd('ls a')
        CompletedProcess(args='ls a', returncode=1, stdout=[], stderr=['ls: a: No such file or directory'])
        >>> assert 'Requirement already satisfied' in cmd('pip install pip', py=True).stdout[0]
        >>> cmd('ls a', shell=False, lines=False)  # Extra '\' added to avoid docstring error.
        CompletedProcess(args=['ls', 'a'], returncode=1, stdout='', stderr='ls: a: No such file or directory\\n')
        >>> cmd('echo a', lines=False)  # Extra '\' added to avoid docstring error.
        CompletedProcess(args='echo a', returncode=0, stdout='a\\n', stderr='')

    Args:
        command: command.
        exc: raise exception.
        lines: split lines so ``\\n`` is removed from all lines (extra '\' added to avoid docstring error).
        py: runs with python executable.
        shell: expands shell variables and one line (shell True expands variables in shell).
        pysite: run on site python if running on a VENV.

    Returns:
        Union[CompletedProcess, int, list, str]: Completed process output.

    Raises:
        CmdError:
   """
    if py:
        m = '-m'
        if isinstance(command, str) and command.startswith('/'):
            m = str()
        command = f'{str(PYTHON_SITE) if pysite else str(PYTHON_SYS)} {m} {command}'
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


def cmdname(func, sep='_'): return func.__name__.split(**split_sep(sep))[0]


def current_task_name(): return current_task().get_name() if aioloop() else str()


@singledispatch
def delete(data: MutableMapping, key=('self', 'cls',)):
    """
    Deletes item in dict based on key.

    Args:
        data: MutableMapping.
        key: key.

    Returns:
        data: dict with key deleted or the same if key not found.
    """
    key = to_iter(key)
    for item in key:
        with suppress(KeyError):
            del data[item]
    return data


@delete.register
def delete_list(data: list, key=('self', 'cls',)):
    """
    Deletes value in list.

    Args:
        data: MutableMapping.
        key: key.

    Returns:
        data: list with value deleted or the same if key not found.
    """
    key = to_iter(key)
    for item in key:
        with suppress(ValueError):
            data.remove(item)
    return data


def dict_sort(data, ordered=False, reverse=False):
    """
    Order a dict based on keys.

    Args:
        data: dict to be ordered.
        ordered: OrderedDict.
        reverse: reverse.

    Returns:
        Union[dict, collections.OrderedDict]: Dict sorted
    """
    rv = {key: data[key] for key in sorted(data.keys(), reverse=reverse)}
    if ordered:
        return OrderedDict(rv)
    return rv.copy()


def effect(apply, data):
    """
    Perform function on iterable.

    Examples:
        >>> pretty_install()
        >>>
        >>> simple = Simple()
        >>> effect(lambda x: simple.__setattr__(x, dict()), 'a b')
        >>> simple.a, simple.b
        ({}, {})

    Args:
        apply: Function to apply.
        data: Iterable to perform function.

    Returns:
        No Return.
    """
    consume(side_effect(apply, to_iter(data)))


def enumvalue(data):
    """
    Returns Enum Value if Enum Instance or Data.

    Examples:
        >>> pretty_install()
        >>>
        >>> enumvalue(N.ANNOTATIONS)
        '__annotations__'
        >>> enumvalue(None)

    Args:
        data: object.

    Returns:
        Enum Value or Data.
    """
    return data.value if Es(data).enum else data


def get(data, *args, default=None, one=True, recursive=False, with_keys=False):
    """
    Get value of name in Mutabble Mapping/GetType or object

    Examples:
        >>> get(dict(a=1), 'a')
        1
        >>> get(dict(a=1), 'b')
        >>> get(dict(a=1), 'b', default=2)
        2
        >>> get(dict, '__module__')
        'builtins'
        >>> get(dict, '__file__')

    Args:
        data: MutabbleMapping/GetType to get value.
        *args: keys (default: ('name')).
        default: default value (default: None).
        with_keys: return dict names and values or values (default: False).
        one: return [0] if len == 1 and one instead of list (default: True).
        recursive: recursivily for MutableMapping (default: False).

    Returns:
        Value for key.
    """

    def rv(items):
        if recursive and with_keys:
            items = {key: value[0] if len(value) == 1 and one else value for key, value in items}
        if with_keys:
            return items
        return list(r)[0] if len(r := items.values()) == 1 and one else list(r)

    args = args or ('name',)
    es = Es(data)
    if recursive and not es.mm:
        # TODO: to_vars() empty
        # data = to_vars()
        pass
    if es.mm and recursive:
        return rv(defaultdict(list, {k: v for arg in args for k, v in _nested_lookup(arg, data,
                                                                                     with_keys=with_keys)}))
    elif es.mm:
        return rv({arg: data.get(arg, default) for arg in args})

    return rv({attr: getattr(data, attr, default) for attr in args})


@cache
def getnostr(data, attr='value'):
    """
    Get attr if data is not str.

    Examples:
        >>> from rc import getnostr
        >>> simple = Simple(value=1)
        >>> assert getnostr(simple) == 1
        >>> assert getnostr(simple, None) == simple
        >>> assert getnostr('test') == 'test'

    Args:
        data: object.
        attr: attribute name (default: 'value').

    Returns:
        Attr value if not str.
    """
    return (data if isinstance(data, str) else getattr(data, attr)) if attr else data


def getset(data, name, default=None, setvalue=True):
    """
    Sets attribute with default if it does not exists and returns value.

    Examples:
        >>> class Dict: pass
        >>> class Slots: __slots__ = ('a', )
        >>>
        >>> d = Dict()
        >>> s = Slots()
        >>> getset(d, 'a')
        >>> # noinspection PyUnresolvedReferences
        >>> d.a
        >>> getset(s, 'a')
        >>> s.a
        >>>
        >>> d = Dict()
        >>> s = Slots()
        >>> getset(d, 'a', 2)
        2
        >>> # noinspection PyUnresolvedReferences
        >>> d.a
        2
        >>> getset(s, 'a', 2)
        2
        >>> s.a
        2
        >>>
        >>> class Dict: a = 1
        >>> class Slots:
        ...     __slots__ = ('a', )
        ...     def __init__(self):
        ...         self.a = 1
        >>> d = Dict()
        >>> s = Slots()
        >>> getset(d, 'a')
        1
        >>> getset(s, 'a')
        1
        >>> getset(d, 'a', 2)
        1
        >>> getset(s, 'a', 2)
        1

    Args:
        data: object.
        name: attr name.
        default: default value (default: None)
        setvalue: setattr in object if AttributeError (default: True).

    Returns:
        Attribute value or sets default value and returns.
    """
    try:
        return object.__getattribute__(data, name)
    except AttributeError:
        if setvalue:
            object.__setattr__(data, name, default)
            return object.__getattribute__(data, name)
        return default


def iseven(number): return Es(number).even


def in_dict(data, items=None, **kwargs):
    """
    Is Item in Dict?.

    Examples:
        >>> in_dict(globals(), {'True': True, 'False': False})
        True
        >>> in_dict(globals()['__builtins__'], {'True': True, 'False': False}, __name__='builtins')
        True
        >>> in_dict(globals(), {'True': True, 'False': False}, __name__='builtins')
        True
        >>> Es(BUILTINS).builtin
        True
        >>> Es(dict(__name__='builtins')).builtin
        True

    Args:
        data: Dict
        items: Dict with key and values for not str keys (default: None)
        **kwargs: keys and values.

    Returns:
        True if items in Dict.
    """
    if Es(data).mm:
        for key, value in ((items if items else dict()) | kwargs).items():
            values = nested_lookup(key, data)
            if not values or value not in values:
                return False
        return True
    return False


def join_newline(data): return NEWLINE.join(data)


def map_reduce_even(iterable): return map_reduce(iterable, keyfunc=iseven)


def map_with_args(data, func, /, *args, pred=lambda x: True if x else False, split=' ', **kwargs):
    """
    Apply pred/filter to data and map with args and kwargs.

    Examples:
        >>> pretty_install()
        >>> # noinspection PyUnresolvedReferences
        >>> def f(i, *ar, **kw):
        ...     return f'{i}: {[a(i) for a in ar]}, {", ".join([f"{k}: {v(i)}" for k, v in kw.items()])}'
        >>> map_with_args('0.1.2', f, int, list, pred=lambda x: x != '0', split='.', int=int, str=str)
        ["1: [1, ['1']], int: 1, str: 1", "2: [2, ['2']], int: 2, str: 2"]

    Args:
        data: data.
        func: final function to map.
        *args: args to final map function.
        pred: pred to filter data before map.
        split: split for data str.
        **kwargs: kwargs to final map function.

    Returns:
        List with results.
    """
    return [func(item, *args, **kwargs) for item in yield_if(data, pred=pred, split=split)]


def newprop(name=None, default=None):
    """
    Get a new property with getter, setter and deleter.

    Examples:
        >>> class Test:
        ...     prop = newprop()
        ...     callable = newprop(default=str)
        >>>
        >>> test = Test()
        >>> '_prop' not in vars(test)
        True
        >>> test.prop
        >>> '_prop' in vars(test)
        True
        >>> test.prop
        >>> test.prop = 2
        >>> test.prop
        2
        >>> del test.prop
        >>> '_prop' in vars(test)
        False
        >>> test.prop
        >>> '_callable' not in vars(test)
        True
        >>> test.callable  # doctest: +ELLIPSIS
        '....Test object at 0x...>'
        >>> '_callable' in vars(test)
        True
        >>> test.callable  # doctest: +ELLIPSIS
        '....Test object at 0x...>'
        >>> test.callable = 2
        >>> test.callable
        2
        >>> del test.callable
        >>> '_callable' in vars(test)
        False
        >>> test.callable  # doctest: +ELLIPSIS
        '....Test object at 0x...>'

    Args:
        name: property name (attribute name: _name). :func:' varname`is used if no name (default: varname())
        default: default for getter if attribute is not defined.
            Could be a callable/partial that will be called with self (default: None)

    Returns:
        Property.
    """
    name = f'_{name if name else varname()}'
    return property(
        lambda self:
        getset(self, name, default=default(self) if Es(default).instance(Callable, partial) else default),
        lambda self, value: self.__setattr__(name, value),
        lambda self: self.__delattr__(name)
    )


def noexception(func, *args, default_=None, exc_=Exception, **kwargs):
    """
    Execute function suppressing exceptions.

    Examples:
        >>> noexception(dict(a=1).pop, 'b', default_=2, exc_=KeyError)
        2

    Args:
        func: callable.
        *args: args.
        default_: default value if exception is raised.
        exc_: exception or exceptions.
        **kwargs: kwargs.

    Returns:
        Any: Function return.
    """
    with suppress(exc_):
        return func(*args, **kwargs)
    return default_


def prefixed(name: str) -> str:
    try:
        return f'{name.upper()}_'
    except AttributeError:
        pass


def repr_format(obj, attrs, clear=True, newline=False):
    cls = obj.__class__
    if clear:
        for item in dir(cls):
            if (attr := getattr(cls, item, None)) and (c := getattr(attr, 'cache_clear', None)):
                # noinspection PyUnboundLocalVariable
                c()
    new = NEWLINE if newline else str()
    msg = f',{new if newline else " "}'.join([f"{arg}: {repr(getattr(obj, arg))}" for arg in to_iter(attrs)])
    return f'{cls.__name__}({new}{msg}{new})'


@decorator
def runwarning(func, *args, **kwargs):
    with catch_warnings(record=False):
        filterwarnings('ignore', category=RuntimeWarning)
        warnings.showwarning = lambda *_args, **_kwargs: None
        rv = func(*args, **kwargs)
        return rv


def split_sep(sep='_'): return dict(sep=sep) if sep else dict()


def startswith(name: str, builtins=True): return name.startswith('__') if builtins else name.startswith('_')


def to_camel(text, replace=True):
    """
    Convert to Camel

    Examples:
        >>> to_camel(N.IGNORE_ATTR.name)
        'IgnoreAttr'
        >>> to_camel(N.IGNORE_ATTR.name, replace=False)
        'Ignore_Attr'
        >>> to_camel(N.IGNORE_ATTR.value, replace=False)
        '__Ignore_Attr__'

    Args:
        text: text to convert.
        replace: remove '_'  (default: True)

    Returns:
        Camel text.
    """
    rv = ''.join(map(str.title, to_iter(text, '_')))
    return rv.replace('_', '') if replace else rv


def to_iter(data, always=False, split=' '):
    """
    To iter.

    Examples:
        >>> pretty_install()
        >>> to_iter('test1')
        ['test1']
        >>> to_iter('test1 test2')
        ['test1', 'test2']
        >>> to_iter(dict(a=1))
        {'a': 1}
        >>> to_iter(dict(a=1), always=True)
        [{'a': 1}]
        >>> to_iter('test1.test2')
        ['test1.test2']
        >>> to_iter('test1.test2', split='.')
        ['test1', 'test2']

    Args:
        data: data.
        always: return any iterable into a list.
        split: split for str.

    Returns:
        Iterable.
    """
    es = Es(data)
    if es.str:
        data = data.split(split)
    elif not es.iterable or always:
        data = [data]
    return data


def to_vars():
    """
    Object to dict with no copy or deepcopy.

    To be used for:
        - get recursivily
        - repr

    Returns:
        Dict.
    """
    pass


def token_open(file):
    """
    Read file with tokenize to use in nested classes ast node.

    Args:
        file: filename

    Returns:
        SourceUtil
    """
    with tokenize.open(str(file)) as f:
        return f.read()


def varname(index=2, lower=True, sep='_'):
    """
    Caller var name.

    Examples:
        >>> from rc import varname
        >>>
        >>> def function() -> str:
        ...     return varname()
        >>>
        >>> class ClassTest:
        ...     def __init__(self):
        ...         self.name = varname()
        ...
        ...     @property
        ...     def prop(self):
        ...         return varname()
        ...
        ...     # noinspection PyMethodMayBeStatic
        ...     def method(self):
        ...         return varname()
        >>>
        >>> @dataclass
        ... class DataClassTest:
        ...     def __post_init__(self):
        ...         self.name = varname()
        >>>
        >>> name = varname(1)
        >>> Function = function()
        >>> classtest = ClassTest()
        >>> method = classtest.method()
        >>> prop = classtest.prop
        >>> dataclasstest = DataClassTest()
        >>>
        >>> def test_var():
        ...     assert name == 'name'
        >>>
        >>> def test_function():
        ...     assert Function == function.__name__.lower()
        >>>
        >>> def test_class():
        ...     assert classtest.name == ClassTest.__name__.lower()
        >>>
        >>> def test_method():
        ...     assert classtest.method() == ClassTest.__name__.lower()
        ...     assert method == 'method'
        >>> def test_property():
        ...     assert classtest.prop == ClassTest.__name__.lower()
        ...     assert prop == 'prop'
        >>> def test_dataclass():
        ...     assert dataclasstest.name == DataClassTest.__name__.lower()

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
        Optional[str]: Var name.
    """
    with suppress(IndexError, KeyError):
        _stack = insstack()
        func = _stack[index - 1].function
        index = index + 1 if func == POST_INIT_NAME else index
        if line := textwrap.dedent(_stack[index].code_context[0]):
            if var := re.sub(f'(.| ){func}.*', str(), line.split(' = ')[0].replace('assert ', str()).split(' ')[0]):
                return (var.lower() if lower else var).split(**split_sep(sep))[0]


def yield_if(data, pred=lambda x: True if x else False, split=' ', apply=None):
    """
    Yield value if condition is met and apply function if predicate.

    Examples:
        >>> pretty_install()
        >>> list(yield_if([True, None]))
        [True]
        >>> list(yield_if('test1.test2', pred=lambda x: x.endswith('2'), split='.'))
        ['test2']
        >>> list(yield_if('test1.test2', pred=lambda x: x.endswith('2'), split='.', \
        apply=lambda x: x.removeprefix('test')))
        ['2']
        >>> list(yield_if('test1.test2', pred=lambda x: x.endswith('2'), split='.', \
        apply=(lambda x: x.removeprefix('test'), lambda x: int(x))))
        [2]

    Args:
        data: data
        pred: predicate (default: if value)
        split: split char for str.
        apply: functions to apply if predicate is met.

    Returns:
        Yield values if condition is met and apply functions if provided.
    """
    for item in to_iter(data, split=split):
        if pred(item):
            if apply:
                for func in to_iter(apply):
                    item = func(item)
            yield item


def yield_last(data, split=' '):
    """
    Yield value if condition is met and apply function if predicate.

    Examples:
        >>> pretty_install()
        >>> list(yield_last([True, None]))
        [(False, True, None), (True, None, None)]
        >>> list(yield_last('first last'))
        [(False, 'first', None), (True, 'last', None)]
        >>> list(yield_last('first.last', split='.'))
        [(False, 'first', None), (True, 'last', None)]
        >>> list(yield_last(dict(first=1, last=2)))
        [(False, 'first', 1), (True, 'last', 2)]

    Args:
        data: data.
        split: split char for str.

    Returns:
        Yield value and True when is the last item on iterable
    """
    data = to_iter(data, split=split)
    mm = Es(data).mm
    total = len(data)
    count = 0
    for i in data:
        count += 1
        yield count == total, *(i, data.get(i) if mm else None,)


# </editor-fold>
# <editor-fold desc="Exceptions">
class CmdError(Exception):
    """Thrown if execution of cmd command fails with non-zero status code."""

    def __init__(self, rv):
        command = rv.args
        rc = rv.returncode
        stderr = rv.stderr
        stdout = rv.stdout
        super().__init__(f'{command=}', f'{rc=}', f'{stderr=}', f'{stdout=}')


class CmdAioError(CmdError):
    """Thrown if execution of aiocmd command fails with non-zero status code."""

    def __init__(self, rv):
        super().__init__(rv)


# </editor-fold>
# <editor-fold desc="Types">
class AnnotationsType(metaclass=ABCMeta):
    """
    Annotations Type.

    Examples:
        >>> named = namedtuple('named', 'a', defaults=('a', ))
        >>> Named = NamedTuple('Named', a=str)
        >>>
        >>> Es(named).annotationstype_sub
        False
        >>> Es(named()).annotationstype
        False
        >>>
        >>> Es(Named).annotationstype_sub
        True
        >>> Es(Named(a='a')).annotationstype
        True
    """

    @classmethod
    def __subclasshook__(cls, C):
        if cls is AnnotationsType:
            return N.ANNOTATIONS.mro_first_dict(C) is not NotImplemented
        return NotImplemented


class AsDictMethodType(metaclass=ABCMeta):
    """
    AsDict Method Support (Class, Static and Method).

    Examples:
        >>> class AsDictClass: asdict = classmethod(lambda cls, *args, **kwargs: dict())
        >>> class AsDictM: asdict = lambda self, *args, **kwargs: dict()
        >>> class AsDictP: asdict = property(lambda self: dict())
        >>> class AsDictStatic: asdict = staticmethod(lambda cls, *args, **kwargs: dict())
        >>>
        >>> c = AsDictClass()
        >>> m = AsDictM()
        >>> p = AsDictP()
        >>> s = AsDictStatic()
        >>>
        >>> Es(AsDictClass).asdictmethod_sub
        True
        >>> Es(c).asdictmethod
        True
        >>>
        >>> Es(AsDictM).asdictmethod_sub
        True
        >>> Es(m).asdictmethod
        True
        >>>
        >>> Es(AsDictP).asdictmethod_sub
        False
        >>> Es(p).asdictmethod
        False
        >>>
        >>> Es(AsDictStatic).asdictmethod_sub
        True
        >>> Es(s).asdictmethod
        True
    """

    # noinspection PyUnusedLocal
    @abstractmethod
    def asdict(self, *args, **kwargs):
        pass

    @classmethod
    def __subclasshook__(cls, C):
        if cls is AsDictMethodType:
            value = N.asdict.mro_first_dict(C)
            es = Es(value)
            return value is not NotImplemented and any(
                [es.clsmethod, es.lambdatype, es.method, es.static]) and not es.prop
        return NotImplemented


class AsDictPropertyType(metaclass=ABCMeta):
    """
    AsDict Property Type.

    Examples:
        >>> class AsDictClass: asdict = classmethod(lambda cls, *args, **kwargs: dict())
        >>> class AsDictM: asdict = lambda self, *args, **kwargs: dict()
        >>> class AsDictP: asdict = property(lambda self: dict())
        >>> class AsDictStatic: asdict = staticmethod(lambda cls, *args, **kwargs: dict())
        >>>
        >>> c = AsDictClass()
        >>> m = AsDictM()
        >>> p = AsDictP()
        >>> s = AsDictStatic()
        >>>
        >>> Es(AsDictClass).asdictproperty_sub
        False
        >>> Es(c).asdictproperty
        False
        >>>
        >>> Es(AsDictM).asdictproperty_sub
        False
        >>> Es(m).asdictproperty
        False
        >>>
        >>> Es(AsDictP).asdictproperty_sub
        True
        >>> Es(p).asdictproperty
        True
        >>>
        >>> Es(AsDictStatic).asdictproperty_sub
        False
        >>> Es(s).asdictproperty
        False
    """

    @property
    @abstractmethod
    def asdict(self):
        return dict()

    @classmethod
    def __subclasshook__(cls, C):
        if cls is AsDictPropertyType:
            return (value := N.asdict.mro_first_dict(C)) is not NotImplemented \
                   and Es(value).prop
        return NotImplemented


class DataType(metaclass=ABCMeta):
    """
    Data Type.

    Examples:
        >>> from dataclasses import make_dataclass
        >>>
        >>> Data = make_dataclass('C', [('a', int, datafield(default=1))])
        >>> class Dict: a = 1
        >>>
        >>> data = Data()
        >>> d = Dict()
        >>>
        >>> Es(Data).datatype_sub
        True
        >>> Es(data).datatype
        True
        >>> Es(Data).datatype
        False
        >>>
        >>> Es(Dict).datatype_sub
        False
        >>> Es(d).datatype
        False
    """
    __annotations__ = dict()
    __dataclass_fields__ = dict()

    @abstractmethod
    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def __subclasshook__(cls, C):
        if cls is DataType:
            return N.ANNOTATIONS.mro_first_dict(C) is not NotImplemented \
                   and N.DATACLASS_FIELDS.mro_first_dict(C) is not NotImplemented \
                   and N.REPR.mro_first_dict(C) is not NotImplemented
        return NotImplemented


class DictType(metaclass=ABCMeta):
    """
    Dict Type.

    Examples:
        >>> class Dict: a = 1
        >>> class Slots: a = 1; __slots__ = tuple()
        >>>
        >>> d = Dict()
        >>> s = Slots()
        >>>
        >>> Es(Dict).dicttype_sub
        True
        >>> Es(d).dicttype
        True
        >>>
        >>> Es(Slots).dicttype_sub
        False
        >>> Es(s).dicttype
        False
    """
    __dict__ = dict()

    @classmethod
    def __subclasshook__(cls, C):
        if cls is DictType:
            return N.DICT.mro_first_dict(C) is not NotImplemented
        return NotImplemented


class GetAttrNoBuiltinType(metaclass=ABCMeta):
    """
    Get Attr Type (Everything but builtins, except: object and errors)

    Examples:
        >>> class Dict: a = 1
        >>> class G: a = 1; get = lambda self, item: self.__getattribute__(item)
        >>> Named = namedtuple('Named', 'a')
        >>>
        >>> d = Dict()
        >>> dct = dict(a=1)
        >>> g = G()
        >>> n = Named('a')
        >>> t = tuple()
        >>>
        >>> Es(Dict).getattrnobuiltintype_sub
        True
        >>> Es(d).getattrnobuiltintype
        True
        >>>
        >>> Es(G).getattrnobuiltintype_sub
        False
        >>> Es(g).getattrnobuiltintype
        False
        >>>
        >>> Es(dict).getattrnobuiltintype_sub
        False
        >>> Es(dct).getattrnobuiltintype
        False
        >>>
        >>> Es(tuple).getattrnobuiltintype_sub
        False
        >>> Es(t).getattrnobuiltintype
        False
        >>>
        >>> Es(list).getattrnobuiltintype_sub
        False
        >>> Es(list()).getattrnobuiltintype
        False
        >>>
        >>> Es(Named).getattrnobuiltintype_sub
        True
        >>> Es(n).getattrnobuiltintype
        True
        """

    @abstractmethod
    def __getattribute__(self, n):
        return object.__getattribute__(self, n)

    @classmethod
    def __subclasshook__(cls, C):
        if cls is GetAttrNoBuiltinType:
            g = N.get_.mro_first_dict(C)
            return any([N._field_defaults.mro_first_dict(C) is not NotImplemented,
                        not allin(C.__mro__, BUILTINS_CLASSES) and g is NotImplemented or
                        (g is not NotImplemented and not callable(g))])
        return NotImplemented


class GetAttrType(metaclass=ABCMeta):
    """
    Get Attr Type (Everything but GetType)

    Examples:
        >>> class Dict: a = 1
        >>> class G: a = 1; get = lambda self, item: self.__getattribute__(item)
        >>> Named = namedtuple('Named', 'a')
        >>>
        >>> d = Dict()
        >>> dct = dict(a=1)
        >>> g = G()
        >>> n = Named('a')
        >>> t = tuple()
        >>>
        >>> Es(Dict).getattrtype_sub
        True
        >>> Es(d).getattrtype
        True
        >>>
        >>> Es(G).getattrtype_sub
        False
        >>> Es(g).getattrtype
        False
        >>>
        >>> Es(dict).getattrtype_sub
        False
        >>> Es(dct).getattrtype
        False
        >>>
        >>> Es(tuple).getattrtype_sub
        True
        >>> Es(t).getattrtype
        True
        >>>
        >>> Es(list).getattrtype_sub
        True
        >>> Es(list()).getattrtype
        True
        >>>
        >>> Es(Named).getattrtype_sub
        True
        >>> Es(n).getattrtype
        True
        """

    @abstractmethod
    def __getattribute__(self, n):
        return object.__getattribute__(self, n)

    @classmethod
    def __subclasshook__(cls, C):
        if cls is GetAttrType:
            g = N.get_.mro_first_dict(C)
            return any([N._field_defaults.mro_first_dict(C) is not NotImplemented,
                        g is NotImplemented or (g is not NotImplemented and not callable(g))])
        return NotImplemented


@runtime_checkable
class GetItemSupportType(Protocol):
    """Supports __getitem__."""
    __slots__ = tuple()

    @abstractmethod
    def __getitem__(self, index):
        return self[index]


@runtime_checkable
class GetSupportType(Protocol):
    """Supports get method."""
    __slots__ = tuple()

    @abstractmethod
    def get(self, name, default=None):
        return self, name, default


@runtime_checkable
class GetType(Protocol):
    """
    Get Type.

    Examples:
        >>> class Dict: a = 1
        >>> class G: a = 1; get = lambda self, item: self.__getattribute__(item)
        >>> class Slots: a = 1; __slots__ = tuple()
        >>>
        >>> d = Dict()
        >>> dct = dict(a=1)
        >>> g = G()
        >>>
        >>> dct.get('a')
        1
        >>> g.get('a')
        1
        >>>
        >>> Es(Dict).gettype_sub
        False
        >>> Es(d).gettype
        False
        >>>
        >>> Es(G).gettype_sub
        True
        >>> Es(g).gettype
        True
        >>>
        >>> Es(dict).gettype_sub
        True
        >>> Es(dct).gettype
        True
    """

    @abstractmethod
    def get(self, name, default=None):
        pass


class NamedType(metaclass=ABCMeta):
    """
    named Type.

    Examples:
        >>> named = namedtuple('named', 'a', defaults=('a', ))
        >>>
        >>> Es(named()).namedtype
        True
        >>> Es(named).namedtype_sub
        True
        >>>
        >>> Es(named()).tuple
        True
        >>> issubclass(named, tuple)
        True
    """
    _fields = tuple()
    _field_defaults = dict()

    @abstractmethod
    def _asdict(self):
        return dict()

    @classmethod
    def __subclasshook__(cls, C):
        if cls is NamedType:
            _asdict = N._asdict.mro_first_dict(C)
            rv = N._field_defaults.mro_first_dict(C) is not NotImplemented and (
                    _asdict is not NotImplemented and callable(_asdict)) and N._fields.mro_first_dict(
                C) is not NotImplemented
            return rv
        return NotImplemented


class NamedAnnotationsType(metaclass=ABCMeta):
    """
    named Type.

    Examples:
        >>> named = namedtuple('named', 'a', defaults=('a', ))
        >>> Named = NamedTuple('Named', a=str)
        >>>
        >>> Es(named()).named_annotationstype
        False
        >>> Es(named).named_annotationstype_sub
        False
        >>>
        >>> Es(Named('a')).named_annotationstype
        True
        >>> Es(Named).named_annotationstype_sub
        True
        >>>
        >>> Es(named()).tuple
        True
        >>> issubclass(named, tuple)
        True
    """
    __annotations__ = dict()
    _fields = tuple()
    _field_defaults = dict()

    @abstractmethod
    def _asdict(self):
        return dict()

    @classmethod
    def __subclasshook__(cls, C):
        if cls is NamedAnnotationsType:
            _asdict = N._asdict.mro_first_dict(C)
            _a = _asdict is not NotImplemented and callable(_asdict)
            value = N.ANNOTATIONS.mro_first_dict(C)
            return value is not NotImplemented and N._field_defaults.mro_first_dict(
                C) is not NotImplemented and _a and N._fields.mro_first_dict(C) is not NotImplemented
        return NotImplemented


class SlotsType(metaclass=ABCMeta):
    """
    Slots Type.

    Examples:
        >>> class Dict: a = 1
        >>> class Slots: a = 1; __slots__ = tuple()
        >>>
        >>> d = Dict()
        >>> s = Slots()
        >>>
        >>> Es(Dict).slotstype_sub
        False
        >>> Es(d).slotstype
        False
        >>>
        >>> Es(Slots).slotstype_sub
        True
        >>> Es(s).slotstype
        True
    """

    @classmethod
    def __subclasshook__(cls, C):
        if cls is SlotsType:
            return N.SLOTS.mro_first_dict_no_object(C) is not NotImplemented
        return NotImplemented


# </editor-fold>
# <editor-fold desc="Echo">
def black(msg, bold=False, nl=True, underline=False,
          blink=False, err=False, reset=True, rc=None) -> None:
    """
    Black.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='black', err=err, reset=reset)
    if rc is not None:
        Exit(rc)


def blue(msg, bold=False, nl=True, underline=False,
         blink=False, err=False, reset=True, rc=None) -> None:
    """
    Blue.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='blue', err=err, reset=reset)
    if rc is not None:
        Exit(rc)


def cyan(msg, bold=False, nl=True, underline=False,
         blink=False, err=False, reset=True, rc=None) -> None:
    """
    Cyan.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='cyan', err=err, reset=reset)
    if rc is not None:
        Exit(rc)


def green(msg, bold=False, nl=True, underline=False,
          blink=False, err=False, reset=True, rc=None) -> None:
    """
    Green.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='green', err=err, reset=reset)
    if rc is not None:
        Exit(rc)


def magenta(msg, bold=False, nl=True, underline=False,
            blink=False, err=False, reset=True, rc=None) -> None:
    """
    Magenta.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='magenta', err=err, reset=reset)
    if rc is not None:
        Exit(rc)


def red(msg, bold=False, nl=True, underline=False,
        blink=False, err=True, reset=True, rc=None) -> None:
    """
    Red.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='red', err=err, reset=reset)
    if rc is not None:
        Exit(rc)


def white(msg, bold=False, nl=True, underline=False,
          blink=False, err=False, reset=True, rc=None) -> None:
    """
    White.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='white', err=err, reset=reset)
    if rc is not None:
        Exit(rc)


def yellow(msg, bold=False, nl=True, underline=False,
           blink=False, err=True, reset=True, rc=None) -> None:
    """
    Yellow.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='yellow', err=err, reset=reset)
    if rc is not None:
        Exit(rc)


def bblack(msg, bold=False, nl=True, underline=False,
           blink=False, err=False, reset=True, rc=None) -> None:
    """
    Black.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='bright_black', err=err,
          reset=reset)
    if rc is not None:
        Exit(rc)


def bblue(msg, bold=False, nl=True, underline=False,
          blink=False, err=False, reset=True, rc=None) -> None:
    """
    Bblue.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='bright_blue', err=err,
          reset=reset)
    if rc is not None:
        Exit(rc)


def bcyan(msg, bold=False, nl=True, underline=False,
          blink=False, err=False, reset=True, rc=None) -> None:
    """
    Bcyan.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='bright_cyan', err=err,
          reset=reset)
    if rc is not None:
        Exit(rc)


def bgreen(msg, bold=False, nl=True, underline=False,
           blink=False, err=False, reset=True, rc=None) -> None:
    """
    Bgreen.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='bright_green', err=err,
          reset=reset)
    if rc is not None:
        Exit(rc)


def bmagenta(msg, bold=False, nl=True, underline=False,
             blink=False, err=False, reset=True, rc=None) -> None:
    """
    Bmagenta.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='bright_magenta', err=err,
          reset=reset)
    if rc is not None:
        Exit(rc)


def bred(msg, bold=False, nl=True, underline=False,
         blink=False, err=False, reset=True, rc=None) -> None:
    """
    Bred.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='bright_red', err=err,
          reset=reset)
    if rc is not None:
        Exit(rc)


def bwhite(msg, bold=False, nl=True, underline=False,
           blink=False, err=False, reset=True, rc=None) -> None:
    """
    Bwhite.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='bright_white', err=err,
          reset=reset)
    if rc is not None:
        Exit(rc)


def byellow(msg, bold=False, nl=True, underline=False,
            blink=False, err=False, reset=True, rc=None) -> None:
    """
    Byellow.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='bright_yellow', err=err,
          reset=reset)
    if rc is not None:
        Exit(rc)


# </editor-fold>
# <editor-fold desc="Test">
class TestAsync:
    _async_classmethod = varname(1, sep=str())
    _classmethod = varname(1, sep=str())
    _async_method = varname(1, sep=str())
    _method = varname(1, sep=str())
    _cprop = varname(1, sep=str())
    _async_pprop = varname(1, sep=str())
    _pprop = varname(1, sep=str())
    _async_prop = varname(1, sep=str())
    _prop = varname(1, sep=str())
    _async_staticmethod = varname(1, sep=str())
    _staticmethod = varname(1, sep=str())

    @classmethod
    async def async_classmethod(cls): return cls._async_classmethod

    @classmethod
    def classmethod(cls): return cls._classmethod

    async def async_method(self): return self._async_method

    def method(self): return self._method

    @cached_property
    def cprop(self): return self._cprop

    @property
    async def async_prop(self): return self._async_prop

    @property
    def prop(self): return self._prop

    @staticmethod
    async def async_staticmethod(): return TestAsync._async_staticmethod

    @staticmethod
    def staticmethod(): return TestAsync._staticmethod


@dataclass
class TestData:
    __data = varname(1)
    __dataclass_classvar__: ClassVar[str] = '__dataclass_classvar__'
    __dataclass_classvar: ClassVar[str] = '__dataclass_classvar'
    __dataclass_default_factory: Union[dict, str] = datafield(default_factory=dict, init=False)
    __dataclass_default_factory_init: Union[dict, str] = datafield(default_factory=dict)
    dataclass_classvar: ClassVar[str] = 'dataclass_classvar'
    dataclass_default_factory: Union[dict, str] = datafield(default_factory=dict, init=False)
    dataclass_default_factory_init: Union[dict, str] = datafield(default_factory=dict)
    dataclass_default: str = datafield(default='dataclass_default', init=False)
    dataclass_default_init: str = datafield(default='dataclass_default_init')
    dataclass_initvar: InitVar[str] = 'dataclass_initvar'
    dataclass_str: str = 'dataclass_integer'

    def __post_init__(self, dataclass_initvar): pass

    __class_getitem__ = classmethod(GenericAlias)


class TestDataDictMix(TestData):
    subclass_annotated_str: str = 'subclass_annotated_str'
    subclass_classvar: ClassVar[str] = 'subclass_classvar'
    subclass_str = 'subclass_str'

    def __init__(self, dataclass_initvar='dataclass_initvar_1', subclass_dynamic='subclass_dynamic'):
        super().__init__()
        super().__post_init__(dataclass_initvar=dataclass_initvar)
        self.subclass_dynamic = subclass_dynamic


class TestDataDictSlotMix(TestDataDictMix):
    __slots__ = ('_slot_property', 'slot',)

    # Add init=True dataclass attrs if it subclassed and not @dataclass
    def __init__(self, dataclass_initvar='dataclass_initvar_2', slot_property='slot_property', slot='slot'):
        super().__init__()
        super().__post_init__(dataclass_initvar=dataclass_initvar)
        self._slot_property = slot_property
        self.slot = slot

    @property
    def slot_property(self):
        return self._slot_property


# </editor-fold>
# <editor-fold desc="Start">
colorama.init()
getLogger(paramiko.__name__).setLevel(NOTSET)
environ['PYTHONWARNINGS'] = 'ignore'
PathLike.register(Path)
pickle_np.register_handlers()
pretty_install(console=console, expand_all=True)
# traceback_install(console=console, extra_lines=5, show_locals=True)
disable_warnings()

# </editor-fold>
# <editor-fold desc="Instances">
user = User()
# </editor-fold>
