# -*- coding: utf-8 -*-
"""Utils Module."""
__all__ = (
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
    'datafield',
    'datafields',
    'insstack',
    'PathLib',
    'Simple',

    # Imports PyPi
    'np',
    'pd',
    'Exit',
    'dpathdelete',
    'dpathget',
    'dpathnew',
    'dpathset',
    'dpathsearch',
    'dpathvalues',
    'Environs',
    'GitRepo',
    'Console',
    'pretty_install',
    'traceback_install',

    # Constants
    'Alias',
    'BUILTINS',
    'BUILTINS_CLASSES',
    'BUILTINS_FUNCTIONS',
    'BUILTINS_OTHER',
    'CRLock',
    'console',
    'DATACLASS_FIELDS',
    'debug',
    'fmic',
    'fmicc',
    'FRAME_SYS_INIT',
    'FUNCTION_MODULE',
    'ic',
    'icc',
    'IgnoreAttr',
    'IgnoreCopy',
    'IgnoreStr',
    'LST',
    'MISSING_TYPE',
    'NEWLINE',
    'POST_INIT_NAME',
    'pp',
    'print_exception',
    'PYTHON_SYS',
    'PYTHON_SITE',
    'RunningLoop',
    'SeqNoStr',
    'SeqUnion',

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
    # 'MetaType',
    'NamedType',
    'NamedAnnotationsType',
    'SlotsType',
    # 'Types',

    # Enums
    'Attr',
    'ChainRV',
    'EnumDict',
    'EnumDictAlias',
    'Executor',
    'Kind',
    'MroGenValue',
    'Mro',
    'NameGenValue',
    'Name',

    # Bases
    'Annotation',
    'BoxKeys',
    'Chain',
    'Es',
    # 'InfoCls',
    'pproperty',
    # 'Meta',
    # 'MetaType',
    # 'RV',
    # 'Seq',
    # 'slist',
    # 'stuple',

    # Inspect
    # 'Attribute',
    # 'InfoClsBase',
    # 'InfoCls1',

    # Functions
    'aioloop',
    'allin',
    'annotations',
    'annotations_init',
    'anyin',
    'cmd',
    # 'asdict',
    # 'asdict_props',
    'cmdname',
    'current_task_name',
    'delete',
    'delete_list',
    'dict_sort',
    'effect',
    'get',
    'get_getattrtype',
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

    # Meta
    # 'BaseMeta',
    # 'Class',

    # Classes
    # 'AsDict',
    # 'Base',
    # 'Cls',
    # 'info',

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
    # 'TestBase',
    # 'TestData',
    # 'TestDataDictMix',
    # 'TestDataDictSlotMix',
)

import ast
import collections.abc
import functools
import re
import subprocess
import sys
import textwrap
import tokenize
import types
import typing
import warnings
from abc import ABCMeta
from abc import abstractmethod
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
from asyncio import current_task
from asyncio import get_running_loop
from asyncio import iscoroutine
from asyncio.events import _RunningLoop
from collections import ChainMap
from collections import defaultdict
from collections import namedtuple
from collections import OrderedDict
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
from contextlib import suppress
from dataclasses import _FIELDS
from dataclasses import _MISSING_TYPE
from dataclasses import _POST_INIT_NAME
from dataclasses import Field
from dataclasses import field as datafield
from dataclasses import fields as datafields
from dataclasses import InitVar
from enum import auto
from enum import Enum
from functools import cached_property
from functools import partial
from functools import singledispatch
from functools import singledispatchmethod
from inspect import classify_class_attrs
from inspect import FrameInfo
from inspect import getfile
from inspect import getsource
from inspect import getsourcefile
from inspect import getsourcelines
from inspect import isasyncgenfunction
from inspect import isawaitable
from inspect import iscoroutinefunction
from inspect import isgetsetdescriptor
from inspect import ismemberdescriptor
from inspect import ismethoddescriptor
from inspect import isroutine
from inspect import stack as insstack
from io import BytesIO
from io import FileIO
from io import StringIO
from operator import attrgetter
from pathlib import Path as PathLib
from subprocess import CompletedProcess
from threading import _CRLock
from types import AsyncGeneratorType
from types import BuiltinFunctionType
from types import ClassMethodDescriptorType
from types import CodeType
from types import CoroutineType
from types import DynamicClassAttribute
from types import FrameType
from types import FunctionType
from types import GeneratorType
from types import GetSetDescriptorType
from types import LambdaType
from types import MappingProxyType
from types import MemberDescriptorType
from types import MethodType
from types import MethodWrapperType
from types import ModuleType
from types import SimpleNamespace as Simple
from types import TracebackType
from types import WrapperDescriptorType
from typing import _alias
from typing import Any
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
from typing import Union
from warnings import catch_warnings
from warnings import filterwarnings

import numpy as np
import pandas as pd
from box import Box
from bson import ObjectId
from click import secho
from click.exceptions import Exit as Exit
from decorator import decorator
from devtools import Debug
from dpath.util import delete as dpathdelete
from dpath.util import get as dpathget
from dpath.util import new as dpathnew
from dpath.util import search as dpathsearch
from dpath.util import set as dpathset
from dpath.util import values as dpathvalues
from environs import Env as Environs
from git import GitConfigParser
from git import Remote
from git import Repo as GitRepo
from git.refs import SymbolicReference as GitSymbolicReference
from icecream import IceCreamDebugger
from jsonpickle.util import is_collections
from jsonpickle.util import is_installed
from jsonpickle.util import is_module_function
from jsonpickle.util import is_noncomplex
from jsonpickle.util import is_object
from jsonpickle.util import is_picklable
from jsonpickle.util import is_primitive
from jsonpickle.util import is_reducible
from jsonpickle.util import is_reducible_sequence_subclass
from jsonpickle.util import is_unicode
from more_itertools import collapse
from more_itertools import consume
from more_itertools import first_true
from more_itertools import map_reduce
from more_itertools import side_effect
from nested_lookup import nested_lookup
from rich.console import Console as Console
from rich.pretty import install as pretty_install
from rich.traceback import install as traceback_install

# <editor-fold desc="Constants">
Alias = _alias
BUILTINS = (_bltns if isinstance(_bltns := globals()['__builtins__'], dict) else vars(_bltns)).copy()
BUILTINS_CLASSES = tuple(filter(lambda x: isinstance(x, type), BUILTINS.values()))
BUILTINS_FUNCTIONS = tuple(filter(lambda x: isinstance(x, (BuiltinFunctionType, FunctionType,)), BUILTINS.values()))
BUILTINS_OTHER = tuple(map(BUILTINS.get, ('__doc__', '__import__', '__spec__', 'copyright', 'credits', 'exit',
                                          'help', 'license', 'quit',)))
CRLock = _CRLock
console = Console(color_system='256')
DATACLASS_FIELDS = _FIELDS
debug = Debug(highlight=True)
fmic = IceCreamDebugger(prefix=str()).format
fmicc = IceCreamDebugger(prefix=str(), includeContext=True).format
FRAME_SYS_INIT = sys._getframe(0)
FUNCTION_MODULE = '<module>'
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
LST = Union[typing.MutableSequence, typing.MutableSet, tuple]
MISSING_TYPE = _MISSING_TYPE
NEWLINE = '\n'
POST_INIT_NAME = _POST_INIT_NAME
pp = console.print
print_exception = console.print_exception
pretty_install(console=console, expand_all=True)
PYTHON_SYS = PathLib(sys.executable)
PYTHON_SITE = PathLib(PYTHON_SYS).resolve()
# rich.traceback.install(console=console, extra_lines=5, show_locals=True)
RunningLoop = _RunningLoop
SeqNoStr = Union[typing.Iterator, typing.KeysView, typing.MutableSequence, typing.MutableSet, tuple, typing.ValuesView]
SeqUnion = Union[typing.AnyStr, typing.ByteString, typing.Iterator, typing.KeysView, typing.MutableSequence,
                 typing.MutableSet, typing.Sequence, tuple, typing.ValuesView]


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
            return Mro.annotations.firstdict(C) is not NotImplemented
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
            value = Mro.asdict.firstdict(C)
            es = Es(value)
            return value is not NotImplemented and any(
                [es.classmethod, es.lambdatype, es.method, es.staticmethod]) and not es.prop
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
            return (value := Mro.asdict.firstdict(C)) is not NotImplemented and Es(value).prop
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
            return Mro.annotations.firstdict(C) is not NotImplemented \
                   and Mro.dataclass_fields.firstdict(C) is not NotImplemented \
                   and Mro.repr.firstdict(C) is not NotImplemented
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
            return Mro.dict.firstdict(C) is not NotImplemented
        return NotImplemented


class GetAttrNoBuiltinType(metaclass=ABCMeta):
    """
    Get Attr Type (Everything but builtins, except: object and errors)

    Examples:
        >>> class Dict: a = 1
        >>> class Get: a = 1; get = lambda self, item: self.__getattribute__(item)
        >>> Named = namedtuple('Named', 'a')
        >>>
        >>> d = Dict()
        >>> dct = dict(a=1)
        >>> g = Get()
        >>> n = Named('a')
        >>> t = tuple()
        >>>
        >>> Es(Dict).getattrnobuiltintype_sub
        True
        >>> Es(d).getattrnobuiltintype
        True
        >>>
        >>> Es(Get).getattrnobuiltintype_sub
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
            g = Mro.get.firstdict(C)
            return any([Mro._field_defaults.firstdict(C) is not NotImplemented,
                        not allin(C.__mro__, BUILTINS_CLASSES) and g is NotImplemented or
                        (g is not NotImplemented and not callable(g))])
        return NotImplemented


class GetAttrType(metaclass=ABCMeta):
    """
    Get Attr Type (Everything but GetType)

    Examples:
        >>> class Dict: a = 1
        >>> class Get: a = 1; get = lambda self, item: self.__getattribute__(item)
        >>> Named = namedtuple('Named', 'a')
        >>>
        >>> d = Dict()
        >>> dct = dict(a=1)
        >>> g = Get()
        >>> n = Named('a')
        >>> t = tuple()
        >>>
        >>> Es(Dict).getattrtype_sub
        True
        >>> Es(d).getattrtype
        True
        >>>
        >>> Es(Get).getattrtype_sub
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
            g = Mro.get.firstdict(C)
            return any([Mro._field_defaults.firstdict(C) is not NotImplemented,
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
        >>> class Get: a = 1; get = lambda self, item: self.__getattribute__(item)
        >>> class Slots: a = 1; __slots__ = tuple()
        >>>
        >>> d = Dict()
        >>> dct = dict(a=1)
        >>> g = Get()
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
        >>> Es(Get).gettype_sub
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
            _asdict = Mro._asdict.firstdict(C)
            rv = Mro._field_defaults.firstdict(C) is not NotImplemented and (_asdict is not NotImplemented and callable(
                _asdict)) and Mro._fields.firstdict(C) is not NotImplemented
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
            _asdict = Mro._asdict.firstdict(C)
            _a = _asdict is not NotImplemented and callable(_asdict)
            return Mro.annotations.firstdict(C) is not NotImplemented and Mro._field_defaults.firstdict(
                C) is not NotImplemented and _a and Mro._fields.firstdict(C) is not NotImplemented
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
            return Mro.slots.firstdict_object(C) is not NotImplemented
        return NotImplemented


# </editor-fold>
# <editor-fold desc="Enums">
class Attr(Enum):
    """
    Include Attr.

    Attributes:
    -----------
    ALL:
        include all public, private '_' and builtin '__'
    PRIVATE:
        include private '_' and public vars
    PUBLIC:
        include only public: no '_' and not '__'

    Methods:
    --------
        include(string)
            Include Attr.
    """
    ALL = auto()
    PRIVATE = '__'
    PUBLIC = '_'

    def include(self, obj) -> bool:
        """
        Include Key.

        Examples:
            >>> Attr.PUBLIC.include('_hello')
            False
            >>> Attr.PRIVATE.include('_hello')
            True
            >>> Attr.ALL.include('__hello')
            True

        Args:
            obj: string

        Returns:
            True if key to be included.
        """
        if self is Attr.ALL:
            return True
        return not obj.startswith(self.value)


class ChainRV(Enum):
    ALL = auto()
    FIRST = auto()
    UNIQUE = auto()


class EnumDict(Enum):
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

    def prefix(self, prefix):
        return f'{prefix}_{self.name}'

    @classmethod
    def values(cls):
        return list(cls.asdict().values())


EnumDictAlias = Alias(EnumDict, 1, name=EnumDict.__name__)


class Executor(Enum):
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


class Kind(Enum):
    CLASS = 'class method'
    DATA = 'data'
    METHOD = 'method'
    PROPERTY = 'property'
    STATIC = 'static method'


class MroGenValue(Enum):
    def _generate_next_value_(self, start, count, last_values):
        exclude = ('_asdict', '_field_defaults', '_fields', 'asdict', 'get',)
        return self if self in exclude else f'__{self}__'


class Mro(MroGenValue):
    """
    MRO Helper Calls.

    Private attributes and public which do not conflict with private.

    i.e: name is __name__, use :class:`Attr.name_` for name.
    """
    _asdict = auto()
    _field_defaults = auto()
    _fields = auto()
    all = auto()
    annotations = auto()
    args = auto()
    asdict = auto()
    cache_clear = auto()
    cache_info = auto()
    cached = auto()
    code = auto()
    contains = auto()
    dataclass_fields = auto()
    dataclass_params = auto()
    delattr = auto()
    dict = auto()
    dir = auto()
    doc = auto()
    eq = auto()
    file = auto()
    get = auto()
    getattribute = auto()
    getitem = auto()
    hash_exclude = auto()
    ignore_attr = auto()
    ignore_copy = auto()
    ignore_kwarg = auto()
    ignore_str = auto()
    init_subclass = auto()
    len = auto()
    loader = auto()
    module = auto()
    name = auto()
    package = auto()
    post_init = auto()
    qualname = auto()
    reduce = auto()
    repr = auto()
    repr_exclude = auto()
    repr_newline = auto()
    repr_pproperty = auto()
    setattr = auto()
    slots = auto()
    spec = auto()
    str = auto()
    subclasshook = auto()

    @classmethod
    @functools.cache
    def attrs(cls):
        """
        Get attrs tuple with private converted to real names.

        Examples:
            >>> pretty_install()
            >>>
            >>> Mro.attrs()  # doctest: +ELLIPSIS
            (
                ...,
                'ignore_copy',
                ...
            )

        Returns:
            Tuple of attributes wit private converted to real names.
        """
        return tuple(sorted(cls.members()))

    @classmethod
    def cls(cls, obj):
        """
        Object Class MRO.

        Examples:
            >>> pretty_install()
            >>>
            >>> Mro.cls(dict(a=1))
            (<class 'dict'>, <class 'object'>)

        Args:
            obj: object.

        Returns:
            Object Class MRO.
        """
        return obj.__mro__ if isinstance(obj, type) else type(obj).__mro__

    @property
    def default(self):
        # noinspection PyUnresolvedReferences
        """
        Default ignore values from Type

        Examples:
            >>> Mro.ignore_attr.default  # doctest: +ELLIPSIS
            ('asdict', ...)
            >>> Mro.ignore_attr.default == IgnoreAttr.__args__
            True
            >>> Mro.ignore_copy.default  # doctest: +ELLIPSIS
            (<class '_thread.RLock'>, ...)
            >>> Mro.ignore_copy.default == IgnoreCopy.__args__
            True
            >>> Mro.getattribute.default
            ()

        Returns:
            Tuple with default values
        """
        return noexception(Exception, getattr, globals().get(to_camel(self.name)), Mro.args.value, default_=tuple())

    def first(self, obj):
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
            >>> Mro.repr_newline.first(Test())
            True
            >>> Mro.repr_newline.first(Test2())
            False
            >>> Mro.repr_newline.first(int())
            >>> Mro.repr_pproperty.first(Test())

        Returns:
            First value of attr found in mro and instance if obj is instance.
        """
        for item in self.obj(obj):
            if self.has(item):
                return object.__getattribute__(item, self.value)

    def _firstdict(self, mro):
        for C in mro:
            if self.value in C.__dict__:
                return C.__dict__[self.value]
        return NotImplemented

    def firstdict(self, obj):
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
            >>> Mro.repr_newline.firstdict(Test())
            False
            >>> Mro.repr_newline.firstdict(Test2())
            False
            >>> Mro.repr_newline.firstdict(int())
            NotImplemented
            >>> Mro.repr_pproperty.firstdict(Test())
            NotImplemented
            >>> A = namedtuple('A', 'a')
            >>> Mro.slots.firstdict(A)
            ()

        Returns:
            First value of attr in obj.__class__.__dict__ found in mro.
        """
        return self._firstdict(self.cls(obj))

    def firstdict_object(self, obj):
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
            >>> Mro.repr_newline.firstdict_object(Test())
            False
            >>> Mro.repr_newline.firstdict_object(Test2())
            False
            >>> Mro.repr_newline.firstdict_object(int())
            NotImplemented
            >>> Mro.repr_pproperty.firstdict_object(Test())
            NotImplemented
            >>> A = namedtuple('A', 'a')
            >>> Mro.slots.firstdict_object(A)
            ()
            >>> Mro.slots.firstdict_object(dict)
            NotImplemented
            >>> Mro.slots.firstdict_object(dict())
            NotImplemented

        Returns:
            First value of attr in obj.__class__.__dict__ found in mro excluding object.
        """
        mro = list(self.cls(obj))
        mro.remove(object)
        return self._firstdict(mro)

    @property
    def getter(self):
        """
        Attr getter with real name/value.

        Examples:
            >>> import rc.utils
            >>>
            >>> Mro.module.getter(tuple)
            'builtins'
            >>> Mro.name.getter(tuple)
            'tuple'
            >>> Mro.file.getter(rc.utils)  # doctest: +ELLIPSIS
            '/Users/jose/....py'

        Returns:
            Attr getter with real name/value.
        """
        return attrgetter(self.value)

    def has(self, obj):
        """
        Checks if Object has attr.

        Examples:
            >>> pretty_install()
            >>>
            >>> class Test:
            ...     __repr_newline__ = True
            >>>
            >>> Mro.repr_newline.has(Test)
            True
            >>> Mro.repr_exclude.has(Test)
            False

        Returns:
            True if object has attribute.
        """
        return hasattr(obj, self.value)

    @classmethod
    @functools.cache
    def members(cls):
        """
        Names and real names.

        Examples:
            >>> pretty_install()
            >>>
            >>> Mro.members()  # doctest: +ELLIPSIS
            {
                ...,
                '_field_defaults': '_field_defaults',
                ...,
                'ignore_copy': '__ignore_copy__',
                ...
            }

        Returns:
            Dict with Names and real names..
        """
        return dict_sort({key: value.value for key, value in cls.__members__.items()})

    @classmethod
    def obj(cls, obj):
        """
        Object and Class MRO tuple.

        Examples:
            >>> pretty_install()
            >>>
            >>> Mro.obj(dict(a=1))
            ({'a': 1}, <class 'dict'>, <class 'object'>)

        Args:
            obj: object.

        Returns:
            Object and Class MRO tuple.
        """
        return obj.__mro__ if isinstance(obj, type) else (obj, *type(obj).__mro__)

    @classmethod
    def slot(cls, obj, name):
        """
        Is attribute in slots?

        Examples:
            >>> pretty_install()
            >>>
            >>> class First:
            ...     __slots__ = ('_hash', )
            >>>
            >>> class Test(First):
            ...     __slots__ = ('_prop', '_repr', '_slot', )
            >>>
            >>> Mro.slot(Test(), '_hash')
            True
            >>> Mro.slot(Test(), '_prop')
            True
            >>> Mro.slot(Test(), 'False')
            False

        Args:
            obj: object.
            name: attribute name.

        Returns:
            True if attribute in slots
        """
        return name in cls.slots.val(obj)

    def slotsinclude(self, obj):
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
            >>> slots = sorted(Mro.slots.val(test))
            >>> slots
            ['_hash', '_prop', '_repr', '_slot']
            >>> hash_attrs = sorted(Mro.hash_exclude.slotsinclude(test))
            >>> hash_attrs
            ['_hash', '_prop', '_repr']
            >>> sorted(hash_attrs + list(Mro.hash_exclude.val(test))) == sorted(Mro.slots.val(test))
            True
            >>> repr_attrs = sorted(Mro.repr_exclude.slotsinclude(test))
            >>> repr_attrs
            ['_hash', '_prop', '_slot']
            >>> sorted(repr_attrs + list(Mro.repr_exclude.val(test))) == sorted(Mro.slots.val(test))
            True

        Returns:
            Accumulated values from slots - Accumulated values from mro attr name.
        """
        return self.__class__.slots.val(obj).difference(self.val(obj))

    def val(self, obj):
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
            >>> sorted(Mro.hash_exclude.val(test))
            ['_slot']
            >>> sorted(Mro.ignore_attr.val(test))  # doctest: +ELLIPSIS
            [
                'asdict',
                'attr',
                ...
            ]
            >>> Mro.ignore_copy.val(test).difference(Mro.ignore_copy.default)
            {<class 'tuple'>}
            >>> sorted(Mro.ignore_kwarg.val(test))
            ['kwarg']
            >>> Mro.ignore_str.val(test).difference(Mro.ignore_str.default)
            {<class 'tuple'>}
            >>> sorted(Mro.repr_exclude.val(test))
            ['_repr']
            >>> sorted(Mro.slots.val(test))
            ['_hash', '_prop', '_repr', '_slot']

        Returns:
            All/accumulated values of attr in mro and obj if instance.
        """
        return {*(value for item in self.obj(obj) for value in getattr(item, self.value, tuple())), *self.default}


class NameGenValue(Enum):
    def _generate_next_value_(self, start, count, last_values):
        return f'__{self}' if self.endswith('__') else self.removesuffix('_')


class Name(NameGenValue):
    """Name Enum Class."""
    all__ = auto()
    class__ = auto()
    annotations__ = auto()
    builtins__ = auto()
    cached__ = auto()
    code__ = auto()
    contains__ = auto()
    dataclass_fields__ = auto()
    dataclass_params__ = auto()
    delattr__ = auto()
    dict__ = auto()
    dir__ = auto()
    doc__ = auto()
    eq__ = auto()
    file__ = auto()
    getattribute__ = auto()
    len__ = auto()
    loader__ = auto()
    members__ = auto()
    module__ = auto()
    mro__ = auto()
    name__ = auto()
    package__ = auto()
    qualname__ = auto()
    reduce__ = auto()
    repr__ = auto()
    setattr__ = auto()
    slots__ = auto()
    spec__ = auto()
    str__ = auto()
    _asdict = auto()
    add = auto()
    append = auto()
    asdict = auto()
    cls_ = auto()  # To avoid conflict with Name.cls
    clear = auto()
    co_name = auto()
    code_context = auto()
    copy = auto()
    count = auto()
    data = auto()
    endswith = auto()
    extend = auto()
    external = auto()
    f_back = auto()
    f_code = auto()
    f_globals = auto()
    f_lineno = auto()
    f_locals = auto()
    filename = auto()
    frame = auto()
    function = auto()
    get_ = auto()  # To avoid conflict with Name.get
    globals = auto()
    index = auto()
    item = auto()
    items = auto()
    keys = auto()
    kind = auto()
    lineno = auto()
    locals = auto()
    name_ = auto()  # To avoid conflict with Enum.name
    origin = auto()
    obj = auto()
    object = auto()
    REPO = auto()
    pop = auto()
    popitem = auto()
    PYPI = auto()
    remove = auto()
    reverse = auto()
    self_ = auto()  # To avoid conflict with Enum
    sort = auto()
    startswith = auto()
    tb_frame = auto()
    tb_lineno = auto()
    tb_next = auto()
    update = auto()
    value_ = auto()  # To avoid conflict with Enum.value
    values = auto()
    vars = auto()

    @classmethod
    @functools.cache
    def _attrs(cls):
        """
        Get map_reduce (dict) attrs lists converted to real names.

        Examples:
            >>> Name._attrs().keys()
            dict_keys([True, False])
            >>> Name._attrs().values()  # doctest: +ELLIPSIS
            dict_values([['__all__', ...], ['_asdict', ...])

        Returns:
            [True]: private attrs.
            [False[: public attrs.
        """
        return map_reduce(cls.__members__, lambda x: x.endswith('__'), lambda x: cls[x].value)

    @classmethod
    @functools.cache
    def attrs(cls):
        """
        Get attrs tuple with private converted to real names.

        Examples:
            >>> pretty_install()
            >>> Name.attrs()  # doctest: +ELLIPSIS
            (
                '__all__',
                ...,
                '_asdict',
                ...
            )

        Returns:
            Tuple of attributes and values.
        """
        return tuple(collapse(Name._attrs().values()))

    @singledispatchmethod
    def get(self, obj: GetType, default=None):
        """
        Get value from GetType/MutableMapping.

        Examples:
            >>> from ast import unparse
            >>> from inspect import getmodulename
            >>>
            >>> pretty_install()
            >>> f = insstack()[0]
            >>> globs_locs = (f.frame.f_globals | f.frame.f_locals).copy()
            >>> Name.filename.get(f), Name.function.get(f), Name.code_context.get(f)[0], Name.source(f)
            (
                PosixPath('<doctest get[7]>'),
                '<module>',
                'f = insstack()[0]\\n',
                'f = insstack()[0]\\n'
            )
            >>> Name.file__.get(globs_locs)  # doctest: +ELLIPSIS
            PosixPath('/Users/jose/....py')
            >>> assert Name.file__.get(globs_locs) == PathLib(__file__) == PathLib(Name.spec__.get(globs_locs).origin)
            >>> assert Name.name__.get(globs_locs) == getmodulename(__file__) == Name.spec__.get(globs_locs).name
            >>> Name.source(globs_locs)  # doctest: +ELLIPSIS
            '# -*- coding: utf-8 -*-\\n...
            >>> Name.source(__file__)  # doctest: +ELLIPSIS
            '# -*- coding: utf-8 -*-\\n...
            >>> assert Name.source(globs_locs) == Name.source(__file__)
            >>> unparse(Name.node(globs_locs)) == unparse(Name.node(__file__))  # Unparse does not have encoding line
            True
            >>> Name.source(f) in unparse(Name.node(globs_locs))
            True
            >>> Name.source(f) in unparse(Name.node(__file__))
            True

        Args:
            obj: object
            default: None

        Returns:
            Value from get() method
        """
        if self is Name.file__:
            return self.path(obj)
        return obj.get(self.value, default)

    @get.register
    def get_getattrtype(self, obj: GetAttrType, default=None):
        """
        Get value of attribute from GetAttrType.

        Examples:
            >>> from ast import unparse
            >>> from inspect import getmodulename
            >>> import rc.utils
            >>>
            >>> pretty_install()
            >>> f = insstack()[0]
            >>> globs_locs = (f.frame.f_globals | f.frame.f_locals).copy()
            >>> Name.file__.get(globs_locs)  # doctest: +ELLIPSIS
            PosixPath('/Users/jose/....py')
            >>> Name.file__.get(rc.utils)  # doctest: +ELLIPSIS
            PosixPath('/Users/jose/....py')
            >>> PathLib(Name.spec__.get(globs_locs).origin)  # doctest: +ELLIPSIS
            PosixPath('/Users/jose/....py')
            >>> PathLib(Name.spec__.get(rc.utils).origin)  # doctest: +ELLIPSIS
            PosixPath('/Users/jose/....py')
            >>> Name.spec__.get(rc.utils).name == rc.utils.__name__
            True
            >>> Name.spec__.get(rc.utils).name.split('.')[0] == rc.utils.__package__
            True
            >>> Name.name__.get(rc.utils) == rc.utils.__name__
            True
            >>> Name.package__.get(rc.utils) == rc.utils.__package__
            True

        Args:
            obj: object
            default: None

        Returns:
            Value from __getattribute__ method.
        """
        if self is Name.file__:
            try:
                p = object.__getattribute__(obj, Name.file__.value)
                return PathLib(p)
            except AttributeError:
                return self.path(obj)
        try:
            return object.__getattribute__(obj, self.value)
        except AttributeError:
            return default

    @get.register
    def get_frameinfo(self, obj: FrameInfo, default=None):
        """
        Get value of attribute from FrameInfo.

        Examples:
            >>> from ast import unparse
            >>>
            >>> f = insstack()[0]
            >>> assert f == FrameInfo(Name.frame.get(f), str(Name.filename.get(f)), Name.lineno.get(f),\
            Name.function.get(f), Name.code_context.get(f), Name.index.get(f))
            >>> assert Name.filename.get(f) == Name.file__.get(f)
            >>> Name.source(f)
            'f = insstack()[0]\\n'
            >>> unparse(Name.node(f))
            'f = insstack()[0]'
            >>> unparse(Name.node('pass'))
            'pass'
            >>> assert str(Name.file__.get(f)) == str(Name.filename.get(f))
            >>> assert Name.name__.get(f) == Name.co_name.get(f) == Name.function.get(f)
            >>> assert Name.lineno.get(f) == Name.f_lineno.get(f) == Name.tb_lineno.get(f)
            >>> assert Name.vars.get(f) == (f.frame.f_globals | f.frame.f_locals).copy()
            >>> assert Name.vars.get(f) == (Name.f_globals.get(f) | Name.f_locals.get(f)).copy()
            >>> assert Name.vars.get(f) == (Name.globals.get(f) | Name.locals.get(f)).copy()
            >>> assert unparse(Name.node(f)) in Name.code_context.get(f)[0]
            >>> assert Name.spec__.get(f).origin == __file__

        Args:
            obj: object
            default: None

        Returns:
            Value from FrameInfo method
        """
        if self in [Name.file__, Name.filename]:
            return PathLib(obj.filename)
        if self in [Name.name__, Name.co_name, Name.function]:
            return obj.function
        if self in [Name.lineno, Name.f_lineno, Name.tb_lineno]:
            return obj.lineno
        if self in [Name.f_globals, Name.globals]:
            return obj.frame.f_globals
        if self in [Name.f_locals, Name.locals]:
            return obj.frame.f_locals
        if self in [Name.frame, Name.tb_frame]:
            return obj.frame
        if self is Name.vars:
            return (obj.frame.f_globals | obj.frame.f_locals).copy()
        if self in [Name.code__, Name.f_code]:
            return obj.frame.f_code
        if self in [Name.f_back, Name.tb_next]:
            return obj.frame.f_back
        if self is Name.index:
            return obj.index
        if self is Name.code_context:
            return obj.code_context
        return self.get((obj.frame.f_globals | obj.frame.f_locals).copy(), default=default)

    @get.register
    @functools.cache
    def get_frametype(self, obj: FrameType, default=None):
        """
        Get value of attribute from FrameType.

        Examples:
            >>> from ast import unparse
            >>>
            >>> frameinfo = insstack()[0]
            >>> f = frameinfo.frame
            >>> assert Name.filename.get(f) == Name.filename.get(frameinfo)
            >>> assert Name.frame.get(f) == Name.frame.get(frameinfo)
            >>> assert Name.lineno.get(f) == Name.lineno.get(frameinfo)
            >>> assert Name.function.get(f) == Name.function.get(frameinfo)
            >>> assert frameinfo == FrameInfo(Name.frame.get(f), str(Name.filename.get(f)), Name.lineno.get(f),\
            Name.function.get(f), frameinfo.code_context, frameinfo.index)
            >>> assert Name.filename.get(f) == Name.file__.get(f)
            >>> Name.source(f)
            'frameinfo = insstack()[0]\\n'
            >>> unparse(Name.node(f))
            'frameinfo = insstack()[0]'
            >>> unparse(Name.node('pass'))
            'pass'
            >>> assert str(Name.file__.get(f)) == str(Name.filename.get(f))
            >>> assert Name.name__.get(f) == Name.co_name.get(f) == Name.function.get(f)
            >>> assert Name.lineno.get(f) == Name.f_lineno.get(f) == Name.tb_lineno.get(f)
            >>> assert Name.vars.get(f) == (f.f_globals | f.f_locals).copy()
            >>> assert Name.vars.get(f) == (Name.f_globals.get(f) | Name.f_locals.get(f)).copy()
            >>> assert Name.vars.get(f) == (Name.globals.get(f) | Name.locals.get(f)).copy()
            >>> assert unparse(Name.node(f)) in Name.code_context.get(frameinfo)[0]
            >>> assert Name.spec__.get(f).origin == __file__

        Args:
            obj: object
            default: None

        Returns:
            Value from FrameType method
        """
        if self in [Name.file__, Name.filename]:
            return self.path(obj)
        if self in [Name.name__, Name.co_name, Name.function]:
            return obj.f_code.co_name
        if self in [Name.lineno, Name.f_lineno, Name.tb_lineno]:
            return obj.f_lineno
        if self in [Name.f_globals, Name.globals]:
            return obj.f_globals
        if self in [Name.f_locals, Name.locals]:
            return obj.f_locals
        if self in [Name.frame, Name.tb_frame]:
            return obj
        if self is Name.vars:
            return (obj.f_globals | obj.f_locals).copy()
        if self in [Name.code__, Name.f_code]:
            return obj.f_code
        if self in [Name.f_back, Name.tb_next]:
            return obj.f_back
        return self.get((obj.f_globals | obj.f_locals).copy(), default=default)

    @get.register
    @functools.cache
    def get_tracebacktype(self, obj: TracebackType, default=None):
        """
        Get value of attribute from TracebackType.

        Args:
            obj: object
            default: None

        Returns:
            Value from TracebackType method
        """
        if self in [Name.file__, Name.filename]:
            return self.path(obj)
        if self in [Name.name__, Name.co_name, Name.function]:
            return obj.tb_frame.f_code.co_name
        if self in [Name.lineno, Name.f_lineno, Name.tb_lineno]:
            return obj.tb_lineno
        if self in [Name.f_globals, Name.globals]:
            return obj.tb_frame.f_globals
        if self in [Name.f_locals, Name.locals]:
            return obj.tb_frame.f_locals
        if self in [Name.frame, Name.tb_frame]:
            return obj.tb_frame
        if self is Name.vars:
            return (obj.tb_frame.f_globals | obj.tb_frame.f_locals).copy()
        if self in [Name.code__, Name.f_code]:
            return obj.tb_frame.f_code
        if self in [Name.f_back, Name.tb_next]:
            return obj.tb_next
        return self.get((obj.tb_frame.f_globals | obj.tb_frame.f_locals).copy(), default=default)

    @property
    def getter(self):
        """
        Attr getter with real name for private and public which conflicts with Enum.

        Examples:
            >>> import rc.utils
            >>>
            >>> Name.module__.getter(tuple)
            'builtins'
            >>> Name.name__.getter(tuple)
            'tuple'
            >>> Name.file__.getter(rc.utils)  # doctest: +ELLIPSIS
            '/Users/jose/....py'

        Returns:
            Attr getter with real name for private and public which conflicts with Enum.
        """
        return attrgetter(self.value)

    def has(self, obj):
        """
        Checks if has attr with real name for private and public which conflicts with Enum.

        Examples:
            >>> import rc.utils
            >>>
            >>> Name.module__.has(tuple)
            True
            >>> Name.name__.has(tuple)
            True
            >>> Name.file__.has(tuple)
            False
            >>> Name.file__.has(rc.utils)
            True

        Returns:
            Checks if has attr with real name for private and public which conflicts with Enum.
        """
        return hasattr(obj, self.value)

    @classmethod
    def node(cls, obj, complete=False, line=False):
        """
        Get node of object.

        Examples:
            >>> from ast import unparse
            >>> from inspect import getmodulename
            >>>
            >>> pretty_install()
            >>> f = insstack()[0]
            >>> globs_locs = (f.frame.f_globals | f.frame.f_locals).copy()
            >>> Name.filename.get(f), Name.function.get(f), Name.code_context.get(f)[0], Name.source(f) \
             # doctest: +ELLIPSIS
            (
                PosixPath('<doctest ...node[...]>'),
                '<module>',
                'f = insstack()[0]\\n',
                'f = insstack()[0]\\n'
            )
            >>> Name.file__.get(globs_locs)  # doctest: +ELLIPSIS
            PosixPath('/Users/jose/....py')
            >>> assert Name.file__.get(globs_locs) == PathLib(__file__) == PathLib(Name.spec__.get(globs_locs).origin)
            >>> assert Name.name__.get(globs_locs) == getmodulename(__file__) == Name.spec__.get(globs_locs).name
            >>> Name.source(globs_locs)  # doctest: +ELLIPSIS
            '# -*- coding: utf-8 -*-\\n...
            >>> Name.source(__file__)  # doctest: +ELLIPSIS
            '# -*- coding: utf-8 -*-\\n...
            >>> assert Name.source(globs_locs) == Name.source(__file__)
            >>> unparse(Name.node(globs_locs)) == unparse(Name.node(__file__))  # Unparse does not have encoding line
            True
            >>> Name.source(f) in unparse(Name.node(globs_locs))
            True
            >>> Name.source(f) in unparse(Name.node(__file__))
            True

        Args:
            obj: object.
            complete: return complete node for file (always for module and frame corresponding to module)
                or object node (default=False)
            line: return line

        Returns:
            Node.
        """
        return ast.parse(cls.source(obj, complete, line) or str(obj))

    @classmethod
    def path(cls, obj):
        """
        Get path of object.

        Examples:
            >>> from ast import unparse
            >>> from inspect import getmodulename
            >>>
            >>> pretty_install()
            >>> frameinfo = insstack()[0]
            >>> globs_locs = (frameinfo.frame.f_globals | frameinfo.frame.f_locals).copy()
            >>> Name.path(Name.path)  # doctest: +ELLIPSIS
            PosixPath('/Users/jose/....py')
            >>> Name.path(__file__)  # doctest: +ELLIPSIS
            PosixPath('/Users/jose/....py')
            >>> Name.path(allin)  # doctest: +ELLIPSIS
            PosixPath('/Users/jose/....py')
            >>> Name.path(dict(a=1))
            PosixPath("{'a': 1}")

        Args:
            obj: object.

        Returns:
            Path.
        """
        es = Es(obj)
        if es.mm:
            f = obj.get(Name.file__.value)
        elif es.frameinfo:
            f = obj.filename
        else:
            try:
                f = getsourcefile(obj) or getfile(obj)
            except TypeError:
                f = None
        return PathLib(f or str(obj))

    @classmethod
    @functools.cache
    def private(cls):
        """
        Get private attrs tuple converted to real names.

        Examples:
            >>> pretty_install()
            >>> Name.private()  # doctest: +ELLIPSIS
            (
                '__all__',
                ...
            )

        Returns:
            Tuple of private attrs converted to real names.
        """
        return tuple(cls._attrs()[True])

    @classmethod
    @functools.cache
    def public(cls):
        """
        Get public attrs tuple.

        Examples:
            >>> pretty_install()
            >>> Name.public()  # doctest: +ELLIPSIS
            (
                '_asdict',
                ...
            )

        Returns:
            Tuple of public attrs.
        """
        return tuple(cls._attrs()[False])

    @classmethod
    def _source(cls, obj, line=False):
        f = cls.file__.get(obj) or obj
        if (p := PathLib(f)).is_file():
            if s := token_open(p):
                if line:
                    return s, 1
                return s

    @classmethod
    def source(cls, obj, complete=False, line=False):
        """
        Get source of object.

        Examples:
            >>> from ast import unparse
            >>> from inspect import getmodulename
            >>> import rc.utils
            >>>
            >>> pretty_install()
            >>>
            >>> Name.source(__file__)  # doctest: +ELLIPSIS
            '# -*- coding: utf-8 -*-\\n...
            >>> Name.source(__file__, complete=True)  # doctest: +ELLIPSIS
            '# -*- coding: utf-8 -*-\\n...
            >>>
            >>> Name.source(Name.source)  # doctest: +ELLIPSIS
            '    @classmethod\\n    def source(cls, obj, complete=False, line=False):\\n...'
            >>> Name.source(Name.source, complete=True)  # doctest: +ELLIPSIS
            '# -*- coding: utf-8 -*-\\n...
            >>> Name.source(Name.source).splitlines()[1] in Name.source(Name.source, complete=True)
            True
            >>>
            >>> Name.source(allin)  # doctest: +ELLIPSIS
            'def allin(origin, destination):\\n...'
            >>> Name.source(allin, line=True)  # doctest: +ELLIPSIS
            (
                'def allin(origin, destination):\\n...',
                ...
            )
            >>> Name.source(allin, complete=True)  # doctest: +ELLIPSIS
            '# -*- coding: utf-8 -*-\\n...
            >>> Name.source(allin, complete=True, line=True)  # doctest: +ELLIPSIS
            (
                '# -*- coding: utf-8 -*-\\n...,
                ...
            )
            >>> Name.source(allin).splitlines()[0] in Name.source(allin, complete=True)
            True
            >>>
            >>> Name.source(dict(a=1))
            "{'a': 1}"
            >>> Name.source(dict(a=1), complete=True)
            "{'a': 1}"
            >>>
            >>> Name.source(rc.utils)  # doctest: +ELLIPSIS
            '# -*- coding: utf-8 -*-\\n...
            >>> Name.source(rc.utils, complete=True)  # doctest: +ELLIPSIS
            '# -*- coding: utf-8 -*-\\n...
            >>>
            >>> frameinfo = insstack()[0]
            >>> Name.source(frameinfo), frameinfo.function
            ('frameinfo = insstack()[0]\\n', '<module>')
            >>> Name.source(frameinfo, complete=True), frameinfo.function
            ('frameinfo = insstack()[0]\\n', '<module>')
            >>>
            >>> frametype = frameinfo.frame
            >>> Name.source(frametype), frametype.f_code.co_name
            ('frameinfo = insstack()[0]\\n', '<module>')
            >>> Name.source(frameinfo, complete=True), frametype.f_code.co_name
            ('frameinfo = insstack()[0]\\n', '<module>')
            >>>
            >>> Name.source(None)
            'None'

        Args:
            obj: object.
            complete: return complete source file (always for module and frame corresponding to module)
                or object source (default=False)
            line: return line

        Returns:
            Source.
        """
        es = Es(obj)
        if any([es.moduletype, (es.frameinfo and obj.function == FUNCTION_MODULE),
                (es.frametype and obj.f_code.co_name == FUNCTION_MODULE) or
                (es.tracebacktype and obj.tb_frame.f_code.co_name == FUNCTION_MODULE) or
                complete]):
            if source := cls._source(obj, line):
                return source

        try:
            if line:
                lines, lnum = getsourcelines(obj.frame if es.frameinfo else obj)
                return ''.join(lines), lnum
            return getsource(obj.frame if es.frameinfo else obj)
        except (OSError, TypeError):
            if source := cls._source(obj, line):
                return source
            if line:
                return str(obj), 1
            return str(obj)


# </editor-fold>
# <editor-fold desc="Bases">
Annotation = namedtuple('Annotation', 'any args classvar cls default final hint initvar literal name optional '
                                      'origin union')


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


class Es:
    # noinspection PyUnresolvedReferences
    """
        Is Instance, Subclass, etc. Helper Class

        Examples:
            >>> pretty_install()
            >>>
            >>> es = Es(2)
            >>> es.int
            True
            >>> es.bool
            False
            >>> es.instance(dict, tuple)
            False
            >>> es(dict, tuple)
            False
            >>> def func(): pass
            >>> Es(func).coro
            False
            >>> async def async_func(): pass
            >>> es = Es(async_func)
            >>> es.coro, es.coroutinefunction, es.asyncgen, es.asyncgenfunction, es.awaitable, es.coroutine
            (True, True, False, False, False, False)
            >>> es = {i.name: Es(i.object) for i in classify_class_attrs(TestAsync)}
            >>> es['async_classmethod'].coro, es['async_classmethod'].coroutinefunction, \
            es['async_classmethod'].asyncgen, es['async_classmethod'].asyncgenfunction, \
            es['async_classmethod'].awaitable, es['async_classmethod'].coroutine
            (True, True, False, False, False, False)
            >>> es['async_method'].coro, es['async_method'].coroutinefunction, \
            es['async_method'].asyncgen, es['async_method'].asyncgenfunction, \
            es['async_method'].awaitable, es['async_method'].coroutine
            (True, True, False, False, False, False)
            >>> es['async_pprop'].coro, es['async_pprop'].coroutinefunction, \
            es['async_pprop'].asyncgen, es['async_pprop'].asyncgenfunction, \
            es['async_pprop'].awaitable, es['async_pprop'].coroutine
            (True, True, False, False, False, False)
            >>> es['async_prop'].coro, es['async_prop'].coroutinefunction, \
            es['async_prop'].asyncgen, es['async_prop'].asyncgenfunction, \
            es['async_prop'].awaitable, es['async_prop'].coroutine
            (True, True, False, False, False, False)
            >>> es['async_staticmethod'].coro, es['async_staticmethod'].coroutinefunction, \
            es['async_staticmethod'].asyncgen, es['async_staticmethod'].asyncgenfunction, \
            es['async_staticmethod'].awaitable, es['async_staticmethod'].coroutine
            (True, True, False, False, False, False)
            >>> assert es['classmethod'].coro == False
            >>> assert es['cprop'].coro == False
            >>> assert es['method'].coro == False
            >>> assert es['pprop'].coro == False
            >>> assert es['prop'].coro == False
            >>> assert es['staticmethod'].coro == False

        Attributes:
        -----------
        data: Any
            object to provide information (default: None)
        """
    __slots__ = ('data',)

    def __init__(self, data=None): self.data = data

    def __call__(self, *args): return isinstance(self.data, args)

    # TODO: Test: __getstate__ and doc.
    def __getstate__(self): return dict(data=self.data)

    def __repr__(self): return f'{self.__class__.__name__}({self.data})'

    # TODO: Test: __setstate__ and doc.
    def __setstate__(self, state): self.data = state['data']

    def __str__(self): return str(self.data)

    _func = property(
        lambda self:
        self.data.fget if self.prop else self.data.__func__ if self(classmethod, staticmethod) else self.data)
    annotation = property(lambda self: self(Annotation))
    annotationstype = property(lambda self: self(AnnotationsType))
    annotationstype_sub = property(lambda self: self.subclass(AnnotationsType))
    # TODO: asdict_props
    # asdict = property(lambda self: dict(data=self.data) | asdict_props(self))
    asdictmethod = property(lambda self: self(AsDictMethodType))
    asdictmethod_sub = property(lambda self: self.subclass(AsDictMethodType))
    asdictproperty = property(lambda self: self(AsDictPropertyType))
    asdictproperty_sub = property(lambda self: self.subclass(AsDictPropertyType))
    ast = property(lambda self: self(AST))
    asyncfor = property(lambda self: isinstance(self._func, AsyncFor))
    asyncfunctiondef = property(lambda self: isinstance(self._func, AsyncFunctionDef))
    asyncgen = property(lambda self: isinstance(self._func, AsyncGeneratorType))
    asyncgenfunction = property(lambda self: isasyncgenfunction(self._func))
    asyncwith = property(lambda self: isinstance(self._func, AsyncWith))
    # TODO: Attribute
    # attribute = property(lambda self: self(Attribute))
    await_ast = property(lambda self: isinstance(self._func, Await))
    awaitable = property(lambda self: isawaitable(self._func))
    bool = property(lambda self: self(int) and self(bool))
    builtin = property(lambda self: any([in_dict(BUILTINS, self.data), self.builtinclass, self.builtinfunction]))
    builtinclass = property(lambda self: self.data in BUILTINS_CLASSES)
    builtinfunction = property(lambda self: self.data in BUILTINS_FUNCTIONS)
    builtinfunctiontype = property(lambda self: self(BuiltinFunctionType))
    bytesio = property(lambda self: self(BytesIO))  # :class:`typing.BinaryIO`
    cache = property(lambda self: Mro.cache_clear.has(self.data))
    cached_property = property(lambda self: self(cached_property))
    callable = property(lambda self: self(Callable))
    chain = property(lambda self: self(Chain))
    chainmap = property(lambda self: self(ChainMap))
    classdef = property(lambda self: self(ClassDef))
    classmethod = property(lambda self: self(classmethod))
    classmethoddescriptortype = property(lambda self: self(ClassMethodDescriptorType))
    classvar = property(
        lambda self: (self.datafield and get_origin(self.data.type) == ClassVar) or get_origin(self.data) == ClassVar)
    codetype = property(lambda self: self(CodeType))
    collections = property(lambda self: is_collections(self.data))
    container = property(lambda self: self(Container))
    coro = property(lambda self: any(
        [self.asyncfor, self.asyncfunctiondef, self.asyncwith, self.await_ast] if self.ast else
        [self.asyncgen, self.asyncgenfunction, self.awaitable, self.coroutine, self.coroutinefunction]))
    coroutine = property(lambda self: iscoroutine(self._func) or isinstance(self._func, CoroutineType))
    coroutinefunction = property(lambda self: iscoroutinefunction(self._func))
    datafactory = property(
        lambda self: self.datafield and Es(self.data.default).missing and hasattr(self.data, 'default_factory'))
    datafield = property(lambda self: self(Field))
    datatype = property(lambda self: self(DataType))
    datatype_sub = property(lambda self: self.subclass(DataType))
    defaultdict = property(lambda self: self(defaultdict))
    deleter = property(lambda self: self.property_any and self.data.fdel is not None)
    dict = property(lambda self: self(dict))
    dicttype = property(lambda self: self(DictType))
    dicttype_sub = property(lambda self: self.subclass(DictType))
    dynamicclassattribute = property(lambda self: self(DynamicClassAttribute))
    dlst = property(lambda self: self(dict, list, set, tuple))
    enum = property(lambda self: self(Enum))
    enum_sub = property(lambda self: self.subclass(Enum))
    enumdict = property(lambda self: self(EnumDict))
    enumdict_sub = property(lambda self: self.subclass(EnumDict))
    even: property(lambda self: not self.data % 2)
    fileio = property(lambda self: self(FileIO))
    float = property(lambda self: self(float))
    frameinfo = property(lambda self: self(FrameInfo))
    frametype = property(lambda self: self(FrameType))
    functiondef = property(lambda self: self(FunctionDef))
    functiontype = property(lambda self: self(FunctionType))
    generator = property(lambda self: self(Generator))
    generatortype = property(lambda self: self(GeneratorType))
    genericalias = property(lambda self: self(types.GenericAlias))
    getattrnobuiltintype = property(lambda self: self(GetAttrNoBuiltinType))
    getattrnobuiltintype_sub = property(lambda self: self.subclass(GetAttrNoBuiltinType))
    getattrtype = property(lambda self: self(GetAttrType))
    getattrtype_sub = property(lambda self: self.subclass(GetAttrType))
    getsetdescriptor = lambda self, n: isgetsetdescriptor(self.cls_attr_value(n)) if n else self.data
    getsetdescriptortype = property(lambda self: self(GetSetDescriptorType))
    gettype = property(lambda self: self(GetType))
    gettype_sub = property(lambda self: self.subclass(GetType))
    hashable = property(lambda self: bool(noexception(TypeError, hash, self.data)))
    import_ast = property(lambda self: self(Import))
    importfrom = property(lambda self: self(ImportFrom))
    initvar = property(
        lambda self: (self.datafield and isinstance(self.data.type, InitVar)) or self(InitVar))
    installed = property(lambda self: is_installed(self.data))
    instance = lambda self, *args: isinstance(self.data, args)
    int = property(lambda self: self(int))
    io = property(lambda self: self.bytesio and self.stringio)  # :class:`typing.IO`
    iterable = property(lambda self: self(Iterable))
    iterator = property(lambda self: self(Iterator))
    lambdatype = property(lambda self: self(LambdaType))
    list = property(lambda self: self(list))
    lst = property(lambda self: self(list, set, tuple))
    mappingproxytype = property(lambda self: self(MappingProxyType))
    mappingproxytype_sub = property(lambda self: self.subclass(MappingProxyType))
    memberdescriptor = property(lambda self: ismemberdescriptor(self.data))
    memberdescriptortype = property(lambda self: self(MemberDescriptorType))
    # TODO: Meta
    # meta = property(lambda self: self(Meta))
    # meta_sub = property(lambda self: self.subclass(Meta))
    # metatype = property(lambda self: self(MetaType))
    # metatype_sub = property(lambda self: self.subclass(MetaType))
    method = property(lambda self: self.methodtype and not self(classmethod, property, staticmethod))
    methoddescriptor = property(lambda self: ismethoddescriptor(self.data))
    methoddescriptortype = property(lambda self: self(types.MethodDescriptorType))
    methodtype = property(lambda self: self(MethodType))  # True if it is an instance method!.
    methodwrappertype = property(lambda self: self(MethodWrapperType))
    methodwrappertype_sub = property(lambda self: self.subclass(MethodWrapperType))
    missing = property(lambda self: self(MISSING_TYPE))
    mlst = property(lambda self: self(MutableMapping, list, set, tuple))
    mm = property(lambda self: self(MutableMapping))
    moduletype = property(lambda self: self(ModuleType))
    module_function = property(lambda self: is_module_function(self.data))
    noncomplex = property(lambda self: is_noncomplex(self.data))
    namedtype = property(lambda self: self(NamedType))
    namedtype_sub = property(lambda self: self.subclass(NamedType))
    named_annotationstype = property(lambda self: self(NamedAnnotationsType))
    named_annotationstype_sub = property(lambda self: self.subclass(NamedAnnotationsType))
    none = property(lambda self: self(type(None)))
    object = property(lambda self: is_object(self.data))
    pathlib = property(lambda self: self(PathLib))
    picklable = lambda self, name: is_picklable(name, self.data)
    primitive = property(lambda self: is_primitive(self.data))
    pproperty = property(lambda self: self(pproperty))
    prop = property(lambda self: self(property))
    property_any = property(lambda self: self.prop or self.cached_property)
    reducible = property(lambda self: is_reducible(self.data))
    reducible_sequence_subclass = property(lambda self: is_reducible_sequence_subclass(self.data))
    routine = property(lambda self: isroutine(self.data))
    sequence = property(lambda self: self(Sequence))
    sequence_sub = property(lambda self: self.subclass(Sequence))
    set = property(lambda self: self(set))
    setter = property(lambda self: self.prop and self.data.fset is not None)
    simple = property(lambda self: self(Simple))
    sized = property(lambda self: self(Sized))
    # TODO: slist
    # slist = property(lambda self: self(slist))
    slotstype = property(lambda self: self(SlotsType))
    slotstype_sub = property(lambda self: self.subclass(SlotsType))
    staticmethod = property(lambda self: self(staticmethod))
    str = property(lambda self: self(str))
    # TODO: stuple
    # stuple = property(lambda self: self(stuple))
    subclass = lambda self, *args: self.type and issubclass(self.data, args)
    stringio = property(lambda self: self(StringIO))  # :class:`typing.TextIO`
    tracebacktype = property(lambda self: self(TracebackType))
    tuple = property(lambda self: self(tuple))
    type = property(lambda self: self(type))
    unicode = property(lambda self: is_unicode(self.data))
    wrapperdescriptortype = property(lambda self: self(WrapperDescriptorType))

    def __class_getitem__(cls, prop):
        # TODO: Examples: __class_getitem__
        """
        Getter with Property Name.

        Examples:
            >>> pretty_install()
            >>>

        Args:
            prop: property

        Returns:
            Getter with Property Name.
        """
        return attrgetter(n) if ((n := cls(prop).property_any.__name__) in cls.__dict__) else NotImplemented


class pproperty(property):
    """
    Print property.

    Examples:
        >>> pretty_install()
        >>> class Test:
        ...     _pp = 0
        ...     @pproperty
        ...     def pp(self):
        ...         self._pp += 1
        ...         prop = Es(self.__class__.__dict__['pp']).prop
        ...         pprop = Es(self.__class__.__dict__['pp']).pproperty
        ...         return self._pp, prop, pprop
        >>> test = Test()
        >>> test.pp
        (1, True, True)
        >>> test.pp
        (2, True, True)
    """

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        super().__init__(fget=fget, fset=fset, fdel=fdel, doc=doc)


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


def annotations(obj, stack=1):
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
        obj: obj.
        stack: stack index to extract globals and locals (default: 1) or frame.

    Returns:
        Annotation: obj annotations. Default are filled with annotation not with class default.
    """

    def value(_cls):
        # TODO: 1) default from annotations, 2) value from kwargs or class defaults.
        return noexception(Exception, _cls)

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
    if a := noexception(TypeError, get_type_hints, obj, globalns=frame.f_globals, localns=frame.f_locals):
        for name, hint in a.items():
            rv |= {name: inner(hint)}
    return dict_sort(rv)


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


@singledispatch
def get(data: GetType, name, default=None):
    """
    Get value of name in Mutabble Mapping/GetType or object.

    Examples:
        >>> get(dict(a=1), 'a')
        1
        >>> get(dict(a=1), 'b')
        >>> get(dict(a=1), 'b', 2)
        2

    Args:
        data: MutabbleMapping/GetType to get value.
        name: key.
        default: default value (default:None)

    Returns:
        Value for key.
    """
    return data.get(name, default)


@get.register
def get_getattrtype(data: GetAttrType, name, default=None):
    """
    Get value of name in Mutabble Mapping or object.

    Examples:
        >>> get(dict, '__module__')
        'builtins'
        >>> get(dict, '__file__')

    Args:
        data: object to get value.
        name: attr name.
        default: default value (default:None)

    Returns:
        Value for attribute.
    """
    try:
        return object.__getattribute__(data, name)
    except AttributeError:
        return default


def getset(data, name, default=None):
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

    Returns:
        Attribute value or sets default value and returns.
    """
    try:
        return object.__getattribute__(data, name)
    except AttributeError:
        object.__setattr__(data, name, default)
        return object.__getattribute__(data, name)


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


def newprop(name=None, default=None, pprop=False):
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
        pprop: pproperty or property (default: False)

    Returns:
        Property.
    """
    func = pproperty if pprop else property
    name = f'_{name if name else varname()}'
    return func(
        lambda self:
        getset(self, name, default=default(self) if Es(default).instance(Callable, partial) else default),
        lambda self, value: self.__setattr__(name, value),
        lambda self: self.__delattr__(name)
    )


def noexception(exception, func, *args, default_=None, **kwargs):
    """
    Execute function suppressing exceptions.

    Examples:
        >>> noexception(KeyError, dict(a=1).pop, 'b', default_=2)
        2

    Args:
        exception: exception or exceptions.
        func: callable.
        *args: args.
        default_: default value if exception is raised.
        **kwargs: kwargs.

    Returns:
        Any: Function return.
    """
    with suppress(exception):
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
        >>> to_camel(Mro.ignore_attr.name)
        'IgnoreAttr'
        >>> to_camel(Mro.ignore_attr.name, replace=False)
        'Ignore_Attr'
        >>> to_camel(Mro.ignore_attr.value, replace=False)
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


def token_open(file):
    """
    Read file with tokenize to use in nested classes ast node.

    Args:
        file: filename

    Returns:
        Source
    """
    with tokenize.open(str(file)) as f:
        return f.read()


def varname(index: int = 2, lower=True, sep: str = '_') -> Optional[str]:
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
    _async_classmethod = varname(1)
    _classmethod = varname(1)
    _async_method = varname(1)
    _method = varname(1)
    _cprop = varname(1)
    _async_pprop = varname(1)
    _pprop = varname(1)
    _async_prop = varname(1)
    _prop = varname(1)
    _async_staticmethod = varname(1)
    _staticmethod = varname(1)

    @classmethod
    async def async_classmethod(cls): return cls._async_classmethod

    @classmethod
    def classmethod(cls): return cls._classmethod

    async def async_method(self): return self._async_method

    def method(self): return self._method

    @cached_property
    def cprop(self): return self._cprop

    @pproperty
    async def async_pprop(self): return self._async_pprop

    @pproperty
    def pprop(self): return self._pprop

    @property
    async def async_prop(self): return self._async_prop

    @property
    def prop(self): return self._prop

    @staticmethod
    async def async_staticmethod(): return TestAsync._async_staticmethod

    @staticmethod
    def staticmethod(): return TestAsync._staticmethod
# </editor-fold>
