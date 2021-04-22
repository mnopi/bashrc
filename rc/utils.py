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
    'PathLib',

    'Environs',
    'GitRepo',

    'BUILTIN_CLASSES',
    'FRAME_SYS_INIT',
    'FUNCTION_MODULE',
    'NEWLINE',
    'PYTHON_SYS',
    'PYTHON_SITE',

    'Alias',
    'CRLock',
    'console',
    'DATACLASS_FIELDS',
    'debug',
    'fmic',
    'fmicc',
    'ic',
    'icc',
    'IGNORE_COPY',
    'IGNORE_STR',
    'LST',
    'MISSING_TYPE',
    'POST_INIT_NAME',
    'pp',
    'print_exception',
    'RunningLoop',
    'SeqNoStr',
    'Seq',

    'pproperty',
    'runwarning',

    'Annotation',
    'AnnotationsType',
    'AsDictClassMethodType',
    'AsDictMethodType',
    'AsDictPropertyType',
    'AsDictStaticMethodType',
    'Attr',
    'Base',
    'BoxKeys',
    'ChainRV',
    'Chain',
    'Cls',
    'CmdError',
    'CmdAioError',
    'DataType',
    'DictType',
    'EnumDict',
    'EnumDictType',
    'Es',
    'Executor',
    'GetAttrNoBuiltinType',
    'GetAttrType',
    'GetSupport',
    'GetType',
    'Instance',
    'Mro',
    'Name',
    'NamedType',
    'NamedAnnotationsType',
    'SlotsType',
    'Types',

    'aioloop',
    'allin',
    'annotations',
    'annotations_init',
    'anyin',
    'cmd',
    'cmdname',
    'current_task_name',
    'dict_sort',
    'dict_sort',
    'get',
    'getset',
    'info',
    'is_even',
    'join_newline',
    'map_reduce_even',
    'map_with_args',
    'noexception',
    'prefixed',
    'repr_format',
    'startswith',
    'to_iter',
    'varname',
    'yield_if',
    'yield_last',

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
)

import _abc
import ast
import inspect
import re
import subprocess
import sys
import textwrap
import tokenize
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
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import suppress
from copy import copy
from dataclasses import _FIELDS
from dataclasses import _MISSING_TYPE
from dataclasses import _POST_INIT_NAME
from dataclasses import fields
from dataclasses import InitVar
from dataclasses import is_dataclass
from enum import auto
from enum import Enum
from functools import cache
from functools import partial
from functools import singledispatch
from functools import singledispatchmethod
from inspect import classify_class_attrs
from inspect import FrameInfo
from inspect import getfile
from inspect import getmodule
from inspect import getsource
from inspect import getsourcefile
from inspect import getsourcelines
from inspect import isasyncgen
from inspect import isasyncgenfunction
from inspect import isawaitable
from inspect import iscoroutinefunction
from inspect import isgetsetdescriptor
from inspect import ismemberdescriptor
from inspect import ismethoddescriptor
from inspect import isroutine
from operator import attrgetter
from pathlib import Path as PathLib
from subprocess import CompletedProcess
from threading import _CRLock
from types import BuiltinFunctionType
from types import CodeType
from types import FrameType
from types import FunctionType
from types import LambdaType
from types import MethodWrapperType
from types import ModuleType
from types import TracebackType
from typing import _alias
from typing import Any
from typing import BinaryIO
from typing import ByteString
from typing import Callable
from typing import ClassVar
from typing import Final
from typing import Generator
from typing import get_args
from typing import get_origin
from typing import get_type_hints
from typing import IO
from typing import Iterable
from typing import Iterator
from typing import KeysView
from typing import Literal
from typing import MutableMapping
from typing import MutableSequence
from typing import MutableSet
from typing import NamedTuple
from typing import Optional
from typing import Protocol
from typing import runtime_checkable
from typing import Sequence
from typing import TextIO
from typing import Type
from typing import Union
from typing import ValuesView
from warnings import catch_warnings
from warnings import filterwarnings

from box import Box
from bson import ObjectId
from click import secho
from click.exceptions import Exit
from decorator import decorator
from devtools import Debug
from environs import Env as Environs
from git import GitConfigParser
from git import Remote
from git import Repo as GitRepo
from git.refs import SymbolicReference as GitSymbolicReference
from icecream import IceCreamDebugger
from jsonpickle.util import has_method
from jsonpickle.util import has_reduce
from jsonpickle.util import importable_name
from jsonpickle.util import is_collections
from jsonpickle.util import is_installed
from jsonpickle.util import is_module_function
from jsonpickle.util import is_noncomplex
from jsonpickle.util import is_object
from jsonpickle.util import is_picklable
from jsonpickle.util import is_primitive
from jsonpickle.util import is_reducible
from jsonpickle.util import is_reducible_sequence_subclass
from jsonpickle.util import is_sequence
from jsonpickle.util import is_sequence_subclass
from jsonpickle.util import is_unicode
from more_itertools import bucket
from more_itertools import collapse
from more_itertools import first_true
from more_itertools import map_reduce
from rich import pretty
from rich.console import Console

BUILTIN_CLASSES = tuple(filter(
    lambda x:
    isinstance(x, type) and not issubclass(x, BaseException), copy(globals()['__builtins__']).values()))
FRAME_SYS_INIT = sys._getframe(0)
FUNCTION_MODULE = '<module>'
NEWLINE = '\n'
PYTHON_SYS = PathLib(sys.executable)
PYTHON_SITE = PathLib(PYTHON_SYS).resolve()

Alias = _alias
CRLock = _CRLock
console = Console(color_system='256')
DATACLASS_FIELDS = _FIELDS
debug = Debug(highlight=True)
fmic = IceCreamDebugger(prefix=str()).format
fmicc = IceCreamDebugger(prefix=str(), includeContext=True).format
ic = IceCreamDebugger(prefix=str())
icc = IceCreamDebugger(prefix=str(), includeContext=True)
IGNORE_COPY = (CRLock, Environs, FrameType, GitConfigParser, GitSymbolicReference, Remote, )
"""True or class for repr instead of nested asdict and deepcopy. No deepcopy (default: (:class:`rc.CRLock`,
:class:`rc.Environs`, :class:`types.FrameType`, :class:`git.GitConfigParser`, :class:`rc.GitSymbolicReference`,
:class:`git.Remote`, ))"""
IGNORE_STR = (GitConfigParser, GitRepo, ObjectId, PathLib, )
"""Use str value for object (default: (:class:`git.GitConfigParser`, :class:`rc.GitRepo`, :class:`bson.ObjectId`, 
:class:`rc.PathLib`, ))."""
LST = Union[MutableSet, MutableSequence, tuple]
MISSING_TYPE = _MISSING_TYPE
POST_INIT_NAME = _POST_INIT_NAME
pp = console.print
print_exception = console.print_exception
pretty.install(console=console, expand_all=True)
# rich.traceback.install(console=console, extra_lines=5, show_locals=True)
RunningLoop = _RunningLoop
SeqNoStr = Union[LST, KeysView, ValuesView, Iterator]
Seq = Union[SeqNoStr, Sequence, ByteString, str, bytes]


class pproperty(property):
    """
    Print property.

    Examples:
        >>> from functools import cache
        >>> from rich import pretty
        >>> from rc import pproperty
        >>>
        >>> pretty.install()
        >>> class Test:
        ...     _pp = 0
        ...     @pproperty
        ...     @cache
        ...     def pp(self):
        ...         self._pp += 1
        ...         prop = isinstance(self.__class__.__dict__['pp'], property)
        ...         pprop = isinstance(self.__class__.__dict__['pp'], pproperty)
        ...         return self._pp, prop, pprop
        >>> test = Test()
        >>> test.pp
        (1, True, True)
        >>> test.pp
        (1, True, True)
    """
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        super().__init__(fget=fget, fset=fset, fdel=fdel, doc=doc)


@decorator
def runwarning(func, *args, **kwargs):
    with catch_warnings(record=False):
        filterwarnings('ignore', category=RuntimeWarning)
        warnings.showwarning = lambda *_args, **_kwargs: None
        rv = func(*args, **kwargs)
        return rv


Annotation = namedtuple('Annotation', 'any args classvar cls default final hint initvar literal name optional '
                                      'origin union')


class AnnotationsType(metaclass=ABCMeta):
    """
    Annotations Type.

    Examples:
        >>> from collections import namedtuple
        >>> from typing import NamedTuple
        >>>
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
    __subclasshook__ = classmethod(lambda cls, C: cls is AnnotationsType and '__annotations__' in C.__dict__)


class AsDictClassMethodType(metaclass=ABCMeta):
    """
    AsDict Class Method Type.

    Examples:
        >>> from rc import AsDictClassMethodType
        >>> from rc import info
        >>>
        >>> class AsDictClass: asdict = classmethod(lambda cls, *args, **kwargs: dict())
        >>> class AsDictMethod: asdict = lambda self, *args, **kwargs: dict()
        >>> class AsDictProperty: asdict = property(lambda self: dict())
        >>> class AsDictStatic: asdict = staticmethod(lambda cls, *args, **kwargs: dict())
        >>>
        >>> c = AsDictClass()
        >>> m = AsDictMethod()
        >>> p = AsDictProperty()
        >>> s = AsDictStatic()
        >>>
        >>> Es(AsDictClass).asdict_classmethodtype_sub
        True
        >>> Es(c).asdict_classmethodtype
        True
        >>>
        >>> Es(AsDictMethod).asdict_classmethodtype_sub
        False
        >>> Es(m).asdict_classmethodtype
        False
        >>>
        >>> Es(AsDictProperty).asdict_classmethodtype_sub
        False
        >>> Es(p).asdict_classmethodtype
        False
        >>>
        >>> Es(AsDictStatic).asdict_classmethodtype_sub
        False
        >>> Es(s).asdict_classmethodtype
        False
        """
    __subclasshook__ = classmethod(
        lambda cls, C:
        cls is AsDictClassMethodType and 'asdict' in C.__dict__ and Es(C.__dict__['asdict']).classmethod)
    asdict = classmethod(lambda cls, *args, **kwargs: dict())


class AsDictMethodType(metaclass=ABCMeta):
    """
    AsDict Method Type.

    Examples:
        >>> from rc import AsDictMethodType
        >>> from rc import info
        >>>
        >>> class AsDictClass: asdict = classmethod(lambda cls, *args, **kwargs: dict())
        >>> class AsDictMethod: asdict = lambda self, *args, **kwargs: dict()
        >>> class AsDictProperty: asdict = property(lambda self: dict())
        >>> class AsDictStatic: asdict = staticmethod(lambda cls, *args, **kwargs: dict())
        >>>
        >>> c = AsDictClass()
        >>> m = AsDictMethod()
        >>> p = AsDictProperty()
        >>> s = AsDictStatic()
        >>>
        >>> Es(AsDictClass).asdict_methodtype_sub
        False
        >>> Es(c).asdict_methodtype
        False
        >>>
        >>> Es(AsDictMethod).asdict_methodtype_sub
        True
        >>> Es(m).asdict_methodtype
        True
        >>>
        >>> Es(AsDictProperty).asdict_methodtype_sub
        False
        >>> Es(p).asdict_methodtype
        False
        >>>
        >>> Es(AsDictStatic).asdict_methodtype_sub
        False
        >>> Es(s).asdict_methodtype
        False
    """
    __subclasshook__ = classmethod(
        lambda cls, C: cls is AsDictMethodType and 'asdict' in C.__dict__ and Es(C.__dict__['asdict']).method)
    asdict = lambda self, *args, **kwargs: dict()


class AsDictPropertyType(metaclass=ABCMeta):
    """
    AsDict Property Type.

    Examples:
        >>> from rc import AsDictPropertyType
        >>> from rc import info
        >>>
        >>> class AsDictClass: asdict = classmethod(lambda cls, *args, **kwargs: dict())
        >>> class AsDictMethod: asdict = lambda self, *args, **kwargs: dict()
        >>> class AsDictProperty: asdict = property(lambda self: dict())
        >>> class AsDictStatic: asdict = staticmethod(lambda cls, *args, **kwargs: dict())
        >>>
        >>> c = AsDictClass()
        >>> m = AsDictMethod()
        >>> p = AsDictProperty()
        >>> s = AsDictStatic()
        >>>
        >>> Es(AsDictClass).asdict_propertytype_sub
        False
        >>> Es(c).asdict_propertytype
        False
        >>>
        >>> Es(AsDictMethod).asdict_propertytype_sub
        False
        >>> Es(m).asdict_propertytype
        False
        >>>
        >>> Es(AsDictProperty).asdict_propertytype_sub
        True
        >>> Es(p).asdict_propertytype
        True
        >>>
        >>> Es(AsDictStatic).asdict_propertytype_sub
        False
        >>> Es(s).asdict_propertytype
        False
    """
    __subclasshook__ = classmethod(
        lambda cls, C: cls is AsDictPropertyType and 'asdict' in C.__dict__ and Es(C.__dict__['asdict']).prop)
    asdict = property(lambda self: dict())


class AsDictStaticMethodType(metaclass=ABCMeta):
    """
    AsDict Static Method Type.

    Examples:
        >>> from rc import AsDictStaticMethodType
        >>> from rc import info
        >>>
        >>> class AsDictClass: asdict = classmethod(lambda cls, *args, **kwargs: dict())
        >>> class AsDictMethod: asdict = lambda self, *args, **kwargs: dict()
        >>> class AsDictProperty: asdict = property(lambda self: dict())
        >>> class AsDictStatic: asdict = staticmethod(lambda cls, *args, **kwargs: dict())
        >>>
        >>> c = AsDictClass()
        >>> m = AsDictMethod()
        >>> p = AsDictProperty()
        >>> s = AsDictStatic()
        >>>
        >>> Es(AsDictClass).asdict_staticmethodtype_sub
        False
        >>> Es(c).asdict_staticmethodtype
        False
        >>>
        >>> Es(AsDictMethod).asdict_staticmethodtype_sub
        False
        >>> Es(m).asdict_staticmethodtype
        False
        >>>
        >>> Es(AsDictProperty).asdict_staticmethodtype_sub
        False
        >>> Es(p).asdict_staticmethodtype
        False
        >>>
        >>> Es(AsDictStatic).asdict_staticmethodtype_sub
        True
        >>> Es(s).asdict_staticmethodtype
        True

    """
    __subclasshook__ = classmethod(
        lambda cls, C:
        cls is AsDictStaticMethodType and 'asdict' in C.__dict__ and Es(C.__dict__['asdict']).staticmethod)
    asdict = staticmethod(lambda *args, **kwargs: dict())


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


class Base:
    """
    Base Helper Class.

    Attributes:
    -----------
        __slots__: tuple[str]
            slots (default: tuple()).
        __hash_exclude__: tuple[str]
            Exclude slot attr for hash (default: tuple()).
        __ignore_attr__: tuple[str]
            Exclude instance attribute (default: tuple()).
        __ignore_copy__: tuple[Type, ...]
            True or class for repr instead of nested asdict and deepcopy. No deepcopy (default: tuple()).
        __ignore_kwarg__: tuple[str]
            Exclude attr from kwargs (default: tuple()).
        __ignore_str__: tuple[Type, ...]
            Use str value for object (default: tuple()).
        __repr_exclude__: tuple[str]
            Exclude slot attr for repr (default: tuple()).
        __repr_newline__: bool
            New line per attr in repr (default: True).
        __repr_pproperty__: bool
            Include :class:`pproperty` in repr (default: True).

    Methods:
    --------
        __getattribute__(item, default=None)
            :class:`function`:  Sets ``None`` as default value is attr is not defined.
        __hash__(self)
            :class:`function`: hash
        __repr__(self)
            :class:`function`: repr
        get(cls, name, default=None)
            :class:`function`: Get attribute value.
        info(self, key=Attr.PRIVATE)
            :class:`function`: :class:`info`

    Examples:
    ---------
        >>> from rich import pretty
        >>> from rc import Base
        >>> from rc import Cls
        >>> from rc import pproperty
        >>>
        >>> pretty.install()
        >>> class Test(Base):
        ...     classvar: ClassVar[int] = 1
        ...     initvar: InitVar[int] = 1
        ...     __slots__ = ('_hash', '_prop', '_repr', '_slot', )
        ...     __hash_exclude__ = ('_slot', )
        ...     __repr_exclude__ = ('_repr', )
        ...     prop = Cls.propnew('prop')
        ...     async def method_async(self):
        ...         pass
        ...     @classmethod
        ...     def clsmethod(cls):
        ...         pass
        ...     @staticmethod
        ...     def static(cls):
        ...         pass
        ...     @pproperty
        ...     def pprop(self):
        ...         return 'pprop'
        ...     @pproperty
        ...     async def pprop_async(self):
        ...         return 'pprop_async'
        >>>
        >>> test = Test()
        >>>
        >>> sorted(Mro.hash_exclude.val(test))
        ['_slot']
        >>> sorted(Mro.ignore_attr.val(test))
        []
        >>> Mro.ignore_copy.val(test).difference(IGNORE_COPY)
        set()
        >>> sorted(Mro.ignore_kwarg.val(test))
        []
        >>> Mro.ignore_str.val(test).difference(IGNORE_STR)
        set()
        >>> sorted(Mro.repr_exclude.val(test))
        ['_repr']
        >>> sorted(Mro.slots.val(test))
        ['_hash', '_prop', '_repr', '_slot']
        >>> Mro.repr_newline.first(test)
        True
        >>> Mro.repr_pproperty.first(test)
        True
        >>> Mro.slot(test, '_hash')
        True
        >>>
        >>> test.info().cls.name
        'Test'
        >>> repr(test)  # doctest: +ELLIPSIS
        'Test(_hash: None,\\n_prop: None,\\n_slot: None,\\npprop: pprop)'
        >>> assert test.__repr_exclude__[0] not in repr(test)
        >>> test.prop
        >>> test.prop = 2
        >>> test.prop
        2
        >>> del test.prop
        >>> test.prop
        >>> assert hash((test._hash, test._prop, test._repr)) == hash(test)
        >>> set(test.__slots__).difference(test.info().cls.data_attrs)
        set()
        >>> sorted(test.info().cls.data_attrs)
        ['_hash', '_prop', '_repr', '_slot', 'classvar', 'initvar']
        >>> '__slots__' in sorted(test.info(key=Attr.ALL).cls.data_attrs)
        True
        >>> test.get.__name__ in test.info().cls.methods
        True
        >>> test.get.__name__ in test.info().cls.callables
        True
        >>> test.clsmethod.__name__ in test.info().cls.classmethods
        True
        >>> test.static.__name__ in test.info().cls.staticmethods
        True
        >>> 'prop' in test.info().cls.properties
        True
        >>> 'pprop' in test.info().cls.pproperties
        True
        >>> test.info().cls.importable_name  # doctest: +ELLIPSIS
        '....Test'
        >>> test.info().cls.importable_name.split('.')[0] == test.info().cls.modname
        True
        >>> test.info().cls.qualname
        'Test'
        >>> test.info().cls.attr_value('pprop')  # doctest: +ELLIPSIS
        <....pproperty object at 0x...>
        >>> test.info().instance.attr_value('pprop')
        'pprop'
        >>> test.info().attr_value('pprop')
        'pprop'
        >>> test.info().module  # doctest: +ELLIPSIS
        <module '...' from '/Users/jose/....py'>
        >>> assert sorted(test.info().dir) == sorted(test.info().cls.dir) == sorted(test.info().instance.dir)
        >>> sorted(test.info().cls.memberdescriptors)
        ['_hash', '_prop', '_repr', '_slot']
        >>> sorted(test.info().cls.memberdescriptors) == sorted(test.__slots__)
        True
        >>> test.info().cls.methoddescriptors  # doctest: +ELLIPSIS
        [
            '__delattr__',
            ...,
            '__subclasshook__',
            'clsmethod',
            'static'
        ]
        >>> sorted(test.info().cls.methods)
        ['get', 'info', 'method_async']
        >>> sorted(test.info().cls().methods)  # doctest: +ELLIPSIS
        [
            '__delattr__',
            ...,
            '__str__',
            'get',
            'info',
            'method_async'
        ]
        >>> test.info().cls.callables
        ['clsmethod', 'get', 'info', 'method_async', 'static']
        >>> test.info().cls().callables  # doctest: +ELLIPSIS
        [
            '__delattr__',
            '__dir__',
            ...,
            '__str__',
            '__subclasshook__',
            'clsmethod',
            'get',
            'info',
            'method_async',
            'static'
        ]
        >>> sorted(test.info().cls().callables) == \
        sorted(test.info().cls().methods + test.info().cls().classmethods + test.info().cls().staticmethods)
        True
        >>> test.info().cls.setters
        ['prop']
        >>> test.info().cls.deleters
        ['prop']
        >>> test.info().cls.is_attr('_hash')
        True
        >>> test.info().cls.is_callable('prop')
        False
        >>> test.info().cls.is_classmethod('clsmethod')
        True
        >>> test.info().cls.is_data('_hash')
        True
        >>> test.info().cls.is_deleter('prop')
        True
        >>> test.info().cls.is_memberdescriptor('_hash')
        True
        >>> test.info().cls.is_method('__repr__')
        True
        >>> test.info().cls.is_methoddescriptor('__repr__')
        False
        >>> test.info().cls.is_methoddescriptor('__str__')
        True
        >>> test.info().cls.is_pproperty('pprop')
        True
        >>> test.info().cls.is_property('prop')
        True
        >>> test.info().cls.is_routine('prop')
        False
        >>> test.info().cls.is_setter('prop')
        True
        >>> test.info().cls.is_staticmethod('static')
        True
        >>> test.info().cls.is_attr('classvar')
        True
        >>> test.info().instance.is_attr('classvar')
        True
        >>> test.info().is_attr('classvar')
        True
        >>> sorted(test.info().cls.coros)
        ['method_async']
        >>> test.info().cls.is_coro('method_async')
        True
        >>> sorted(test.info().instance.coros)
        ['method_async', 'pprop_async']
        >>> test.info().instance.coros_pproperty
        ['pprop_async']
        >>> test.info().instance.coros_prop
        ['pprop_async']
        >>> test.info().instance.is_coro('pprop_async')
        True
        >>> test.info().instance.is_coro('method_async')
        True
        >>> test.info().instance.is_coro_pproperty('pprop_async')
        True
        >>> test.info().instance.is_coro_prop('pprop_async')
        True
        >>> sorted(test.info().coros)
        ['method_async', 'pprop_async']
        >>> test.info().coros_pproperty
        ['pprop_async']
        >>> test.info().coros_prop
        ['pprop_async']
        >>> test.info().is_coro('pprop_async')
        True
        >>> test.info().is_coro('method_async')
        True
        >>> test.info().is_coro_pproperty('pprop_async')
        True
        >>> test.info().is_coro_prop('pprop_async')
        True
        >>> test.info().cls.classvars
        ['classvar']
        >>> test.info().cls.initvars
        ['initvar']
        >>> test.info().cls.is_classvar('classvar')
        True
        >>> test.info().cls.is_initvar('initvar')
        True
    """
    __slots__ = tuple()
    __hash_exclude__ = tuple()
    """Exclude slot attr for hash (default: tuple())."""
    __ignore_attr__ = tuple()
    """Exclude instance attribute (default: tuple())."""
    __ignore_copy__ = tuple()
    """True or class for repr instead of nested asdict and deepcopy. No deepcopy (default: tuple())"""
    __ignore_kwarg__ = tuple()
    """Exclude attr from kwargs (default: tuple())."""
    __ignore_str__ = tuple()
    """Use str value for object (default: tuple())."""
    __repr_exclude__ = tuple()
    """Exclude slot attr for repr (default: tuple())."""
    __repr_newline__ = True
    """New line per attr in repr (default: True)."""
    __repr_pproperty__ = True
    """Include :class:`pproperty` in repr (default: True)."""

    def __getattribute__(self, name, default=None):
        """
        Sets attribute:
            - with None if it does not exists and involked with getattr()
                # getattr does not pass default to __getattribute__
            - with default if called directly.

        Examples:
            >>> from rc import Base
            >>> class Dict(Base): pass
            >>> class Slots(Base): __slots__ = ('a', )
            >>>
            >>> d = Dict()
            >>> s = Slots()
            >>> getattr(d, 'a')
            >>> d.a
            >>> getattr(s, 'a')
            >>> s.a
            >>>
            >>> d = Dict()
            >>> s = Slots()
            >>> getattr(d, 'a', 2)  # getattr does not pass default to __getattribute__
            >>> d.a
            >>> getattr(s, 'a', 2)  # getattr does not pass default to __getattribute__
            >>> s.a
            >>>
            >>> d = Dict()
            >>> s = Slots()
            >>> d.__getattribute__('a', 2)
            2
            >>> d.a
            2
            >>> s.__getattribute__('a', 2)
            2
            >>> s.a
            2
            >>>
            >>> class Dict(Base): a = 1
            >>> class Slots(Base):
            ...     __slots__ = ('a', )
            ...     def __init__(self):
            ...         self.a = 1
            >>> d = Dict()
            >>> s = Slots()
            >>> getattr(d, 'a')
            1
            >>> getattr(s, 'a')
            1
            >>> getattr(d, 'a', 2)
            1
            >>> getattr(s, 'a', 2)
            1

        Args:
            name: attr name.
            default: default value (default: None).

        Returns:
            Attribute value or sets default value and returns.
        """
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            object.__setattr__(self, name, default)
            return object.__getattribute__(self, name)

    def __hash__(self): return self.info().instance.hash
    def __repr__(self): return self.info().instance.repr

    def get(self, name, default=None):
        """
        Sets attribute:
            - with None if it does not exists and involked with getattr()
                # getattr does not pass default to __getattribute__
            - with default if called directly.

        Examples:
            >>> from rc import Base
            >>> class Dict(Base): pass
            >>> class Slots(Base): __slots__ = ('a', )
            >>>
            >>> d = Dict()
            >>> s = Slots()
            >>> d.get('a')
            >>> d.a
            >>> s.get('a')
            >>> s.a
            >>>
            >>> d = Dict()
            >>> s = Slots()
            >>> getattr(d, 'a', 2)  # getattr does not pass default to __getattribute__
            >>> d.a
            >>> getattr(s, 'a', 2)  # getattr does not pass default to __getattribute__
            >>> s.a
            >>>
            >>> d = Dict()
            >>> s = Slots()
            >>> d.get('a', 2)
            2
            >>> d.a
            2
            >>> s.get('a', 2)
            2
            >>> s.a
            2
            >>>
            >>> class Dict(Base): a = 1
            >>> class Slots(Base):
            ...     __slots__ = ('a', )
            ...     def __init__(self):
            ...         self.a = 1
            >>> d = Dict()
            >>> s = Slots()
            >>> d.get('a')
            1
            >>> s.get('a')
            1
            >>> d.get('a', 2)
            1
            >>> s.get('a', 2)
            1
            >>>
            >>> class Dict(Base, dict): pass
            >>> d = Dict()
            >>> d.get('a', 2)
            2
            >>> d.a  # dict not super().__init__(dict(a=2)

        Args:
            name: attr name.
            default: default value (default: None).

        Returns:
            Attribute value or sets default value and returns.
        """
        if hasattr(self, '__getitem__'):
            if self.__getitem__ is not None:
                try:
                    rv = self.__getitem__(name)
                except KeyError:
                    self[name] = default
                    rv = self.__getitem__(name)
                return rv
        return getset(self, name, default)

    def info(self, depth=None, ignore=False, key=Attr.PRIVATE): return info(self, depth=depth, ignore=ignore, key=key)


class BoxKeys(Box):
    """
    Creates a Box with values from keys.
    """
    def __init__(self, keys, lower=False):
        """
        Creates Box instance.

        Examples:
            >>> from rc import BoxKeys
            >>>
            >>> BoxKeys('a b')
            <Box: {'a': 'a', 'b': 'b'}>
            >>> BoxKeys('A B', lower=True)
            <Box: {'A': 'a', 'B': 'b'}>

        Args:
            keys: keys to use for keys and values.
            lower: lower the keys for values.

        Returns:
            Box:
        """
        super().__init__({item: item.lower() if lower else item for item in to_iter(keys)})


class ChainRV(Enum):
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


# noinspection PyDataclass
class Cls:
    """
    Class Helper Class.

    Attributes:
    -----------
    __slots__: tuple
        slots (default: tuple()).
    _data: Type
        Class to provide information (default: None)
    _ignore: bool
        ignore properties and kwargs :class:`Base.__ignore_kwargs__` (default: False)
    _key: :class:`rc.Key`
        keys to include (default: :attr:`rc.Key.PRIVATE`)

    """
    __slots__ = ('_data', '_ignore', '_key', )

    def __init__(self, data, ignore=False, key=Attr.PRIVATE):
        self.data = data
        self.ignore = ignore
        self.key = key

    def __call__(self, ignore=False, key=Attr.ALL):
        self.key = key
        self.ignore = ignore
        return self

    @cache
    def annotations(self, stack=2):
        return annotations(self.data, stack=stack)

    @cache
    def attr_value(self, name, default=None): return getattr(self.data, name, default)

    @property
    @cache
    def attrs(self):
        attrs = {item for item in self.dir if
                 self._include_attr(item) and item in self.data_attrs and item}
        if self.es().datatype_sub:
            _ = {attrs.add(item.name) for item in fields(self.data) if self._include_attr(item.name)}
        return sorted(list(attrs))

    @property
    @cache
    def by_kind(self): return bucket(self.classified, key=lambda x: x.kind if self.key.include(x.name) else 'e')

    @property
    @cache
    def by_name(self): return {i.name: i for i in self.classified if self.key.include(i.name)}

    @property
    @cache
    def callables(self): return sorted(self.classmethods + self.methods + self.staticmethods)

    @property
    @cache
    def classified(self): return classify_class_attrs(self.data)

    @property
    @cache
    def classmethods(self): return list(map(Name.name_.getter, self.by_kind['class method']))

    @property
    @cache
    def classvars(self): return [key for key, value in self.annotations(stack=3).items() if value.classvar]

    @property
    @cache
    def coros(self): return [i.name for i in self.classified if Es(i.object).coro]

    data = property(
            lambda self: object.__getattribute__(self, '_data'),
            lambda self, value: object.__setattr__(self, '_data', value if isinstance(value, type) else type(value)),
            lambda self: object.__setattr__(self, '_data', None)
        )

    @property
    @cache
    def data_attrs(self): return list(map(Name.name_.getter, self.by_kind['data']))

    @property
    @cache
    def defaults(self) -> dict:
        """Class defaults."""
        def is_missing(default: str) -> bool:
            return isinstance(default, MISSING_TYPE)

        rv = dict()
        rv_data = dict()
        attrs = self.attrs
        if self.es().datatype_sub:
            rv_data = {f.name: f.default if is_missing(
                f.default) and is_missing(f.default_factory) else f.default if is_missing(
                f.default_factory) else f.default_factory() for f in fields(self.data) if f.name in attrs}
        if self.es().namedtype_sub:
            rv = self.data._field_defaults
        elif self.es().dicttype_sub or self.es().slotstype_sub:
            rv = {key: inc[1] for key in attrs if (inc := self.include(key, self.data)) is not None}
        return rv | rv_data

    @property
    @cache
    def deleters(self): return [i.name for i in self.classified if Es(i.object).deleter]

    @property
    @cache
    def dir(self): return [i for i in dir(self.data) if self.key.include(i)]

    @cache
    def es(self, data=None): return Es(data or self.data)
    @cache
    def has_attr(self, name): return hasattr(self.data, name)
    @cache
    def has_method(self, name): return has_method(self.data, name)

    @property
    @cache
    def has_reduce(self): return has_reduce(self.data)

    ignore = property(
            lambda self: object.__getattribute__(self, '_ignore'),
            lambda self, value: object.__setattr__(self, '_ignore', value),
            lambda self: object.__setattr__(self, '_ignore', False)
        )

    @property
    @cache
    def importable_name(self): return importable_name(self.data)

    @cache
    def _include_attr(self, name, exclude=tuple()):
        ignore = {*Mro.ignore_attr.val(self.data), *(Mro.ignore_kwarg.val(self.data) if self.ignore else set()),
                  *exclude, *self.initvars}
        return not any([not self.key.include(name), name in ignore, f'_{self.name}' in name])

    def _include_exclude(self, data, key=True):
        import typing
        i = info(data)
        call = (Environs, )
        return any([i.module == typing, i.module == _abc, i.es().moduletype,
                    False if i.cls.data in call else i.es().callable, i.es().type,
                    self.key.include(data) if key else False])

    def include(self, key=None, data=None):
        es = Es(data)
        if (not es.mm and Cls(data).is_memberdescriptor(key) and key not in Mro.ignore_attr.val(data)) \
                or not self._include_exclude(key):
            if not es.none:
                if (value := get(self.data, key)) and self._include_exclude(value, key=False):
                    return None
                return key, value
            return key, key
        return None

    @property
    @cache
    def initvars(self): return [key for key, value in self.annotations(stack=3).items() if value.initvar]
    @cache
    def is_attr(self, name): return name in self().dir
    @cache
    def is_callable(self, name): return name in self().callables
    @cache
    def is_classmethod(self, name): return name in self().classmethods
    @cache
    def is_classvar(self, name): return name in self().classvars
    @cache
    def is_coro(self, name): return name in self().coros
    @cache
    def is_data(self, name): return name in self().data_attrs
    @cache
    def is_deleter(self, name): return name in self().deleters
    @cache
    def is_initvar(self, name): return name in self().initvars
    @cache
    def is_memberdescriptor(self, name): return name in self().memberdescriptors
    @cache
    def is_method(self, name): return name in self().methods
    @cache
    def is_methoddescriptor(self, name): return name in self().methoddescriptors
    @cache
    def is_pproperty(self, name): return name in self().pproperties
    @cache
    def is_property(self, name): return name in self().properties
    @cache
    def is_routine(self, name): return name in self().routines
    @cache
    def is_setter(self, name): return name in self().setters
    @cache
    def is_staticmethod(self, name): return name in self().staticmethods

    key = property(
            lambda self: object.__getattribute__(self, '_key'),
            lambda self, value: object.__setattr__(self, '_key', value),
            lambda self: object.__setattr__(self, '_key', Attr.PRIVATE)
        )

    @property
    @cache
    def memberdescriptors(self): return [i.name for i in self.classified if Es(i.object).memberdescriptor]

    @property
    @cache
    def methoddescriptors(self):
        """
        Includes classmethod, staticmethod and methods but not functions defined (i.e: def info(self))

        Returns:
            Method descriptors.
        """
        return [i.name for i in self.classified if Es(i.object).methoddescriptor]

    @property
    @cache
    def methods(self): return list(map(Name.name_.getter, self.by_kind['method']))

    @property
    @cache
    def modname(self): return Name._module0.get(self.data, default=str())

    @property
    @cache
    def mro(self): return self.data.__mro__

    @property
    @cache
    def name(self): return self.data.__name__

    @property
    @cache
    def pproperties(self): return [i.name for i in self.classified if Es(i.object).pproperty]

    @property
    @cache
    def properties(self): return list(map(Name.name_.getter, self.by_kind['property']))

    @staticmethod
    def propnew(name, default=None):
        return property(
            lambda self:
            self.__getattribute__(f'_{name}', default=default(self) if isinstance(default, partial) else default),
            lambda self, value: self.__setattr__(f'_{name}', value),
            lambda self: self.__delattr__(f'_{name}')
        )

    @property
    @cache
    def qualname(self): return Name._qualname0.get(self.data, default=str())

    @property
    @cache
    def routines(self): return [i.name for i in self.classified if Es(i.object).routine]

    @property
    @cache
    def setters(self): return [i.name for i in self.classified if Es(i.object).setter]

    @property
    @cache
    def staticmethods(self): return list(map(Name.name_.getter, self.by_kind['static method']))


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


class DataType(metaclass=ABCMeta):
    """
    Data Type.

    Examples:
        >>> from dataclasses import field
        >>> from dataclasses import make_dataclass
        >>>
        >>> Data = make_dataclass('C', [('a', int, field(default=1))])
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
    __subclasshook__ = classmethod(
        lambda cls, C: cls is DataType and '__annotations__' in C.__dict__ and DATACLASS_FIELDS in C.__dict__)


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
    __subclasshook__ = classmethod(lambda cls, C: cls is DictType and '__dict__' in C.__dict__)


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


EnumDictType = Alias(EnumDict, 1, name=EnumDict.__name__)


class Es:
    """
    Is Instance, Subclass, etc. Helper Class

    Examples:
        >>> from rc import Es
        >>> es = Es(2)
        >>> es.int
        True
        >>> es.bool
        False
        >>> es.instance(dict, tuple)
        False
        >>> es(dict, tuple)
        False

    Attributes:
    -----------
    data: Any
        object to provide information (default: None)
    """
    __slots__ = ('data', )
    def __init__(self, data=None): self.data = data

    def __call__(self, *args):
        """
        Call alias of :meth:`Es.instance`

        Examples:
            >>> from rc import Es
            >>> es = Es(2)
            >>> es.int
            True
            >>> es.bool
            False
            >>> es.instance(dict, tuple)
            False
            >>> es(dict, tuple)
            False

        Args:
            *args: type or tuple of types

        Returns:
            Is instance of args
        """
        return self.instance(*args)

    asyncfor = property(lambda self: isinstance(self.data, AsyncFor))
    asyncfunctiondef = property(lambda self: isinstance(self.data, AsyncFunctionDef))
    asyncwith = property(lambda self: isinstance(self.data, AsyncWith))
    annotation = property(lambda self: isinstance(self.data, Annotation))
    annotationstype = property(lambda self: isinstance(self.data, AnnotationsType))
    annotationstype_sub = property(lambda self: issubclass(self.data, AnnotationsType))
    asdict_classmethodtype = property(lambda self: isinstance(self.data, AsDictClassMethodType))
    asdict_classmethodtype_sub = property(lambda self: issubclass(self.data, AsDictClassMethodType))
    asdict_methodtype = property(lambda self: isinstance(self.data, AsDictMethodType))
    asdict_methodtype_sub = property(lambda self: issubclass(self.data, AsDictMethodType))
    asdict_propertytype = property(lambda self: isinstance(self.data, AsDictPropertyType))
    asdict_propertytype_sub = property(lambda self: issubclass(self.data, AsDictPropertyType))
    asdict_staticmethodtype = property(lambda self: isinstance(self.data, AsDictStaticMethodType))
    asdict_staticmethodtype_sub = property(lambda self: issubclass(self.data, AsDictStaticMethodType))
    ast = property(lambda self: isinstance(self.data, AST))
    asyncgen = property(lambda self: isasyncgen(self.data))
    asyncgenfunc = property(lambda self: isasyncgenfunction(self.data))
    attr = lambda self, name: name in self.get_dir
    await_ast = property(lambda self: isinstance(self.data, Await))
    awaitable = property(lambda self: isawaitable(self.data))
    bool = property(lambda self: isinstance(self.data, int) and isinstance(self.data, bool))
    builtinfunctiontype = property(lambda self: isinstance(self.data, BuiltinFunctionType))
    binaryio = property(lambda self: isinstance(self.data, BinaryIO))
    callable = property(lambda self: isinstance(self.data, Callable))
    chain = property(lambda self: isinstance(self.data, Chain))
    chainmap = property(lambda self: isinstance(self.data, ChainMap))
    classdef = property(lambda self: isinstance(self.data, ClassDef))
    classmethod = property(lambda self: isinstance(self.data, classmethod))
    codetype = property(lambda self: isinstance(self.data, CodeType))
    collections = property(lambda self: is_collections(self.data))
    coro = property(
        lambda self: any([self.asyncgen, self.asyncgenfunc, self.awaitable, self.coroutine, self.coroutinefunc]))
    coroutine = property(lambda self: iscoroutine(self.data))
    coroutinefunc = property(lambda self: iscoroutinefunction(self.data))
    dataclass = property(lambda self: is_dataclass(self.data))
    dataclass_sub = property(lambda self: is_dataclass(self.data))
    datatype = property(lambda self: isinstance(self.data, DataType))
    datatype_sub = property(lambda self: issubclass(self.data, DataType))
    defaultdict = property(lambda self: isinstance(self.data, defaultdict))
    deleter = property(lambda self: self.prop and self.data.fdel is not None)
    dict = property(lambda self: isinstance(self.data, dict))
    dicttype = property(lambda self: isinstance(self.data, DictType))
    dicttype_sub = property(lambda self: issubclass(self.data, DictType))
    dlst = property(lambda self: isinstance(self.data, (dict, list, set, tuple)))
    enum = property(lambda self: isinstance(self.data, Enum))
    enum_sub = property(lambda self: issubclass(self.data, Enum))
    enumdict = property(lambda self: isinstance(self.data, EnumDict))
    enumdict_sub = property(lambda self: issubclass(self.data, EnumDict))
    even: property(lambda self: not self.data % 2)
    float = property(lambda self: isinstance(self.data, float))
    frameinfo = property(lambda self: isinstance(self.data, FrameInfo))
    frametype = property(lambda self: isinstance(self.data, FrameType))
    functiondef = property(lambda self: isinstance(self.data, FunctionDef))
    functiontype = property(lambda self: isinstance(self.data, FunctionType))
    generator = property(lambda self: isinstance(self.data, Generator))
    getattrnobuiltintype = property(lambda self: isinstance(self.data, GetAttrNoBuiltinType))
    getattrnobuiltintype_sub = property(lambda self: issubclass(self.data, GetAttrNoBuiltinType))
    getattrtype = property(lambda self: isinstance(self.data, GetAttrType))
    getattrtype_sub = property(lambda self: issubclass(self.data, GetAttrType))
    gettype = property(lambda self: isinstance(self.data, GetType))
    gettype_sub = property(lambda self: issubclass(self.data, GetType))
    getsetdescriptor = lambda self, n: isgetsetdescriptor(self.cls_attr_value(n)) if n else self.data
    hashable = property(lambda self: bool(noexception(TypeError, hash, self.data)))
    import_ast = property(lambda self: isinstance(self.data, Import))
    importfrom = property(lambda self: isinstance(self.data, ImportFrom))
    initvar = property(lambda self: isinstance(self.data, InitVar))
    installed = property(lambda self: is_installed(self.data))
    instance = lambda self, *args: isinstance(self.data, args)
    int = property(lambda self: isinstance(self.data, int))
    io = property(lambda self: isinstance(self.data, IO))
    iterable = property(lambda self: isinstance(self.data, Iterable))
    iterator = property(lambda self: isinstance(self.data, Iterator))
    lambdatype = property(lambda self: isinstance(self.data, LambdaType))
    list = property(lambda self: isinstance(self.data, list))
    lst = property(lambda self: isinstance(self.data, (list, set, tuple)))
    memberdescriptor = property(lambda self: ismemberdescriptor(self.data))
    method = property(
        lambda self:
        callable(self.data) and not type(self)(self.data).instance(classmethod, property, property, staticmethod))
    methoddescriptor = property(lambda self: ismethoddescriptor(self.data))
    methodwrappertype = property(lambda self: isinstance(self.data, MethodWrapperType))
    methodwrappertype_sub = property(lambda self: issubclass(self.data, MethodWrapperType))
    mlst = property(lambda self: isinstance(self.data, (MutableMapping, list, set, tuple)))
    mm = property(lambda self: isinstance(self.data, MutableMapping))
    moduletype = property(lambda self: isinstance(self.data, ModuleType))
    module_function = property(lambda self: is_module_function(self.data))
    noncomplex = property(lambda self: is_noncomplex(self.data))
    namedtype = property(lambda self: isinstance(self.data, NamedType))
    namedtype_sub = property(lambda self: issubclass(self.data, NamedType))
    named_annotationstype = property(lambda self: isinstance(self.data, NamedAnnotationsType))
    named_annotationstype_sub = property(lambda self: issubclass(self.data, NamedAnnotationsType))
    none = property(lambda self: isinstance(self.data, type(None)))
    object = property(lambda self: is_object(self.data))
    pathlib = property(lambda self: isinstance(self.data, PathLib))
    picklable = lambda self, name: is_picklable(name, self.data)
    primitive = property(lambda self: is_primitive(self.data))
    prop = property(lambda self: isinstance(self.data, property))
    pproperty = property(lambda self: isinstance(self.data, pproperty))
    reducible = property(lambda self: is_reducible(self.data))
    reducible_sequence_subclass = property(lambda self: is_reducible_sequence_subclass(self.data))
    routine = property(lambda self: isroutine(self.data))
    sequence = property(lambda self: is_sequence(self.data))
    sequence_subclass = property(lambda self: is_sequence_subclass(self.data))
    set = property(lambda self: isinstance(self.data, set))
    setter = property(lambda self: self.prop and self.data.fset is not None)
    slotstype = property(lambda self: isinstance(self.data, SlotsType))
    slotstype_sub = property(lambda self: issubclass(self.data, SlotsType))
    staticmethod = property(lambda self: isinstance(self.data, staticmethod))
    str = property(lambda self: isinstance(self.data, str))
    subclass = lambda self, *args: issubclass(self.data, args) if self.type else issubclass(type(self.data), args)
    textio = property(lambda self: isinstance(self.data, TextIO))
    tracebacktype = property(lambda self: isinstance(self.data, TracebackType))
    tuple = property(lambda self: isinstance(self.data, tuple))
    type = property(lambda self: isinstance(self.data, type))
    unicode = property(lambda self: is_unicode(self.data))


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


class GetAttrNoBuiltinType(metaclass=ABCMeta):
    """
    Get Attr Type (Everything but builtins, except: object and errors)

    Examples:
        >>> from collections import namedtuple
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
    __getattribute__ = lambda self, n: object.__getattribute__(self, n)

    @classmethod
    def __subclasshook__(cls, C):
        return cls is GetAttrNoBuiltinType and any(['_field_defaults' in C.__dict__,
                                                    not allin(C.__mro__, BUILTIN_CLASSES) and 'get' not in C.__dict__ or
                                                    ('get' in C.__dict__ and not callable(C.__dict__['get']))])


class GetAttrType(metaclass=ABCMeta):
    """
    Get Attr Type (Everything but GetType)

    Examples:
        >>> from collections import namedtuple
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
    __getattribute__ = lambda self, n: object.__getattribute__(self, n)

    @classmethod
    def __subclasshook__(cls, C):
        return cls is GetAttrType and 'get' not in C.__dict__ or \
               ('get' in C.__dict__ and not callable(C.__dict__['get']))


@runtime_checkable
class GetSupport(Protocol):
    """An ABC with one abstract method get."""
    __slots__ = tuple()

    @abstractmethod
    def get(self, name, default=None):
        return self, name, default


class GetType(metaclass=ABCMeta):
    """
    Dict Type.

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
    get = lambda self, name, default=None: getattr(self, name, default)
    __subclasshook__ = classmethod(
        lambda cls, C: cls is GetType and 'get' in C.__dict__ and callable(C.__dict__['get']))


class Instance:
    """
    Instance Helper Class

    Attributes:
    -----------
    __slots__: tuple
        slots (default: tuple()).
    _data: Any
        object to provide instance information (default: None)
    _ignore: bool
        ignore properties and kwargs :class:`info.__ignore_kwargs__` (default: False)
    _key: :class:`Key`
        keys to include (default: :attr:`rc.Key.PRIVATE`)
    cls: :class:`Cls`
        :class:`Cls` (default: Cls(_data, _ignore, _key))
    es: :class:`Es`
        :class:`Es` (default: Es(_data))
    """
    __slots__ = ('_data', '_ignore', '_key', )

    def __init__(self, data, ignore=False, key=Attr.PRIVATE):
        self.data = data
        self.ignore = ignore
        self.key = key

    def __call__(self, ignore=False, key=Attr.ALL):
        self.ignore = ignore
        self.key = key
        return self

    @cache
    def annotations(self, stack=2):
        return annotations(self.data, stack=stack)

    @cache
    def attr_value(self, name, default=None): return getattr(self.data, name, default)

    @property
    @cache
    def cls(self): return Cls(data=self.data, ignore=self.ignore, key=self.key)

    @property
    @cache
    def coros(self): return self.cls.coros + self.coros_prop

    @property
    @cache
    @runwarning
    def coros_pproperty(self): return [i.name for i in self.cls.classified if Es(i.object).pproperty and
                                       Es(object.__getattribute__(self.data, i.name)).coro]

    @property
    @cache
    @runwarning
    def coros_prop(self): return [i.name for i in self.cls.classified if Es(i.object).prop and
                                  Es(object.__getattribute__(self.data, i.name)).coro]

    data = property(
            lambda self: object.__getattribute__(self, '_data'),
            lambda self, value: object.__setattr__(self, '_data', value),
            lambda self: object.__setattr__(self, '_data', None)
        )

    @property
    def dir(self): return [i for i in dir(self.data) if self.key.include(i)]

    def es(self, data=None): return Es(data or self.data)

    def has_attr(self, name): return hasattr(self.data, name)

    def has_method(self, name): return has_method(self.data, name)

    @property
    def has_reduce(self): return has_reduce(self.data)

    @property
    def hash(self):
        return hash(tuple(map(lambda x: getset(self.data, x), Mro.hash_exclude.slotsinclude(self.data))))

    ignore = property(
            lambda self: object.__getattribute__(self, '_ignore'),
            lambda self, value: object.__setattr__(self, '_ignore', value),
            lambda self: object.__setattr__(self, '_ignore', False)
        )

    def is_attr(self, name): return name in self().dir
    def is_coro(self, name): return name in self().coros
    def is_coro_pproperty(self, name): return name in self().coros_pproperty
    def is_coro_prop(self, name): return name in self().coros_prop

    key = property(
            lambda self: object.__getattribute__(self, '_key'),
            lambda self, value: object.__setattr__(self, '_key', value),
            lambda self: object.__setattr__(self, '_key', Attr.PRIVATE)
        )

    @property
    def repr(self):
        attrs = Mro.repr_exclude.slotsinclude(self.data)
        attrs.update(self.cls.pproperties if Mro.repr_pproperty.first(self.data) else list())
        r = [f"{s}: {getset(self.data, s)}" for s in sorted(attrs) if s and not self.is_coro(s)]
        new = f',{NEWLINE if Mro.repr_newline.first(self.data) else " "}'
        return f'{self.cls.name}({new.join(r)})'


class Mro(Enum):
    """MRO Helper Calls"""
    hash_exclude = auto()
    ignore_attr = auto()
    ignore_copy = auto()
    ignore_kwarg = auto()
    ignore_str = auto()
    repr_exclude = auto()
    repr_newline = auto()
    repr_pproperty = auto()

    slots = auto()

    @classmethod
    @cache
    def asdict(cls):
        """
        Get map_reduce (dict) attrs lists converted to real names.

        Examples:
            >>> from rich import pretty
            >>> from rc import Mro
            >>>
            >>> pretty.install()
            >>>
            >>> Mro.asdict()  # doctest: +ELLIPSIS
            {
                ...,
                'ignore_copy': '__ignore_copy__',
                ...
            }

        Returns:
            Dict with name and real name.
        """
        return {key: value.real for key, value in cls.__members__.items()}

    @classmethod
    @cache
    def attrs(cls):
        """
        Get attrs tuple with private converted to real names.

        Examples:
            >>> from rich import pretty
            >>> from rc import Mro
            >>>
            >>> pretty.install()
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
        return tuple(cls.asdict())

    @classmethod
    def cls(cls, obj):
        """
        Object Class MRO.

        Examples:
            >>> from rich import pretty
            >>> from rc import Mro
            >>>
            >>> pretty.install()
            >>>
            >>> Mro.cls(dict(a=1))
            (<class 'dict'>, <class 'object'>)

        Args:
            obj: object.

        Returns:
            Object Class MRO.
        """
        return obj.__mro__ if isinstance(obj, type) else type(obj).__mro__

    def first(self, obj):
        """
        First value of attr found in mro and instance if obj is instance.

        Examples:
            >>> from rich import pretty
            >>> from rc import Mro
            >>>
            >>> pretty.install()
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
                return object.__getattribute__(item, self.real)

    def has(self, obj):
        """
        Checks if Object has attr.

        Examples:
            >>> from rich import pretty
            >>> from rc import Mro
            >>>
            >>> pretty.install()
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
        return hasattr(obj, self.real)

    @classmethod
    def obj(cls, obj):
        """
        Object and Class MRO tuple.

        Examples:
            >>> from rich import pretty
            >>> from rc import Mro
            >>>
            >>> pretty.install()
            >>>
            >>> Mro.obj(dict(a=1))
            ({'a': 1}, <class 'dict'>, <class 'object'>)

        Args:
            obj: object.

        Returns:
            Object and Class MRO tuple.
        """
        return obj.__mro__ if isinstance(obj, type) else (obj, *type(obj).__mro__)

    @property
    def real(self):
        """
        Get real attr name converted for private and attrs that conflict with Enum.

        Examples:
            >>> from rc import Mro
            >>>
            >>> Mro.hash_exclude.real
            '__hash_exclude__'
            >>> Mro.slots.real
            '__slots__'

        Returns:
            Real attribute name converted for private and attrs that conflict with Enum.
        """
        return f'__{self.name}__'

    @classmethod
    def slot(cls, obj, name):
        """
        Is attribute in slots?

        Examples:
            >>> from rich import pretty
            >>> from rc import Mro
            >>>
            >>> pretty.install()
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
            >>> from rich import pretty
            >>> from rc import Mro
            >>>
            >>> pretty.install()
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
            >>> from rich import pretty
            >>> from rc import Mro
            >>>
            >>> pretty.install()
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
            >>> sorted(Mro.ignore_attr.val(test))
            ['attr']
            >>> Mro.ignore_copy.val(test).difference(IGNORE_COPY)
            {<class 'tuple'>}
            >>> sorted(Mro.ignore_kwarg.val(test))
            ['kwarg']
            >>> Mro.ignore_str.val(test).difference(IGNORE_STR)
            {<class 'tuple'>}
            >>> sorted(Mro.repr_exclude.val(test))
            ['_repr']
            >>> sorted(Mro.slots.val(test))
            ['_hash', '_prop', '_repr', '_slot']

        Returns:
            All/accumulated values of attr in mro and obj if instance.
        """
        return {*(value for item in self.obj(obj) for value in getattr(item, self.real, tuple())),
                *(IGNORE_COPY if self is self.__class__.ignore_copy else IGNORE_STR
                if self is self.__class__.ignore_str else tuple())}


class Name(Enum):
    _all0 = auto()
    _class0 = auto()
    _annotations0 = auto()
    _builtins0 = auto()
    _cached0 = auto()
    _code0 = auto()
    _contains0 = auto()
    _dataclass_fields0 = auto()
    _dataclass_params0 = auto()
    _delattr0 = auto()
    _dir0 = auto()
    _dict0 = auto()
    _doc0 = auto()
    _eq0 = auto()
    _file0 = auto()
    _getattribute0 = auto()
    _len0 = auto()
    _loader0 = auto()
    _members0 = auto()
    _module0 = auto()
    _mro0 = auto()
    _name0 = auto()
    _package0 = auto()
    _qualname0 = auto()
    _reduce0 = auto()
    _repr0 = auto()
    _setattr0 = auto()
    _slots0 = auto()
    _spec0 = auto()
    _str0 = auto()
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
    @cache
    def _attrs(cls):
        """
        Get map_reduce (dict) attrs lists converted to real names.

        Examples:
            >>> from rich import pretty
            >>> from rc import Name
            >>>
            >>> pretty.install()
            >>> Name._attrs().keys()
            dict_keys([True, False])
            >>> Name._attrs().values()  # doctest: +ELLIPSIS
            dict_values([['__all__', ...], ['_asdict', ...])

        Returns:
            [True]: private attrs.
            [False[: public attrs.
        """
        return map_reduce(cls.__members__, lambda x: x.endswith('0'), lambda x: cls._real(x))

    @classmethod
    @cache
    def attrs(cls):
        """
        Get attrs tuple with private converted to real names.

        Examples:
            >>> from rich import pretty
            >>> from rc import Name
            >>>
            >>> pretty.install()
            >>> Name.attrs()  # doctest: +ELLIPSIS
            (
                '__all__',
                ...,
                '_asdict',
                ...
            )

        Returns:
            Tuple of attributes wit private converted to real names.
        """
        return tuple(collapse(Name._attrs().values()))

    @singledispatchmethod
    def get(self, obj: GetType, default=None):
        """
        Get value from GetType/MutableMapping.

        Examples:
            >>> import inspect
            >>> from ast import unparse
            >>> from inspect import FrameInfo
            >>> from inspect import getmodulename
            >>> from rich import pretty
            >>> from rc import Name
            >>>
            >>> pretty.install()
            >>> f = inspect.stack()[0]
            >>> globs_locs = (f.frame.f_globals | f.frame.f_locals).copy()
            >>> Name.filename.get(f), Name.function.get(f), Name.code_context.get(f)[0], Name.source(f)
            (
                PosixPath('<doctest get[7]>'),
                '<module>',
                'f = inspect.stack()[0]\\n',
                'f = inspect.stack()[0]\\n'
            )
            >>> Name._file0.get(globs_locs)  # doctest: +ELLIPSIS
            PosixPath('/Users/jose/....py')
            >>> assert Name._file0.get(globs_locs) == PathLib(__file__) == PathLib(Name._spec0.get(globs_locs).origin)
            >>> assert Name._name0.get(globs_locs) == getmodulename(__file__) == Name._spec0.get(globs_locs).name
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
        if self is Name._file0:
            return self.path(obj)
        return obj.get(self.real, default)

    @get.register
    def get_getattrtype(self, obj: GetAttrType, default=None):
        """
        Get value of attribute from GetAttrType.

        Examples:
            >>> import inspect
            >>> from ast import unparse
            >>> from inspect import FrameInfo
            >>> from inspect import getmodulename
            >>> import rc.utils
            >>> from rich import pretty
            >>> from rc import Name
            >>>
            >>> pretty.install()
            >>> f = inspect.stack()[0]
            >>> globs_locs = (f.frame.f_globals | f.frame.f_locals).copy()
            >>> Name._file0.get(globs_locs)  # doctest: +ELLIPSIS
            PosixPath('/Users/jose/....py')
            >>> Name._file0.get(rc.utils)  # doctest: +ELLIPSIS
            PosixPath('/Users/jose/....py')
            >>> PathLib(Name._spec0.get(globs_locs).origin)  # doctest: +ELLIPSIS
            PosixPath('/Users/jose/....py')
            >>> PathLib(Name._spec0.get(rc.utils).origin)  # doctest: +ELLIPSIS
            PosixPath('/Users/jose/....py')
            >>> Name._spec0.get(rc.utils).name == rc.utils.__name__
            True
            >>> Name._spec0.get(rc.utils).name.split('.')[0] == rc.utils.__package__
            True
            >>> Name._name0.get(rc.utils) == rc.utils.__name__
            True
            >>> Name._package0.get(rc.utils) == rc.utils.__package__
            True

        Args:
            obj: object
            default: None

        Returns:
            Value from __getattribute__ method.
        """
        if self is Name._file0:
            try:
                p = object.__getattribute__(obj, Name._file0.real)
                return PathLib(p)
            except AttributeError:
                return self.path(obj)
        try:
            return object.__getattribute__(obj, self.real)
        except AttributeError:
            return default

    @get.register
    def get_frameinfo(self, obj: FrameInfo, default=None):
        """
        Get value of attribute from FrameInfo.

        Examples:
            >>> import inspect
            >>> from ast import unparse
            >>> from inspect import FrameInfo
            >>> from rc import Name
            >>>
            >>> f = inspect.stack()[0]
            >>> assert f == FrameInfo(Name.frame.get(f), str(Name.filename.get(f)), Name.lineno.get(f),\
            Name.function.get(f), Name.code_context.get(f), Name.index.get(f))
            >>> assert Name.filename.get(f) == Name._file0.get(f)
            >>> Name.source(f)
            'f = inspect.stack()[0]\\n'
            >>> unparse(Name.node(f))
            'f = inspect.stack()[0]'
            >>> unparse(Name.node('pass'))
            'pass'
            >>> assert str(Name._file0.get(f)) == str(Name.filename.get(f))
            >>> assert Name._name0.get(f) == Name.co_name.get(f) == Name.function.get(f)
            >>> assert Name.lineno.get(f) == Name.f_lineno.get(f) == Name.tb_lineno.get(f)
            >>> assert Name.vars.get(f) == (f.frame.f_globals | f.frame.f_locals).copy()
            >>> assert Name.vars.get(f) == (Name.f_globals.get(f) | Name.f_locals.get(f)).copy()
            >>> assert Name.vars.get(f) == (Name.globals.get(f) | Name.locals.get(f)).copy()
            >>> assert unparse(Name.node(f)) in Name.code_context.get(f)[0]
            >>> assert Name._spec0.get(f).origin == __file__

        Args:
            obj: object
            default: None

        Returns:
            Value from FrameInfo method
        """
        if self in [Name._file0, Name.filename]:
            return PathLib(obj.filename)
        if self in [Name._name0, Name.co_name, Name.function]:
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
        if self in [Name._code0, Name.f_code]:
            return obj.frame.f_code
        if self in [Name.f_back, Name.tb_next]:
            return obj.frame.f_back
        if self is Name.index:
            return obj.index
        if self is Name.code_context:
            return obj.code_context
        return self.get((obj.frame.f_globals | obj.frame.f_locals).copy(), default=default)

    @get.register
    @cache
    def get_frametype(self, obj: FrameType, default=None):
        """
        Get value of attribute from FrameType.

        Examples:
            >>> import inspect
            >>> from ast import unparse
            >>> from inspect import FrameInfo
            >>> from rc import Name
            >>>
            >>> frameinfo = inspect.stack()[0]
            >>> f = frameinfo.frame
            >>> assert Name.filename.get(f) == Name.filename.get(frameinfo)
            >>> assert Name.frame.get(f) == Name.frame.get(frameinfo)
            >>> assert Name.lineno.get(f) == Name.lineno.get(frameinfo)
            >>> assert Name.function.get(f) == Name.function.get(frameinfo)
            >>> assert frameinfo == FrameInfo(Name.frame.get(f), str(Name.filename.get(f)), Name.lineno.get(f),\
            Name.function.get(f), frameinfo.code_context, frameinfo.index)
            >>> assert Name.filename.get(f) == Name._file0.get(f)
            >>> Name.source(f)
            'frameinfo = inspect.stack()[0]\\n'
            >>> unparse(Name.node(f))
            'frameinfo = inspect.stack()[0]'
            >>> unparse(Name.node('pass'))
            'pass'
            >>> assert str(Name._file0.get(f)) == str(Name.filename.get(f))
            >>> assert Name._name0.get(f) == Name.co_name.get(f) == Name.function.get(f)
            >>> assert Name.lineno.get(f) == Name.f_lineno.get(f) == Name.tb_lineno.get(f)
            >>> assert Name.vars.get(f) == (f.f_globals | f.f_locals).copy()
            >>> assert Name.vars.get(f) == (Name.f_globals.get(f) | Name.f_locals.get(f)).copy()
            >>> assert Name.vars.get(f) == (Name.globals.get(f) | Name.locals.get(f)).copy()
            >>> assert unparse(Name.node(f)) in Name.code_context.get(frameinfo)[0]
            >>> assert Name._spec0.get(f).origin == __file__

        Args:
            obj: object
            default: None

        Returns:
            Value from FrameType method
        """
        if self in [Name._file0, Name.filename]:
            return self.path(obj)
        if self in [Name._name0, Name.co_name, Name.function]:
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
        if self in [Name._code0, Name.f_code]:
            return obj.f_code
        if self in [Name.f_back, Name.tb_next]:
            return obj.f_back
        return self.get((obj.f_globals | obj.f_locals).copy(), default=default)

    @get.register
    @cache
    def get_tracebacktype(self, obj: TracebackType, default=None):
        """
        Get value of attribute from TracebackType.

        Args:
            obj: object
            default: None

        Returns:
            Value from TracebackType method
        """
        if self in [Name._file0, Name.filename]:
            return self.path(obj)
        if self in [Name._name0, Name.co_name, Name.function]:
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
        if self in [Name._code0, Name.f_code]:
            return obj.tb_frame.f_code
        if self in [Name.f_back, Name.tb_next]:
            return obj.tb_next
        return self.get((obj.tb_frame.f_globals | obj.tb_frame.f_locals).copy(), default=default)

    @property
    @cache
    def getter(self):
        """
        Attr getter with real name for private and public which conflicts with Enum.

        Examples:
            >>> import rc.utils
            >>> from rc import Name
            >>>
            >>> Name._module0.getter(tuple)
            'builtins'
            >>> Name._name0.getter(tuple)
            'tuple'
            >>> Name._file0.getter(rc.utils)  # doctest: +ELLIPSIS
            '/Users/jose/....py'

        Returns:
            Attr getter with real name for private and public which conflicts with Enum.
        """
        return attrgetter(self.real)

    def has(self, obj):
        """
        Checks if has attr with real name for private and public which conflicts with Enum.

        Examples:
            >>> import rc.utils
            >>> from rc import Name
            >>>
            >>> Name._module0.has(tuple)
            True
            >>> Name._name0.has(tuple)
            True
            >>> Name._file0.has(tuple)
            False
            >>> Name._file0.has(rc.utils)
            True

        Returns:
            Checks if has attr with real name for private and public which conflicts with Enum.
        """
        return hasattr(obj, self.real)

    @classmethod
    def node(cls, obj, complete=False, line=False):
        """
        Get node of object.

        Examples:
            >>> import inspect
            >>> from ast import unparse
            >>> from inspect import FrameInfo
            >>> from inspect import getmodulename
            >>> from rich import pretty
            >>> from rc import Name
            >>>
            >>> pretty.install()
            >>> f = inspect.stack()[0]
            >>> globs_locs = (f.frame.f_globals | f.frame.f_locals).copy()
            >>> Name.filename.get(f), Name.function.get(f), Name.code_context.get(f)[0], Name.source(f) \
             # doctest: +ELLIPSIS
            (
                PosixPath('<doctest ...node[7]>'),
                '<module>',
                'f = inspect.stack()[0]\\n',
                'f = inspect.stack()[0]\\n'
            )
            >>> Name._file0.get(globs_locs)  # doctest: +ELLIPSIS
            PosixPath('/Users/jose/....py')
            >>> assert Name._file0.get(globs_locs) == PathLib(__file__) == PathLib(Name._spec0.get(globs_locs).origin)
            >>> assert Name._name0.get(globs_locs) == getmodulename(__file__) == Name._spec0.get(globs_locs).name
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
            >>> import inspect
            >>> from ast import unparse
            >>> from inspect import FrameInfo
            >>> from inspect import getmodulename
            >>> from rich import pretty
            >>> from rc import Name
            >>> from rc import allin
            >>>
            >>> pretty.install()
            >>> frameinfo = inspect.stack()[0]
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
            f = obj.get(Name._file0.real)
        elif es.frameinfo:
            f = obj.filename
        else:
            try:
                f = getsourcefile(obj) or getfile(obj)
            except TypeError:
                f = None
        return PathLib(f or str(obj))

    @classmethod
    @cache
    def private(cls):
        """
        Get private attrs tuple converted to real names.

        Examples:
            >>> from rich import pretty
            >>> from rc import Name
            >>>
            >>> pretty.install()
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
    @cache
    def public(cls):
        """
        Get public attrs tuple.

        Examples:
            >>> from rich import pretty
            >>> from rc import Name
            >>>
            >>> pretty.install()
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
    @cache
    def _real(cls, name):
        return f'_{name.replace("0", "_")}_' if name.startswith('_') and name.endswith('0') else name.removesuffix('_')

    @property
    def real(self):
        """
        Get real attr name converted for private and attrs that conflict with Enum.

        Examples:
            >>> from rc import Name
            >>> Name._file0.real
            '__file__'
            >>> Name._asdict.real
            '_asdict'
            >>> Name.cls_.real
            'cls'
            >>> Name.get_.real
            'get'
            >>> Name.name_.real
            'name'
            >>> Name.self_.real
            'self'
            >>> Name.value_.real
            'value'

        Returns:
            Real attribute name converted for private and attrs that conflict with Enum.
        """
        return self._real(self.name)

    @classmethod
    def _source(cls, obj, line=False):
        f = cls._file0.get(obj) or obj
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
            >>> import inspect
            >>> from ast import unparse
            >>> from inspect import FrameInfo
            >>> from inspect import getmodulename
            >>> import rc.utils
            >>> from rich import pretty
            >>> from rc import Name
            >>> from rc import allin
            >>>
            >>> pretty.install()
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
            >>> frameinfo = inspect.stack()[0]
            >>> Name.source(frameinfo), frameinfo.function
            ('frameinfo = inspect.stack()[0]\\n', '<module>')
            >>> Name.source(frameinfo, complete=True), frameinfo.function
            ('frameinfo = inspect.stack()[0]\\n', '<module>')
            >>>
            >>> frametype = frameinfo.frame
            >>> Name.source(frametype), frametype.f_code.co_name
            ('frameinfo = inspect.stack()[0]\\n', '<module>')
            >>> Name.source(frameinfo, complete=True), frametype.f_code.co_name
            ('frameinfo = inspect.stack()[0]\\n', '<module>')
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


class NamedType(metaclass=ABCMeta):
    """
    named Type.

    Examples:
        >>> from collections import namedtuple
        >>>
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
    __subclasshook__ = classmethod(lambda cls, C: cls is NamedType and '_field_defaults' in C.__dict__)
    _asdict = lambda self: dict()
    _fields = tuple()
    _field_defaults = dict()


class NamedAnnotationsType(metaclass=ABCMeta):
    """
    named Type.

    Examples:
        >>> from collections import namedtuple
        >>> from typing import NamedTuple
        >>>
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
    __subclasshook__ = classmethod(
        lambda cls, C:
        cls is NamedAnnotationsType and '__annotations__' in C.__dict__ and '_field_defaults' in C.__dict__)
    _asdict = lambda self: dict()
    _fields = tuple()
    _field_defaults = dict()


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
    __subclasshook__ = classmethod(lambda cls, C: cls is SlotsType and '__slots__' in C.__dict__)


Types = Union[Type[AnnotationsType], Type[AsDictClassMethodType], Type[AsDictMethodType], Type[AsDictPropertyType],
              Type[AsDictStaticMethodType], Type[DataType], Type[DictType], Type[GetAttrType],
              Type[GetAttrNoBuiltinType], Type[GetType], Type[NamedType], Type[NamedAnnotationsType],
              Type[SlotsType], Type[type]]


def aioloop(): return noexception(RuntimeError, get_running_loop)


def allin(origin, destination):
    """
    Checks all items in origin are in destination iterable.

    Examples:
        >>> from rc import allin
        >>> from rc import BUILTIN_CLASSES
        >>>
        >>> class Int(int):
        ...     pass
        >>> allin(tuple.__mro__, BUILTIN_CLASSES)
        True
        >>> allin(Int.__mro__, BUILTIN_CLASSES)
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

# TODO: Finish annotations wit defaults and kwargs and init


def annotations(obj, stack=1):
    """
    Formats obj annotations.

    Examples:
        >>> from dataclasses import dataclass
        >>> from dataclasses import InitVar
        >>> from typing import ClassVar
        >>> from rich.pretty import install
        >>> from rc import annotations
        >>> install()
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
        initvar = Es(cls).initvar  # TODO: Mirar porque el origin debe ser InitVar y entonces ...
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

    frame = stack if Es(stack).frametype else inspect.stack()[stack].frame
    rv = dict()
    if a := noexception(TypeError, get_type_hints, obj, globalns=frame.f_globals, localns=frame.f_locals):
        for name, hint in a.items():
            rv |= {name: inner(hint)}
    return dict_sort(rv)


def annotations_init(cls, stack=2, optional=True, **kwargs):
    """
    Init with defaults or kwargs an annotated class.

    Examples:
        >>> from pathlib import Path
        >>> from typing import NamedTuple
        >>>
        >>> from rc import annotations_init
        >>>
        >>> NoInitValue = NamedTuple('NoInitValue', var=str)

        >>> A = NamedTuple('A', module=str, path=Optional[Path], test=Optional[NoInitValue])
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
        >>> from rc import anyin
        >>> from rc import BUILTIN_CLASSES
        >>>
        >>> class Int(int):
        ...     pass
        >>> anyin(tuple.__mro__, BUILTIN_CLASSES)
        <class 'tuple'>
        >>> assert anyin('tuple int', BUILTIN_CLASSES) is None
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
def delete(data: MutableMapping, key=('self', 'cls', )):
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
def delete_list(data: list, key=('self', 'cls', )):
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


@singledispatch
def get(data: GetType, name, default=None):
    """
    Get value of name in Mutabble Mapping/GetType or object.

    Examples:
        >>> from rc import get
        >>>
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
        >>> from rc import get
        >>>
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


class info:
    """
    Is Instance, etc. Helper Class

    Attributes:
    -----------
    __slots__: tuple
        slots (default: tuple()).
        __ignore_attr__: tuple
        Exclude instance attribute (default: tuple()).
    __ignore_copy__: tuple
        True or class for repr instead of nested asdict and deepcopy. No deepcopy (default: tuple()).
    __ignore_kwarg__: tuple
        Exclude attr from kwargs (default: tuple()).
    __ignore_str__: tuple
        Use str value for object (default: tuple()).
    _data: Any
        object to provide information (default: None)
    _depth: Optional[int]
        recursion depth (default: None)
    _ignore: bool
        ignore properties and kwargs :class:`info.__ignore_kwargs__` (default: False)
    _key: :class:`Key`
        keys to include (default: :attr:`rc.Key.PRIVATE`)
    cls: :class:`Cls`
        :class:`Cls` (default: Cls(_data, _ignore, _ley))
    es: :class:`Es`
        :class:`Es` (default: Es(data))
    instance: :class:`Instance`
        :class:`Instance` (default: Instance(_data, _ignore, _key))
   """
    __slots__ = ('_data', '_depth', '_ignore', '_key', )
    __ignore_attr__ = tuple()
    """Exclude instance attribute (default: tuple())."""
    __ignore_copy__ = tuple()
    """True or class for repr instead of nested asdict and deepcopy. No deepcopy (default: tuple())"""
    __ignore_kwarg__ = tuple()
    """Exclude attr from kwargs (default: tuple())."""
    __ignore_str__ = tuple()
    """Use str value for object (default: tuple())."""

    def __init__(self, data, depth=None, ignore=False, key=Attr.PRIVATE):
        self.data = data
        self.depth = depth
        self.ignore = ignore
        self.key = key

    def __call__(self, depth=None, ignore=False, key=Attr.ALL):
        self.depth = depth or self.depth
        self.ignore = ignore or self.ignore
        self.key = key or self.key
        return self

    @cache
    def annotations(self, stack=2):
        """
        Object Annotations.

        Examples:
            >>> from dataclasses import dataclass
            >>> from dataclasses import InitVar
            >>> from typing import ClassVar
            >>> from rich.pretty import install
            >>> from rc import info
            >>> @dataclass
            ... class Test:
            ...     a: int = 1
            ...     initvar: InitVar[int] = 1
            ...     classvar: ClassVar[int] = 2
            ...     def __post_init__(self, initvar: int):
            ...         self.a = initvar
            >>> ann = info(Test).annotations()
            >>> ann['a'].cls, ann['a'].default
            (<class 'int'>, 0)
            >>> ann['initvar'].cls, ann['initvar'].initvar, ann['initvar'].default
            (<class 'int'>, True, 0)
            >>> ann['classvar'].cls, ann['classvar'].classvar, ann['classvar'].default
            (<class 'int'>, True, 0)

        Args:
            stack: stack index to extract globals and locals (default: 2) or frame.

        Returns:
            Object Annotations.
        """
        return annotations(self.data, stack=stack)

    @cache
    def attr_value(self, name, default=None): return getattr(self.data, name, default)

    @property
    @cache
    def cls(self): return Cls(data=self.data, ignore=self.ignore, key=self.key)

    @property
    @cache
    def coros(self): return self.instance.coros

    @property
    @cache
    def coros_pproperty(self): return self.instance.coros_pproperty

    @property
    @cache
    def coros_prop(self): return self.instance.coros_prop

    data = property(
            lambda self: object.__getattribute__(self, '_data'),
            lambda self, value: object.__setattr__(self, '_data', value),
            lambda self: object.__setattr__(self, '_data', None)
        )

    depth = property(
            lambda self: object.__getattribute__(self, '_depth'),
            lambda self, value: object.__setattr__(self, '_depth', value),
            lambda self: object.__setattr__(self, '_data', None)
        )

    @property
    def dir(self):
        return set(self.cls.dir + self.instance.dir)

    def es(self, data=None): return Es(data or self.data)

    def has_attr(self, name): return self.cls.has_attr(name=name) or self.instance.has_attr(name=name)

    def has_method(self, name): return self.cls.has_method(name=name) or self.instance.has_method(name=name)

    @property
    def has_reduce(self): return self.cls.has_reduce or self.instance.has_reduce

    ignore = property(
            lambda self: object.__getattribute__(self, '_ignore'),
            lambda self, value: object.__setattr__(self, '_ignore', value),
            lambda self: object.__setattr__(self, '_ignore', False)
        )

    @property
    def ignore_attr(self): return

    @property
    def initvars(self):
        """
        InitVars.

        Examples:
            >>> from dataclasses import dataclass
            >>> from rich.pretty import install
            >>> from rc import info


        Returns:
            Object InitVars.
        """
        return [var for var, annotation in self.annotations(index=3).items()
                if (isinstance(annotation, str) and 'InitVar' in annotation) or
                (not isinstance(annotation, str) and isinstance(annotation, InitVar))]

    @property
    @cache
    def instance(self): return Instance(data=self.data, ignore=self.ignore, key=self.key)

    def is_attr(self, name): return self.cls.is_attr(name) or self.instance.is_attr(name)
    def is_coro(self, name): return name in self().coros
    def is_coro_pproperty(self, name): return name in self().coros_pproperty
    def is_coro_prop(self, name): return name in self().coros_prop

    key = property(
            lambda self: object.__getattribute__(self, '_key'),
            lambda self, value: object.__setattr__(self, '_key', value),
            lambda self: object.__setattr__(self, '_key', Attr.PRIVATE)
        )

    @property
    @cache
    def module(self): return getmodule(self.data)


def is_even(number): return Es(number).even


def join_newline(data): return NEWLINE.join(data)


def map_reduce_even(iterable): return map_reduce(iterable, keyfunc=is_even)


def map_with_args(data, func, /, *args, pred=lambda x: True if x else False, split=' ', **kwargs):
    """
    Apply pred/filter to data and map with args and kwargs.

    Examples:
        >>> from rich.pretty import install
        >>> from rc import map_with_args
        >>> install()
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


def noexception(exception, func, *args, default_=None, **kwargs):
    """
    Execute function suppressing exceptions.

    Examples:
        >>> from rc import noexception
        >>>
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


def split_sep(sep='_'): return dict(sep=sep) if sep else dict()


def startswith(name: str, builtin=True): return name.startswith('__') if builtin else name.startswith('_')


def to_iter(data, always=False, split=' '):
    """
    To iter.

    Examples:
        >>> from rich import pretty
        >>> from rc import to_iter
        >>> pretty.install()
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
        _stack = inspect.stack()
        func = _stack[index - 1].function
        index = index + 1 if func == POST_INIT_NAME else index
        if line := textwrap.dedent(_stack[index].code_context[0]):
            if var := re.sub(f'(.| ){func}.*', str(), line.split(' = ')[0].replace('assert ', str()).split(' ')[0]):
                return (var.lower() if lower else var).split(**split_sep(sep))[0]


def yield_if(data, pred=lambda x: True if x else False, split=' ', apply=None):
    """
    Yield value if condition is met and apply function if predicate.

    Examples:
        >>> from rich.pretty import install
        >>> from rc import yield_if
        >>> install()
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
        >>> from rich.pretty import install
        >>> from rc import yield_if
        >>> install()
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
        yield count == total, *(i, data.get(i) if mm else None, )


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
