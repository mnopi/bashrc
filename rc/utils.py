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
    'datafield',
    'datafields',
    'insstack',
    'PathLib',
    'Simple',

    'Exit',
    'dpathdelete',
    'dpathget',
    'dpathnew',
    'dpathset',
    'dpathsearch',
    'dpathvalues',
    'Environs',
    'GitRepo',
    'pretty_install',
    'traceback_install',

    'Alias',
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
    'Seq',

    'Annotation',
    'AnnotationsType',
    'AsDict',
    'AsDictMethod',
    'AsDictProperty',
    'Attr',
    'Attribute',
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
    'IsCoro',
    'Kind',
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
    'asdict',
    'asdict_props',
    'cmdname',
    'current_task_name',
    'delete',
    'delete_list',
    'dict_sort',
    'effect',
    'get',
    'get_getattrtype',
    'getset',
    'info',
    'is_even',
    'iscoro',
    'in_dict',
    'join_newline',
    'map_reduce_even',
    'map_with_args',
    'newprop',
    'noexception',
    'pproperty',
    'prefixed',
    'repr_format',
    'runinloop',
    'runwarning',
    'split_sep',
    'startswith',
    'to_camel',
    'to_iter',
    'token_open',
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

    'TestBase',
    'TestData',
    'TestDataDictMix',
    'TestDataDictSlotMix',
)

import _abc
import ast
import functools
import json
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
from asyncio import Semaphore
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
from dataclasses import dataclass
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
from inspect import stack as insstack
from io import BytesIO
from io import FileIO
from io import StringIO
from logging import Logger
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
from typing import Type
from typing import Union
from warnings import catch_warnings
from warnings import filterwarnings
import jsonpickle
from box import Box
from bson import ObjectId
from click import secho
from click.exceptions import Exit as Exit
from decorator import decorator
from devtools import Debug
from dpath.util import delete as dpathdelete
from dpath.util import get as dpathget
from dpath.util import new as dpathnew
from dpath.util import set as dpathset
from dpath.util import search as dpathsearch
from dpath.util import values as dpathvalues
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
from more_itertools import consume
from more_itertools import first_true
from more_itertools import map_reduce
from more_itertools import side_effect
from nested_lookup import nested_lookup
from rich.console import Console
from rich.pretty import install as pretty_install
from rich.traceback import install as traceback_install

Alias = _alias
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
Seq = Union[typing.AnyStr, typing.ByteString, typing.Iterator, typing.KeysView, typing.MutableSequence,
            typing.MutableSet, typing.Sequence, tuple, typing.ValuesView]

# TODO: iscoro
# TODO: asdict_props
# TODO: runinloop
# TODO: Es.asdict
# TODO: Es.__hash__
# TODO: Es.__getstate__
# TODO: Es.__repr__
# TODO: Es.__setstate__
# TODO: Es.__str__
# TODO: Es.asdict
# TODO: classify_cls


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
    @classmethod
    def __subclasshook__(cls, C):
        if cls is AnnotationsType:
            return Mro.annotations.firstdict(C) is not NotImplemented
        return NotImplemented


class AsDict:
    """
    Dict and Attributes Class.

    Examples:

        .. code-block:: python

            json = jsonpickle.encode(col)
            obj = jsonpickle.decode(obj)
            col.to_file(name=col.col_name)
            assert (Path.cwd() / f'{col.col_name}.json').is_file()
            col.to_file(directory=tmpdir, name=col.col_name, regenerate=True)
            obj = col.from_file(directory=tmpdir, name=col.col_name)
            assert obj == col
    """
    __ignore_attr__ = ['asdict', 'attrs', 'keys', 'kwargs', 'kwargs_dict', 'public', 'values', 'values_dict', ]

    @property
    def asdict(self):
        """
        Dict including properties without routines and recursive.

        Returns:
            dict:
        """
        return info(self).asdict()

    @property
    def attrs(self):
        """
        Attrs including properties.

        Excludes:
            __ignore_attr__
            __ignore_copy__ instances.
            __ignore_kwarg__

        Returns:
            list:
        """
        return info(self).attrs

    def attrs_get(self, *args, default=None):
        """
        Get One or More Attributes.

        Examples:
            >>> from rc import AsDict
            >>> a = AsDict()
            >>> a.d1 = 1
            >>> a.d2 = 2
            >>> a.d3 = 3
            >>> assert a.attrs_get('d1') == {'d1': 1}
            >>> assert a.attrs_get('d1', 'd3') == {'d1': 1, 'd3': 3}
            >>> assert a.attrs_get('d1', 'd4', default=False) == {'d1': 1, 'd4': False}

        Raises:
            ValueError: ValueError

        Args:
            *args: attr(s) name(s).
            default: default.

        Returns:
            dict:
        """
        if not args:
            raise ValueError(f'args must be provided.')
        return {item: getattr(self, item, default) for item in args}

    def attrs_set(self, *args, **kwargs):
        """
        Sets one or more attributes.

        Examples:
            >>> from rc import AsDict
            >>> a = AsDict()
            >>> a.attrs_set(d_1=31, d_2=32)
            >>> a.attrs_set('d_3', 33)
            >>> d_4_5 = dict(d_4=4, d_5=5)
            >>> a.attrs_set(d_4_5)
            >>> a.attrs_set('c_6', 36, c_7=37)


        Raises:
            ValueError: ValueError

        Args:
            *args: attr name and value.
            **kwargs: attrs names and values.
        """
        if args:
            if len(args) > 2 or (len(args) == 1 and not isinstance(args[0], dict)):
                raise ValueError(f'args, invalid args length: {args}. One dict or two args (var name and value.')
            kwargs.update({args[0]: args[1]} if len(args) == 2 else args[0])

        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def defaults(cls, nested=True):
        """
        Return a dict with class attributes names and values.

        Returns:
            list:
        """
        return info(cls, depth=None if nested else 1).asdict(defaults=True)

    def from_file(self, directory=None, name=None, keys=True):
        name = name if name else self.__class__.__name__
        directory = PathLib(directory) if directory else PathLib.cwd()
        with (PathLib(directory) / f'{name}.json').open() as f:
            return jsonpickle.decode(json.load(f), keys=keys)

    @property
    def keys(self):
        """
        Keys from kwargs to init class (not InitVars), exclude __ignore_kwarg__ and properties.

        Returns:
            list:
        """
        return info(self).keys

    @property
    def kwargs(self):
        """
        Kwargs to init class with python objects no recursive, exclude __ignore_kwarg__ and properties.

        Example: Mongo binary.

        Returns:
            dict:
        """
        return info(self).kwargs

    @property
    def kwargs_dict(self):
        """
        Kwargs recursive to init class with python objects as dict, asdict excluding __ignore_kwarg__ and properties.

        Example: Mongo asdict.

        Returns:
            dict:
        """
        return info(self).kwargs_dict

    @property
    def public(self):
        """
        Dict including properties without routines.

        Returns:
            dict:
        """
        return info(self).public

    def to_file(self, directory=None, name=None, regenerate=False, **kwargs):
        name = name if name else self.__class__.__name__
        directory = PathLib(directory) if directory else PathLib.cwd()
        with (PathLib(directory) / f'{name}.json').open(mode='w') as f:
            json.dump(obj=info(self).to_json(regenerate=regenerate, **kwargs), fp=f, indent=4, sort_keys=True)

    def to_json(self, regenerate=True, indent=4, keys=True, max_depth=-1):
        return info(self).to_json(regenerate=regenerate, indent=indent, keys=keys, max_depth=max_depth)

    def to_obj(self, keys=True):
        return info(self).to_obj(keys=keys)

    @property
    def values(self):
        """
        Init python objects kwargs values no properties and not __ignore_kwarg__.

        Returns:
            list:
        """
        return info(self).values

    @property
    def values_dict(self):
        """
        Init python objects kwargs values no properties and not __ignore_kwarg__.

        Returns:
            list:
        """
        return info(self).values_dict


class AsDictMethod(metaclass=ABCMeta):
    """
    AsDict Method Support (Class, Static and Method).

    Examples:
        >>> from rc import AsDictMethod
        >>> from rc import info
        >>>
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
        if cls is AsDictMethod:
            value = Mro.asdict.firstdict(C)
            es = Es(value)
            return value is not NotImplemented and any(
                [es.classmethod, es.lambdatype, es.method, es.staticmethod]) and not es.prop
        return NotImplemented


class AsDictProperty(metaclass=ABCMeta):
    """
    AsDict Property Type.

    Examples:
        >>> from rc import AsDictProperty
        >>> from rc import info
        >>>
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
        if cls is AsDictProperty:
            return (value := Mro.asdict.firstdict(C)) is not NotImplemented and Es(value).prop
        return NotImplemented


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


Attribute = namedtuple('Attribute', 'coro defining es field hint kind name object')


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
        >>> from rc import Base
        >>> from rc import Cls
        >>> from rc import pproperty
        >>> from rc import pretty_install
        >>> from rc import TestBase
        >>>
        >>> pretty_install()
        >>> test = TestBase()
        >>>
        >>> sorted(Mro.hash_exclude.val(test))
        ['_slot']
        >>> sorted(Mro.ignore_attr.val(test))
        []
        >>> Mro.ignore_copy.val(test).difference(Mro.ignore_copy.default)
        set()
        >>> sorted(Mro.ignore_kwarg.val(test))
        []
        >>> Mro.ignore_str.val(test).difference(Mro.ignore_str.default)
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
        >>> test.info.cls.name
        'TestBase'
        >>> repr(test)  # doctest: +ELLIPSIS
        'TestBase(_hash: None,\\n_prop: None,\\n_slot: None,\\npprop: pprop)'
        >>> assert test.__repr_exclude__[0] not in repr(test)
        >>> test.prop
        >>> test.prop = 2
        >>> test.prop
        2
        >>> del test.prop
        >>> test.prop
        >>> assert hash((test._hash, test._prop, test._repr)) == hash(test)
        >>> set(test.__slots__).difference(test.info.cls.data_attrs)
        set()
        >>> sorted(test.info.cls.data_attrs)
        ['_hash', '_prop', '_repr', '_slot', 'classvar', 'initvar']
        >>> '__slots__' in sorted(test.info().cls.data_attrs)  # info() __call__(key=Attr.ALL)
        True
        >>> test.get.__name__ in test.info.cls.method
        True
        >>> test.get.__name__ in test.info.cls.callable
        True
        >>> 'prop' in test.info.cls.prop
        True
        >>> 'pprop' in test.info.cls.pproperty
        True
        >>> test.info.cls.importable_name  # doctest: +ELLIPSIS
        '....TestBase'
        >>> test.info.cls.importable_name == f'{test.info.cls.modname}.{test.info.cls.name}'
        True
        >>> test.info.cls.qualname
        'TestBase'
        >>> test.info.cls.attr_value('pprop')  # doctest: +ELLIPSIS
        <....pproperty object at 0x...>
        >>> test.info.attr_value('pprop')
        'pprop'
        >>> test.info.module  # doctest: +ELLIPSIS
        <module '...' from '/Users/jose/....py'>
        >>> assert sorted(test.info.dir) == sorted(test.info.cls.dir)
        >>> sorted(test.info.cls.memberdescriptor)
        ['_hash', '_prop', '_repr', '_slot']
        >>> sorted(test.info.cls.memberdescriptor) == sorted(test.__slots__)
        True
        >>> test.info.cls.methoddescriptor  # doctest: +ELLIPSIS
        [
            '__delattr__',
            ...,
            '__subclasshook__',
            'clsmethod',
            'static'
        ]
        >>> sorted(test.info.cls.method)
        ['_info', 'get', 'method_async']
        >>> sorted(test.info.cls().method)  # doctest: +ELLIPSIS
        [
            '__delattr__',
            ...,
            '__str__',
            '_info',
            'get',
            'method_async'
        ]
        >>> sorted(test.info.cls().callable) == \
        sorted(list(test.info.cls().method) + list(test.info.cls().classmethod) + list(test.info.cls().staticmethod))
        True
        >>> test.info.cls.setter
        ['prop']
        >>> test.info.cls.deleter
        ['prop']
        >>> test.info.cls.is_attr('_hash')
        True
        >>> test.info.cls.is_data('_hash')
        True
        >>> test.info.cls.is_deleter('prop')
        True
        >>> test.info.cls.is_memberdescriptor('_hash')
        True
        >>> test.info.cls.is_method('__repr__')
        True
        >>> test.info.cls.is_methoddescriptor('__repr__')
        False
        >>> test.info.cls.is_methoddescriptor('__str__')
        True
        >>> test.info.cls.is_pproperty('pprop')
        True
        >>> test.info.cls.is_property('prop')
        True
        >>> test.info.cls.is_routine('prop')
        False
        >>> test.info.cls.is_setter('prop')
        True
        >>> test.info.cls.is_attr('classvar')
        True
        >>> test.info.is_attr('classvar')
        True
        >>> sorted(test.info.coros)
        ['method_async', 'pprop_async']
        >>> test.info.coros_pproperty
        ['pprop_async']
        >>> test.info.coros_property
        ['pprop_async']
        >>> test.info.is_coro('pprop_async')
        True
        >>> test.info.is_coro('method_async')
        True
        >>> test.info.is_coro_pproperty('pprop_async')
        True
        >>> test.info.is_coro_property('pprop_async')
        True
        >>> test.info.cls.classvar
        ['classvar']
        >>> test.info.cls.initvar
        ['initvar']
        >>> test.info.cls.is_classvar('classvar')
        True
        >>> test.info.cls.is_initvar('initvar')
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

    def __hash__(self): return self.info.hash

    def __repr__(self): return self.info.repr

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

    def _info(self, depth=None, ignore=False, key=Attr.PRIVATE): return self.info(depth=depth, ignore=ignore, key=key)

    @property
    def info(self): return info(self)


class BoxKeys(Box):
    """
    Creates a Box with values from keys.
    """

    def __init__(self, keys, value='lower'):
        """
        Creates Box instance.

        Examples:
            >>> from rc import BoxKeys
            >>> from rc import pretty_install
            >>>
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
    """Class Helper Class."""
    __slots__ = ('args', 'asdict', 'attr', 'builtin', 'cache', 'cached_property', 'classified', 'classmethod',
                 'classvar',
                 'coro', 'data',
                 'defaults', 'dir',
                 'dynamicclassattribute',
                 'es', 'factory',
                 'fields', 'ignore', 'initvar', 'key', 'kwargs',
                 'method', 'mro', 'name',
                 'pproperty', 'prop', 'property_any', 'slots', 'staticmethod')
    # __slots__ = ('asyncgen', 'asyncgeneratortype', 'asyncgenfunc', 'attrs', 'awaitable', 'builtinfunctiontype',
    #              'callable', 'classmethod', 'classmethoddescriptortype', 'coroutine',
    #              'coroutinefunc', 'coroutinetype', 'data',
    #              'datafactory', 'datafield',
    #              'deleter', 'defined_class', 'defined_obj', 'defaults', 'dir', 'dynamicclassattribute',
    #              'es', 'functiontype', 'generator', 'generatortype', 'getsetdescriptor',
    #              'ignore', 'initvar', 'key', 'lambdatype',
    #              'mappingproxytype', 'memberdescriptor', 'method', 'methoddescriptor', 'methoddescriptortype',
    #              'methodtype', 'methodwrappertype',
    #              'mro', 'names',
    #              'none', 'routine', 'setter', 'source'
    #              'wrapperdescriptortype', )
    kind_compose = ('annotation', 'data_attrs', 'datafactory_dict', 'datafield_dict', 'slots/memberdescriptor')

    def __init__(self, data, ignore=False, key=Attr.PRIVATE):
        """
        Class Helper init.

        Examples:
            >>> from rc import Cls
            >>> from rc import pretty_install
            >>> from rc import TestData
            >>>
            >>> pretty_install()
            >>>
            >>> test = Cls(TestDataDictSlotMix)
            >>>
            #
            # Data Class Fields
            #
            >>> list(test.datafield)
            [
                'dataclass_classvar',
                'dataclass_default',
                'dataclass_default_factory',
                'dataclass_default_factory_init',
                'dataclass_default_init',
                'dataclass_initvar',
                'dataclass_str'
            ]
            >>> test.datafield  # doctest: +ELLIPSIS
            {
                'dataclass_classvar': Field(name='dataclass_classvar',...),
                'dataclass_default': Field(name='dataclass_default',...),
                'dataclass_default_factory': Field(name='dataclass_default_factory',...),
                'dataclass_default_factory_init': Field(name='dataclass_default_factory_init',...),
                'dataclass_default_init': Field(name='dataclass_default_init',...),
                'dataclass_initvar': Field(name='dataclass_initvar',...),
                'dataclass_str': Field(name='dataclass_str',...)
            }
            >>>
            #
            # Data Class Fields - default_factory - ['dataclass_default_factory', 'dataclass_default_factory_init']
            #
            >>> list(test.datafactory)
            ['dataclass_default_factory', 'dataclass_default_factory_init']
            >>> test.datafactory
            {'dataclass_default_factory': {}, 'dataclass_default_factory_init': {}}
            >>> 'dataclass_default_factory' in TestDataDictSlotMix.__dataclass_fields__
            True
            >>> 'dataclass_default_factory' in dir(TestDataDictSlotMix)
            False
            >>> 'dataclass_default_factory' in vars(TestDataDictSlotMix)
            False
            >>> 'dataclass_default_factory' in vars(TestDataDictSlotMix())
            True
            >>> vars(TestDataDictSlotMix()).get('dataclass_default_factory')
            {}
            >>> getattr(TestDataDictSlotMix(), 'dataclass_default_factory')
            {}
            >>> 'dataclass_default_factory_init' in TestDataDictSlotMix.__dataclass_fields__
            True
            >>> 'dataclass_default_factory_init' in dir(TestDataDictSlotMix)
            False
            >>> 'dataclass_default_factory_init' in vars(TestDataDictSlotMix)
            False
            >>> 'dataclass_default_factory_init' in vars(TestDataDictSlotMix())
            True
            >>> vars(TestDataDictSlotMix()).get('dataclass_default_factory_init')
            {}
            >>> getattr(TestDataDictSlotMix(), 'dataclass_default_factory_init')
            {}
            >>> vars(TestDataDictSlotMix())['dataclass_default_factory_init'] == \
            test.datafactory['dataclass_default_factory_init']
            True
            >>>
            #
            # Data Class Fields - ['dataclass_classvar']
            #
            >>> list(test.classvar)
            ['dataclass_classvar']
            >>> test.classvar  # doctest: +ELLIPSIS
            {
                'dataclass_classvar': Field(name='dataclass_classvar',...)
            }
            >>> 'dataclass_classvar' in TestDataDictSlotMix.__dataclass_fields__
            True
            >>> 'dataclass_classvar' in dir(TestDataDictSlotMix)
            True
            >>> 'dataclass_classvar' in vars(TestDataDictSlotMix)
            False
            >>> 'dataclass_classvar' in vars(TestDataDictSlotMix())
            False
            >>> vars(TestDataDictSlotMix()).get('dataclass_classvar')
            >>> getattr(TestDataDictSlotMix(), 'dataclass_classvar')
            'dataclass_classvar'
            >>>
            #
            # Data Class Fields - ['dataclass_initvar']
            #
            >>> list(test.initvar)
            ['dataclass_initvar']
            >>> test.initvar  # doctest: +ELLIPSIS
            {
                'dataclass_initvar': Field(name='dataclass_initvar',...)
            }
            >>> 'dataclass_initvar' in TestDataDictSlotMix.__dataclass_fields__
            True
            >>> 'dataclass_initvar' in dir(TestDataDictSlotMix)
            True
            >>> 'dataclass_initvar' in vars(TestDataDictSlotMix)
            False
            >>> 'dataclass_initvar' in vars(TestDataDictSlotMix())
            False
            >>> vars(TestDataDictSlotMix()).get('dataclass_initvar')
            >>> getattr(TestDataDictSlotMix(), 'dataclass_initvar')
            'dataclass_initvar'
            >>>
            #
            # Dict - ClassVar - ['subclass_classvar']
            #

x

        Args:
            data: Class to provide information.
            ignore: ignore properties and kwargs :class:`Base.__ignore_kwargs__` (default: False)
            key: keys to include (default: :attr:`rc.Key.PRIVATE`)

        Returns:
            Cls Instance.
        """
        effect(lambda x: self.__setattr__(x, dict()), self.__slots__)
        self.data = data if isinstance(data, type) else type(data)
        self.es = Es(self.data)
        self.ignore = ignore
        self.key = key
        self.classified = dict_sort({i.name: i for i in classify_class_attrs(self.data) if self.key.include(i.name)})
        self.fields = dict(filter(lambda x: self.key.include(x[0]), dict_sort(self.data.__dataclass_fields__).items())) \
            if self.es.datatype_sub else dict()
        self.attr
        factories = filter(lambda x: Es(x[1]).datafactory, self.fields.items())
        if self.es.datatype_sub:
            for i in sorted(self.fields):
                if self.key.include(i):
                    v = self.fields[i]
                    es = Es(v)
                    self.datafield |= {i: v}
                    if es.classvar:
                        self.classvar |= {i: v}
                    elif es.datafactory:
                        self.factory |= {i: v.default_factory()}
                    elif es.initvar:
                        self.initvar |= {i: v}

    def __call__(self, ignore=False, key=Attr.ALL):
        """
        Updates instance with ignore adn key (default: Attr.ALL)

        Examples:
            >>> from rc import Cls
            >>> from rc import pretty_install
            >>>
            >>> pretty_install()
            >>>
            >>> cls = Cls(dict())
            >>> cls.is_callable(Mro.getitem.value)
            False
            >>> cls().callable
            >>> cls().is_callable(Mro.getitem.value)
            True

        Args:
            ignore: ignore properties and kwargs :class:`Base.__ignore_kwargs__` (default: False)            key:
            key: keys to include (default: :attr:`rc.Key.PRIVATE`)

        Returns:
            Updated Cls Instance.
        """
        return self.__init__(data=self.data, ignore=ignore, key=key)

    @functools.cache
    def _annotations(self, stack=2):
        return annotations(self.data, stack=stack)

    @property
    def _asyncgen(self):
        return self.kind[self.asyncgen.__name__]

    @property
    def _asyncgenfunc(self):
        return self.kind[self.asyncgenfunc.__name__]

    @functools.cache
    def _attr_value(self, name, default=None): return getattr(self.data, name, default)

    @property
    def _awaitable(self):
        return self.kind[self.awaitable.__name__]

    @property
    def _builtinfunctiontype(self):
        return self.kind[self.builtinfunctiontype.__name__]

    @property
    def _by_kind(self): return bucket(self.classified, key=lambda x: x.kind if self.key.include(x.name) else 'e')

    @property
    def _by_name(self): return {i.name: i for i in self.classified if self.key.include(i.name)}

    @property
    def _cache(self):
        return self.kind[self.cache.__name__]
        # return sorted([key for key, value in self.data.__dict__.items() if Es(value).cache and self.key.include(key)])

    @property
    def _cached_property(self):
        return self.kind[self.cached_property.__name__]
        # return sorted([key for key, value in self.data.__dict__.items()
        # if Es(value).cached_property and self.key.include(key)])

    @property
    def _callable(self):
        """
        Class Callables filtered based on startswith.

        Examples:
            >>> from rc import Cls
            >>> from rc import pretty_install
            >>> from rc import TestBase
            >>>
            >>> pretty_install()
            >>> test = TestBase()
            >>> test.info.cls.callable
            ['_info', 'clsmethod', 'get', 'method_async', 'static']
            >>> test.info.cls().callable  # doctest: +ELLIPSIS
            [
                '__delattr__',
                '__dir__',
                ...,
                '__str__',
                '__subclasshook__',
                '_info',
                'clsmethod',
                'get',
                'method_async',
                'static'
            ]

        Returns:
            List of Class Callables names filtered based on startswith.
        """
        return self.kind[self.callable.__name__]
        # return sorted(self.classmethods + self.methods + self.staticmethods)

    @property
    def _classmethod(self):
        """
        Class Methods filtered based on startswith.

        Examples:
            >>> from rc import Cls
            >>> from rc import pretty_install
            >>> from rc import TestBase
            >>>
            >>> pretty_install()
            >>> test = TestBase()
            >>> test.info.cls.classmethod
            ['clsmethod']
            >>> test.clsmethod.__name__ in test.info.cls.classmethod
            True
            >>> test.info.cls().classmethod
            ['__init_subclass__', '__subclasshook__', 'clsmethod']
            >>> Mro.init_subclass.value in test.info.cls().classmethod
            True

        Returns:
            List of Class Methods names filtered based on startswith.
        """
        return self.kind[self.classmethod.__name__]
        # return list(map(Name.name_.getter, self.by_kind['class method']))

    # @property
    # def classvar(self):
    #     return self.kind[self.classvar.__name__]
    #     # return [key for key, value in self.annotations(stack=3).items() if value.classvar]

    @property
    def _coro(self):
        return self.kind[self.coro.__name__]

    @property
    def _coroutine(self):
        return self.kind[self.coroutine.__name__]

    @property
    def _coroutinefunc(self):
        return self.kind[self.coroutinefunc.__name__]

    @property
    def _data_attrs(self):
        """
        Class Data Attributes including Dataclasses Fields with default factory are not in classified,
        filtered based on startswith.

        Examples:
            >>> from rc import Cls
            >>> from rc import pretty_install
            >>> from rc import TestData
            >>>
            >>> pretty_install()
            >>> testdata = Cls(TestData)
            >>> testdata.data_attrs
            ['a', 'b', 'c', 'd', 'e', 'f', 'g']
            >>> testdata.dir
            ['a', 'b', 'c', 'd', 'e', 'f', 'g']
            >>> testdata.dir == testdata.data_attrs == testdata.datafield
            True

        Returns:
            List of attribute names.
        """
        return sorted({*map(Name.name_.getter, self.by_kind['data']), *self.datafield_dict.keys()})

    @property
    def _defaults(self):
        """
        Data Class Fields List.

        Examples:
            >>> from rc import Cls
            >>> from rc import pretty_install
            >>> from rc import TestData
            >>>
            >>> pretty_install()
            >>> testdata = Cls(TestData)
            >>> testdata.defaults
            ['a', 'b', 'c', 'd', 'e', 'f', 'g']

        Returns:
            Data Class Field Attribute Names List.
        """
        return self

    @property
    def _deleter(self):
        return self.kind[self.deleter.__name__]
        # return sorted([i.name for i in self.classified if Es(i.object).deleter])

    @property
    def _dir(self):
        # noinspection PyUnresolvedReferences
        """
        Class Data Attributes including Dataclasses Fields with default factory are not in dir,
            filtered based on startswith.

        Examples:
            >>> from rc import Cls
            >>> from rc import pretty_install
            >>> from rc import TestData
            >>>
            >>> pretty_install()
            >>> testdata = Cls(TestData)
            >>> testdata.data_attrs
            ['a', 'b', 'c', 'd', 'e', 'f', 'g']
            >>> testdata.dir
            ['a', 'b', 'c', 'd', 'e', 'f', 'g']
            >>> testdata.data_attrs == sorted([i for i in dir(TestData) \
            if testdata.key.include(i)] + testdata.datafactories)
            True

        Returns:
            List of attribute names.
        """
        return sorted([i for i in {*dir(self.data), *self.datafield_dict.keys()} if self.key.include(i)])

    @property
    def _generator(self):
        return self.kind[self.generator.__name__]

    @property
    def _getsetdescriptor(self):
        return self.kind[self.getsetdescriptor.__name__]

    @functools.cache
    def has_attr(self, name): return hasattr(self.data, name)

    @functools.cache
    def has_method(self, name): return has_method(self.data, name)

    @property
    def has_reduce(self): return has_reduce(self.data)

    @property
    def importable_name(self): return importable_name(self.data)

    @functools.cache
    def is_attr(self, name):
        return name in self.dir

    def is_callable(self, name):
        """
        Is Class Callable filtered based on startswith.

        Examples:
            >>> from rc import pretty_install
            >>> from rc import TestBase
            >>>
            >>> pretty_install()
            >>>
            >>> test = TestBase()
            >>> test.info.cls.is_callable('clsmethod')
            True
            >>> test.info.cls.is_callable(Mro.str.value)
            False
            >>> test.info.cls().is_callable(Mro.str.value)
            True
            >>> test.info.cls.is_callable('prop')
            False

        Returns:
            True if Callable Method name filtered based on startswith.
        """
        return name in self.callable

    @functools.cache
    def is_classmethod(self, name):
        """
        Is Class Method filtered based on startswith.

        Examples:
            >>> from rc import pretty_install
            >>> from rc import TestBase
            >>>
            >>> pretty_install()
            >>>
            >>> test = TestBase()
            >>> test.info.cls.is_classmethod('clsmethod')
            True
            >>> test.info.cls.is_classmethod(Mro.init_subclass.value)
            False
            >>> test.info.cls().is_classmethod(Mro.init_subclass.value)
            True
            >>> test.info.cls.is_classmethod('pprop_async')
            False

        Returns:
            True if Class Method name filtered based on startswith.
        """
        return name in self.classmethod

    @functools.cache
    def is_classvar(self, name):
        return name in self.classvar

    @functools.cache
    def is_data(self, name):
        return name in self.data_attrs

    @functools.cache
    def is_deleter(self, name):
        return name in self.deleter

    @functools.cache
    def is_datafactory(self, name):
        return name in self.datafactory

    @functools.cache
    def is_datafield(self, name):
        return name in self.datafield

    @functools.cache
    def is_initvar(self, name):
        return name in self.initvar

    @functools.cache
    def is_memberdescriptor(self, name):
        return name in self.memberdescriptor

    @functools.cache
    def is_method(self, name):
        return name in self.method

    @functools.cache
    def is_methoddescriptor(self, name):
        return name in self.methoddescriptor

    @functools.cache
    def is_pproperty(self, name):
        return name in self.pproperty

    @functools.cache
    def is_property(self, name):
        return name in self.prop

    @functools.cache
    def is_routine(self, name):
        return name in self.routine

    @functools.cache
    def is_setter(self, name):
        return name in self.setter

    @functools.cache
    def is_staticmethod(self, name):
        """
        Is Static Method filtered based on startswith.

        Examples:
            >>> from rc import pretty_install
            >>> from rc import TestBase
            >>>
            >>> pretty_install()
            >>>
            >>> test = TestBase()
            >>> test.info.cls.is_staticmethod('static')
            True
            >>> test.info.cls.is_staticmethod('pprop_async')
            False

        Returns:
            True if Static Methods name filtered based on startswith.
        """
        return name in self.staticmethod

    def kind(self, value):
        # TODO: coger el code de la clase de inspect y poner el source aqui y ver el code para el async!
        #  o ponerlo en la de Name!!!
        _fields = dict_sort(value.__dataclass_fields__) if self.es.datatype_sub else dict()

        _dict = sorted([i for i in {*dir(value), *_fields.keys()} if self.key.include(i)])
        self._kind = {kind: {name: obj for name, obj in _dict.items() if getattr(Es(obj), kind)
                             and self.key.include(name)} for kind in self.kind_attr}

    @property
    def _mappingproxytype(self):
        return self.kind[self.mappingproxytype.__name__]

    @property
    def _memberdescriptor(self):
        return self.kind[self.memberdescriptor.__name__]
        # return [i.name for i in self.classified if Es(i.object).memberdescriptor]

    @property
    def _method(self):
        return self.kind[self.method.__name__]
        # return list(map(Name.name_.getter, self.by_kind['method']))

    @property
    def _methoddescriptor(self):
        """
        Includes classmethod, staticmethod and methods but not functions defined (i.e: def info(self))

        Returns:
            Method descriptors.
        """
        return self.kind[self.methoddescriptor.__name__]
        # return [i.name for i in self.classified if Es(i.object).methoddescriptor]

    @property
    def _methodwrappertype(self):
        return self.kind[self.methodwrappertype.__name__]

    @property
    def modname(self): return Name._module0.get(self.data, default=str())

    @property
    def _mro(self): return self.data.__mro__

    @property
    def _name(self): return self.data.__name__

    @property
    def _none(self):
        return self.kind[self.none.__name__]

    @property
    def _pproperty(self):
        return self.kind[self.pproperty.__name__]
        # return [i.name for i in self.classified if Es(i.object).pproperty]

    @property
    def _prop(self):
        return self.kind[self.prop.__name__]
        # return list(map(Name.name_.getter, self.by_kind['property']))

    @property
    def _property_any(self):
        return self.kind[self.property_any.__name__]
        # return list(map(Name.name_.getter, self.by_kind['property']))

    @staticmethod
    def propnew(name, default=None):
        return property(
            lambda self:
            self.__getattribute__(f'_{name}', default=default(self) if isinstance(default, partial) else default),
            lambda self, value: self.__setattr__(f'_{name}', value),
            lambda self: self.__delattr__(f'_{name}')
        )

    @property
    def _qualname(self): return Name._qualname0.get(self.data, default=str())

    @property
    def _routine(self):
        return self.kind[self.routine.__name__]
        # return [i.name for i in self.classified if Es(i.object).routine]

    @property
    def _setter(self):
        return self.kind[self.setter.__name__]
        # return [i.name for i in self.classified if Es(i.object).setter]

    @property
    def _staticmethod(self):
        """
        Static Methods filtered based on startswith.

        Examples:
            >>> from rc import Cls
            >>> from rc import pretty_install
            >>> from rc import TestBase
            >>>
            >>> pretty_install()
            >>> test = TestBase()
            >>> test.info.cls.staticmethod
            ['static']
            >>> test.static.__name__ in test.info.cls.staticmethod
            True

        Returns:
            List of Static Methods names filtered based on startswith.
        """
        return self.kind[self.staticmethod.__name__]
        # return list(map(Name.name_.getter, self.by_kind['static method']))


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
    builtins = (v if isinstance(v := globals()['__builtins__'], dict) else vars(v)).copy()
    builtins_classes = tuple(filter(lambda x: isinstance(x, type), builtins.values()))
    builtins_functions = tuple(filter(lambda x: isinstance(x, (BuiltinFunctionType, FunctionType, )), builtins.values()))
    builtins_other = tuple(map(builtins.get, ('__doc__', '__import__', '__spec__', 'copyright', 'credits', 'exit',
                                              'help', 'license', 'quit', )))
    __slots__ = ('data', )
    __ignore_attr__ = ('asdict', )

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

    def __getstate__(self): return dict(data=self.data)
    def __repr__(self): return f'{self.__class__.__name__}({self.data})'
    def __setstate__(self, state): self.data = state['data']
    def __str__(self): return str(self.data)
    _data = property(lambda self: self.data.object if self.attribute else self.data)
    _fget = property(lambda self: self._data.fget if self.property_any else None)
    _func = property(lambda self: self._data.__func__ if self.instance(classmethod, staticmethod) else None)
    annotation = property(lambda self: isinstance(self._data, Annotation))
    annotationstype = property(lambda self: isinstance(self._data, AnnotationsType))
    annotationstype_sub = property(lambda self: issubclass(self._data, AnnotationsType))
    asdict = property(lambda self: dict(data=self.data) | asdict_props(self))
    asdictmethod = property(lambda self: isinstance(self._data, AsDictMethod))
    asdictmethod_sub = property(lambda self: issubclass(self._data, AsDictMethod))
    asdictproperty = property(lambda self: isinstance(self._data, AsDictProperty))
    asdictproperty_sub = property(lambda self: issubclass(self._data, AsDictProperty))
    ast = property(lambda self: isinstance(self._data, AST))
    asyncfor = property(lambda self: isinstance(self._data, AsyncFor))
    asyncfunctiondef = property(lambda self: isinstance(self._data, AsyncFunctionDef))
    asyncgen = property(lambda self: isasyncgen(self._fget or self._func or self._data))
    asyncgeneratortype = property(lambda self: isinstance(self._fget or self._func or self._data, AsyncGeneratorType))
    asyncgenfunc = property(lambda self: isasyncgenfunction(self._fget or self._func or self._data))
    asyncwith = property(lambda self: isinstance(self._data, AsyncWith))
    attribute = property(lambda self: isinstance(self.data, Attribute))
    await_ast = property(lambda self: isinstance(self._data, Await))
    awaitable = property(lambda self: isawaitable(self._fget or self._func or self._data))
    bool = property(lambda self: isinstance(self._data, int) and isinstance(self._data, bool))
    builtin = property(lambda self: any([in_dict(self.builtins, self._data), self.builtinclass, self.builtinfunction]))
    builtinclass = property(lambda self: self._data in self.builtins_classes)
    builtinfunction = property(lambda self: self._data in self.builtins_functions)
    builtinfunctiontype = property(lambda self: isinstance(self._data, BuiltinFunctionType))
    bytesio = property(lambda self: isinstance(self._data, BytesIO))  # :class:`typing.BinaryIO`
    cache = property(lambda self: Mro.cache_clear.has(self._data))
    cached_property = property(lambda self: isinstance(self._data, cached_property))
    callable = property(lambda self: isinstance(self._data, Callable))
    chain = property(lambda self: isinstance(self._data, Chain))
    chainmap = property(lambda self: isinstance(self._data, ChainMap))
    classdef = property(lambda self: isinstance(self._data, ClassDef))
    classmethod = property(lambda self: isinstance(self._data, classmethod))
    classmethoddescriptortype = property(lambda self: isinstance(self._data, ClassMethodDescriptorType))
    classvar = property(
        lambda self: (self.datafield and get_origin(self._data.type) == ClassVar) or get_origin(self._data) == ClassVar)
    codetype = property(lambda self: isinstance(self._data, CodeType))
    collections = property(lambda self: is_collections(self._data))
    container = property(lambda self: isinstance(self._data, Container))
    coro = property(
        lambda self: any([self.asyncgen, self.asyncgenfunc, self.awaitable, self.coroutine, self.coroutinefunc]))
    coroutine = property(lambda self: iscoroutine(self._fget or self._func or self._data))
    coroutinefunc = property(lambda self: iscoroutinefunction(self._fget or self._func or self._data))
    coroutinetype = property(lambda self: isinstance(self._data, CoroutineType))
    datafactory = property(
        lambda self: self.datafield and Es(self._data.default).missing and hasattr(self._data, 'default_factory'))
    datafield = property(lambda self: isinstance(self._data, Field))
    datatype = property(lambda self: isinstance(self._data, DataType))
    datatype_sub = property(lambda self: issubclass(self._data, DataType))
    defaultdict = property(lambda self: isinstance(self._data, defaultdict))
    deleter = property(lambda self: self.property_any and self._data.fdel is not None)
    dict = property(lambda self: isinstance(self._data, dict))
    dicttype = property(lambda self: isinstance(self._data, DictType))
    dicttype_sub = property(lambda self: issubclass(self._data, DictType))
    dynamicclassattribute = property(lambda self: isinstance(self._data, DynamicClassAttribute))
    dlst = property(lambda self: isinstance(self._data, (dict, list, set, tuple)))
    enum = property(lambda self: isinstance(self._data, Enum))
    enum_sub = property(lambda self: issubclass(self._data, Enum))
    enumdict = property(lambda self: isinstance(self._data, EnumDict))
    enumdict_sub = property(lambda self: issubclass(self._data, EnumDict))
    even: property(lambda self: not self._data % 2)
    fileio = property(lambda self: isinstance(self._data, FileIO))
    float = property(lambda self: isinstance(self._data, float))
    frameinfo = property(lambda self: isinstance(self._data, FrameInfo))
    frametype = property(lambda self: isinstance(self._data, FrameType))
    functiondef = property(lambda self: isinstance(self._data, FunctionDef))
    functiontype = property(lambda self: isinstance(self._data, FunctionType))
    generator = property(lambda self: isinstance(self._data, Generator))
    generatortype = property(lambda self: isinstance(self._data, GeneratorType))
    genericalias = property(lambda self: isinstance(self._data, types.GenericAlias))
    getattrnobuiltintype = property(lambda self: isinstance(self._data, GetAttrNoBuiltinType))
    getattrnobuiltintype_sub = property(lambda self: issubclass(self._data, GetAttrNoBuiltinType))
    getattrtype = property(lambda self: isinstance(self._data, GetAttrType))
    getattrtype_sub = property(lambda self: issubclass(self._data, GetAttrType))
    getsetdescriptor = lambda self, n: isgetsetdescriptor(self.cls_attr_value(n)) if n else self._data
    getsetdescriptortype = property(lambda self: isinstance(self._data, GetSetDescriptorType))
    gettype = property(lambda self: isinstance(self._data, GetType))
    gettype_sub = property(lambda self: issubclass(self._data, GetType))
    hashable = property(lambda self: bool(noexception(TypeError, hash, self._data)))
    import_ast = property(lambda self: isinstance(self._data, Import))
    importfrom = property(lambda self: isinstance(self._data, ImportFrom))
    initvar = property(
        lambda self: (self.datafield and isinstance(self._data.type, InitVar)) or isinstance(self._data, InitVar))
    installed = property(lambda self: is_installed(self._data))
    instance = lambda self, *args: isinstance(self._data, args)
    int = property(lambda self: isinstance(self._data, int))
    io = property(lambda self: self.bytesio and self.stringio)  # :class:`typing.IO`
    iterable = property(lambda self: isinstance(self._data, Iterable))
    iterator = property(lambda self: isinstance(self._data, Iterator))
    lambdatype = property(lambda self: isinstance(self._data, LambdaType))
    list = property(lambda self: isinstance(self._data, list))
    lst = property(lambda self: isinstance(self._data, (list, set, tuple)))
    mappingproxytype = property(lambda self: isinstance(self._data, MappingProxyType))
    mappingproxytype_sub = property(lambda self: issubclass(self._data, MappingProxyType))
    memberdescriptor = property(lambda self: ismemberdescriptor(self._data))
    memberdescriptortype = property(lambda self: isinstance(self._data, MemberDescriptorType))
    method = property(lambda self: self.methodtype and not self.instance(classmethod, property, staticmethod))
    methoddescriptor = property(lambda self: ismethoddescriptor(self._data))
    methoddescriptortype = property(lambda self: isinstance(self._data, types.MethodDescriptorType))
    methodtype = property(lambda self: isinstance(self._data, MethodType))  # True if it is an instance method!.
    methodwrappertype = property(lambda self: isinstance(self._data, MethodWrapperType))
    methodwrappertype_sub = property(lambda self: issubclass(self._data, MethodWrapperType))
    missing = property(lambda self: isinstance(self._data, MISSING_TYPE))
    mlst = property(lambda self: isinstance(self._data, (MutableMapping, list, set, tuple)))
    mm = property(lambda self: isinstance(self._data, MutableMapping))
    moduletype = property(lambda self: isinstance(self._data, ModuleType))
    module_function = property(lambda self: is_module_function(self._data))
    noncomplex = property(lambda self: is_noncomplex(self._data))
    namedtype = property(lambda self: isinstance(self._data, NamedType))
    namedtype_sub = property(lambda self: issubclass(self._data, NamedType))
    named_annotationstype = property(lambda self: isinstance(self._data, NamedAnnotationsType))
    named_annotationstype_sub = property(lambda self: issubclass(self._data, NamedAnnotationsType))
    none = property(lambda self: isinstance(self._data, type(None)))
    object = property(lambda self: is_object(self._data))
    pathlib = property(lambda self: isinstance(self._data, PathLib))
    picklable = lambda self, name: is_picklable(name, self._data)
    primitive = property(lambda self: is_primitive(self._data))
    pproperty = property(lambda self: isinstance(self._data, pproperty))
    prop = property(lambda self: isinstance(self._data, property))
    property_any = property(lambda self: self.prop or self.cached_property)
    reducible = property(lambda self: is_reducible(self._data))
    reducible_sequence_subclass = property(lambda self: is_reducible_sequence_subclass(self._data))
    routine = property(lambda self: isroutine(self._data))
    sequence = property(lambda self: is_sequence(self._data))
    sequence_subclass = property(lambda self: is_sequence_subclass(self._data))
    set = property(lambda self: isinstance(self._data, set))
    setter = property(lambda self: self.prop and self._data.fset is not None)
    simple = property(lambda self: isinstance(self._data, Simple))
    sized = property(lambda self: isinstance(self._data, Sized))
    slotstype = property(lambda self: isinstance(self._data, SlotsType))
    slotstype_sub = property(lambda self: issubclass(self._data, SlotsType))
    staticmethod = property(lambda self: isinstance(self._data, staticmethod))
    str = property(lambda self: isinstance(self._data, str))
    subclass = lambda self, *args: issubclass(self._data, args) if self.type else issubclass(type(self._data), args)
    stringio = property(lambda self: isinstance(self._data, StringIO))  # :class:`typing.TextIO`
    tracebacktype = property(lambda self: isinstance(self._data, TracebackType))
    tuple = property(lambda self: isinstance(self._data, tuple))
    type = property(lambda self: isinstance(self._data, type))
    unicode = property(lambda self: is_unicode(self._data))
    wrapperdescriptortype = property(lambda self: isinstance(self._data, WrapperDescriptorType))


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
    @abstractmethod
    def __getattribute__(self, n):
        return object.__getattribute__(self, n)

    @classmethod
    def __subclasshook__(cls, C):
        if cls is GetAttrNoBuiltinType:
            g = Mro.get.firstdict(C)
            return any([Mro._field_defaults.firstdict(C) is not NotImplemented,
                        not allin(C.__mro__, Es.builtins_classes) and g is NotImplemented or
                        (g is not NotImplemented and not callable(g))])
        return NotImplemented


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
class GetSupport(Protocol):
    """An ABC with one abstract method get."""
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


IsCoro = namedtuple('IsCoro', 'asyncfor asyncfunctiondef asyncgen asyncgeneratortype asyncgenfunc asyncwith '
                              'awaitable coro coroutine coroutinefunc coroutinetype')


class Kind(Enum):
    CLASS = 'class method'
    DATA = 'data'
    METHOD = 'method'
    PROPERTY = 'property'
    STATIC = 'static method'


class _Mro(Enum):
    def _generate_next_value_(self, start, count, last_values):
        exclude = ('_asdict', '_field_defaults', '_fields', 'asdict', 'get', )
        return self if self in exclude else f'__{self}__'


class Mro(_Mro):
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
            >>> from rc import Mro
            >>> from rc import pretty_install
            >>>
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
            >>> from rc import Mro
            >>> from rc import pretty_install
            >>>
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
            >>> from rc import CRLock
            >>> from rc import IgnoreAttr
            >>> from rc import IgnoreCopy
            >>> from rc import Mro
            >>>
            >>> Mro.ignore_attr.default  # doctest: +ELLIPSIS
            (
                'asdict',
                ...
            )
            >>> Mro.ignore_attr.default == IgnoreAttr.__args__
            True
            >>> Mro.ignore_copy.default  # doctest: +ELLIPSIS
            (
                <class '_thread.RLock'>,
                ...
            )
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
            >>> from rc import Mro
            >>> from rc import pretty_install
            >>>
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
            >>> from collections import namedtuple
            >>> from rc import Mro
            >>> from rc import pretty_install
            >>>
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
            >>> from collections import namedtuple
            >>> from rc import Mro
            >>> from rc import pretty_install
            >>>
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
            >>> from rc import Mro
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
            >>> from rc import Mro
            >>> from rc import pretty_install
            >>>
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
            >>> from rc import Mro
            >>> from rc import pretty_install
            >>>
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
            >>> from rc import Mro
            >>> from rc import pretty_install
            >>>
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
            >>> from rc import Mro
            >>> from rc import pretty_install
            >>>
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
            >>> from rc import Mro
            >>> from rc import pretty_install
            >>>
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
            >>> from rc import Mro
            >>> from rc import pretty_install
            >>>
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
    _dict0 = auto()
    _dir0 = auto()
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
    @functools.cache
    def _attrs(cls):
        """
        Get map_reduce (dict) attrs lists converted to real names.

        Examples:
            >>> from rc import Name
            >>> from rc import pretty_install
            >>>
            >>> pretty_install()
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
    @functools.cache
    def attrs(cls):
        """
        Get attrs tuple with private converted to real names.

        Examples:
            >>> from rc import Name
            >>> from rc import pretty_install
            >>>
            >>> pretty_install()
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
            >>> from ast import unparse
            >>> from inspect import FrameInfo
            >>> from inspect import getmodulename
            >>> from rc import insstack
            >>> from rc import Name
            >>> from rc import pretty_install
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
            >>> from ast import unparse
            >>> from inspect import FrameInfo
            >>> from inspect import getmodulename
            >>> import rc.utils
            >>> from rc import insstack
            >>> from rc import Name
            >>> from rc import pretty_install
            >>>
            >>> pretty_install()
            >>> f = insstack()[0]
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
            >>> from ast import unparse
            >>> from inspect import FrameInfo
            >>> from rc import insstack
            >>> from rc import Name
            >>>
            >>> f = insstack()[0]
            >>> assert f == FrameInfo(Name.frame.get(f), str(Name.filename.get(f)), Name.lineno.get(f),\
            Name.function.get(f), Name.code_context.get(f), Name.index.get(f))
            >>> assert Name.filename.get(f) == Name._file0.get(f)
            >>> Name.source(f)
            'f = insstack()[0]\\n'
            >>> unparse(Name.node(f))
            'f = insstack()[0]'
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
    @functools.cache
    def get_frametype(self, obj: FrameType, default=None):
        """
        Get value of attribute from FrameType.

        Examples:
            >>> from ast import unparse
            >>> from inspect import FrameInfo
            >>> from rc import insstack
            >>> from rc import Name
            >>>
            >>> frameinfo = insstack()[0]
            >>> f = frameinfo.frame
            >>> assert Name.filename.get(f) == Name.filename.get(frameinfo)
            >>> assert Name.frame.get(f) == Name.frame.get(frameinfo)
            >>> assert Name.lineno.get(f) == Name.lineno.get(frameinfo)
            >>> assert Name.function.get(f) == Name.function.get(frameinfo)
            >>> assert frameinfo == FrameInfo(Name.frame.get(f), str(Name.filename.get(f)), Name.lineno.get(f),\
            Name.function.get(f), frameinfo.code_context, frameinfo.index)
            >>> assert Name.filename.get(f) == Name._file0.get(f)
            >>> Name.source(f)
            'frameinfo = insstack()[0]\\n'
            >>> unparse(Name.node(f))
            'frameinfo = insstack()[0]'
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
            >>> from ast import unparse
            >>> from inspect import FrameInfo
            >>> from inspect import getmodulename
            >>> from rc import insstack
            >>> from rc import Name
            >>> from rc import pretty_install
            >>>
            >>> pretty_install()
            >>> f = insstack()[0]
            >>> globs_locs = (f.frame.f_globals | f.frame.f_locals).copy()
            >>> Name.filename.get(f), Name.function.get(f), Name.code_context.get(f)[0], Name.source(f) \
             # doctest: +ELLIPSIS
            (
                PosixPath('<doctest ...node[7]>'),
                '<module>',
                'f = insstack()[0]\\n',
                'f = insstack()[0]\\n'
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
            >>> from ast import unparse
            >>> from inspect import FrameInfo
            >>> from inspect import getmodulename
            >>> from rc import allin
            >>> from rc import insstack
            >>> from rc import Name
            >>> from rc import pretty_install
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
    @functools.cache
    def private(cls):
        """
        Get private attrs tuple converted to real names.

        Examples:
            >>> from rc import Name
            >>> from rc import pretty_install
            >>>
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
            >>> from rc import Name
            >>> from rc import pretty_install
            >>>
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
    @functools.cache
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
            >>> from ast import unparse
            >>> from inspect import FrameInfo
            >>> from inspect import getmodulename
            >>> import rc.utils
            >>> from rc import insstack
            >>> from rc import Name
            >>> from rc import allin
            >>> from rc import pretty_install
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


Types = Union[Type[AnnotationsType], Type[AsDictMethod], Type[AsDictProperty],
              Type[DataType], Type[DictType], Type[GetAttrType],
              Type[GetAttrNoBuiltinType], Type[GetType], Type[NamedType], Type[NamedAnnotationsType],
              Type[SlotsType], Type[type]]


def aioloop(): return noexception(RuntimeError, get_running_loop)


def allin(origin, destination):
    """
    Checks all items in origin are in destination iterable.

    Examples:
        >>> from rc import allin
        >>> from rc import Es
        >>>
        >>> class Int(int):
        ...     pass
        >>> allin(tuple.__mro__, Es.builtins_classes)
        True
        >>> allin(Int.__mro__, Es.builtins_classes)
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
        >>> from rc import annotations
        >>> from rc import pretty_install
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
        >>> from rc import Es
        >>>
        >>> class Int(int):
        ...     pass
        >>> anyin(tuple.__mro__, Es.builtins_classes)
        <class 'tuple'>
        >>> assert anyin('tuple int', Es.builtins_classes) is None
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


@singledispatchmethod
def asdict(data: Semaphore, convert=True):
    return dict(locked=data.locked(), value=data._value) if convert else data


@asdict.register
def asdict_chain(data: Chain, convert=True):
    data.rv = ChainRV.FIRST
    return dict(data) if convert else data


@asdict.register
def asdict_enum(data: Enum, convert=True):
    return {data.name: data.value} if convert else data


@asdict.register
def asdict_environs(data: Environs, convert=True):
    return data.dump() if convert else data


@asdict.register
def asdict_gettype(data: GetType, convert=True):
    return data if convert else data


@asdict.register
def asdict_gitsymbolic(data: GitSymbolicReference, convert=True):
    return dict(repo=data.repo, path=data.path) if convert else data


@asdict.register
def asdict_logger(data: Logger, convert=True):
    return dict(name=data.name, level=data.level) if convert else data


@asdict.register
def asdict_namedtype(data: NamedType, convert=True):
    return data._asdict() if convert else data


@asdict.register
def asdict_remote(data: Remote, convert=True):
    return dict(repo=data.repo, name=data.name) if convert else data


def asdict_ignorestr(data, convert=True):
    return str(data) if convert else data


def asdict_props(data, key=Attr.PUBLIC, pprop=False):
    """
    Properties to dict.

    Examples:
        >>> from rc import asdict_props
        >>> from rc import Es
        >>> from rc import pretty_install
        >>>
        >>> pretty_install()
        >>>

    Args:
        data: object to get properties.
        key: startswith to include.
        pprop: convert only :class:`rc.pproperty` or all properties excluding __ignore_attr__.

    Returns:
        Dict with names and values for properties.
    """
    pass


def asdict_type(data, convert=True):
    rv = data
    es = Es(data)
    if es.enum_sub:
        rv = {key: value._value_ for key, value in data.__members__.items()} if convert else data
    elif es.datatype_sub or es.dicttype_sub or es.namedtype_sub or es.slotstype_sub:
        rv = info(data).defaults if convert else data
    return rv


def classify_cls(data):
    """
    Classify Object Class or Class.

    Examples:
        >>> from rc import classify_cls
        >>> from rc import pretty_install
        >>>
        >>> pretty_install()
        >>>

    Args:
        data: any object.

    Returns:
        Attribute.
    """
    def attribute(defining, kind, field, name, obj):
        e = Es(obj)
        f = Es(field)
        return Attribute(coro=e.coro, defining=defining, es=e, field=field, hint=hints.get(name),
                         kind=kind, name=name, object=obj)
    es = Es(data)
    fields = data.__dataclass_fields__ if es.datatype_sub or es.datatype else {}
    data = data if es.type else data.__class__
    hints = annotations(data, stack=2) if es.annotationstype_sub else dict()
    classified = dict()
    for item in classify_class_attrs(data):
        classified |= {item.name: attribute(item.defining_class, Kind[item.kind.split(' ')[0].upper()],
                                            fields.pop(item.name, None), item.name, item.object)}
    for key, value in fields:
        cls = data
        for C in cls.__mro__:
            if Es(C).datatype_sub and key in C.__dataclass_fields__:
                cls = C
        key_obj = getattr(cls, key)
        classified |= {key: attribute(cls, Kind.DATA, value, key, key_obj)}
    return dict_sort(classified)


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
        >>> from rc import effect
        >>> from rc import pretty_install
        >>> from rc import Simple
        >>>
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


# noinspection PyDataclass
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
   """
    __slots__ = ('_data', '_depth', '_ignore', '_key',)
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

    def annotations(self, stack=2):
        """
        Object Annotations.

        Examples:
            >>> from dataclasses import dataclass
            >>> from dataclasses import InitVar
            >>> from typing import ClassVar
            >>> from rc import info
            >>>
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

    # noinspection PyUnusedLocal
    def asdict(self, count: int = ..., defaults: bool = ...):
        """
        Dict excluding.

        Returns:
            dict:
        """
        # convert = self.depth is None or self.depth > 1
        # # if self.es().enumenuminstance:
        # #     self.data = {self.data.name: self.data.value}
        # # elif self.es().namedtuple:
        # #     self.data = self.data._asdict().copy()
        # # elif self.instance(self.getmroinsattr('__ignore_str__')) and convert:
        # #     self.data = str(self.data)
        # # elif self.instance(Semaphore) and convert:
        # #     self.data = dict(locked=self.data.locked(), value=self.data._value)
        # # elif self.instance(GitSymbolicReference) and convert:
        # #     self.data = dict(repo=self.data.repo, path=self.data.path)
        # # elif self.instance(Remote) and convert:
        # #     self.data = dict(repo=self.data.repo, name=self.data.name)
        # # elif self.instance(Environs) and convert:
        # #     self.data = self.data.dump()
        # # elif self.instance(Logger) and convert:
        # #     self.data = dict(name=self.data.name, level=self.data.level)
        # # if self.enumenumcls:
        # #     self.data = {key: value._value_ for key, value in self.getcls.__members__.items()}
        # # elif self.chainmap and convert:
        # #     self.data.rv = ChainRV.FIRST
        # #     self.data = dict(self.data).copy()
        # if any([self.dataclass, self.dataclass_instance, self.dict_cls, self.dict_instance, self.slots_cls,
        #           self.slots_instance]):
        #     self.data = self.defaults if defaults else self.defaults | self.vars
        # # elif self.mutablemapping and convert:
        # #     self.data = dict(self.data).copy()
        # if self.mlst:
        #     rv = dict() if (mm := self.mutablemapping) else list()
        #     for key in self.data:
        #         value = get(self.data, key) if mm else key
        #         if value:
        #             if (inc := self.include(key, self.data if mm else None)) is None:
        #                 continue
        #             else:
        #                 value = inc[1]
        #                 if self.depth is None or count < self.depth:
        #                     value = self.new(value).asdict(count=count + 1, defaults=defaults)
        #         rv.update({key: value}) if mm else rv.append(value)
        #     return rv if mm else type(self.data)(rv)
        # if (inc := self.include(self.data)) is not None:
        #     if self.getsetdescriptor() or self.iscoro or isinstance(inc[1], \
        #     (*self.getmroinsattr('__ignore_copy__'),)) \
        #             or (self.depth is not None and self.depth > 1):
        #         return inc[1]
        #     try:
        #         return deepcopy(inc[1])
        #     except TypeError as exception:
        #         if "cannot pickle '_thread.lock' object" == str(exception):
        #             return inc[1]
        return self.data

    def attr_value(self, name, default=None): return getattr(self.data, name, default)

    @property
    def attrs(self) -> list:
        """
        Attrs including properties if not self.ignore.

        Excludes:
            __ignore_attr__
            __ignore_kwarg__ if not self.ignore.

        Returns:
            list:
        """
        return sorted([attr for attr in {*self.attrs_cls, *self.cls.memberdescriptor,
                                         *(vars(self.data) if self.es().datatype or self.es().datatype_sub else []),
                                         *(self.cls.prop if not self.ignore else [])}
                       if self._include_attr(attr, self.cls.callable) and attr not in self.cls.setter])

    @property
    def attrs_cls(self):
        attrs = {item for item in self.cls.dir if
                 self._include_attr(item) and item in self.cls.data_attrs and item}
        if self.cls.es.datatype_sub:
            _ = {attrs.add(item.name) for item in datafields(self.data) if self._include_attr(item.name)}
        return sorted(list(attrs))

    @property
    def cls(self): return Cls(data=self.data, ignore=self.ignore, key=self.key)

    @property
    def coro(self): return [i.name for i in self.cls.classified if Es(i.object).coro] + self.coros_property

    @property
    def coro_pproperty(self): return [i.name for i in self.cls.classified if Es(i.object).pproperty and
                                      Es(object.__getattribute__(self.data, i.name)).coro]

    @property
    def coro_prop(self): return [i.name for i in self.cls.classified if Es(i.object).prop and
                                 Es(object.__getattribute__(self.data, i.name)).coro]

    data = property(
        lambda self: object.__getattribute__(self, '_data'),
        lambda self, value: object.__setattr__(self, '_data', value),
        lambda self: object.__setattr__(self, '_data', None)
    )

    @property
    def defaults(self):
        """Class defaults."""

        def is_missing(default: str) -> bool:
            return isinstance(default, MISSING_TYPE)

        rv = dict()
        rv_data = dict()
        attrs = self.attrs_cls
        if self.cls.es.datatype_sub:
            rv_data = {f.name: f.default if is_missing(
                f.default) and is_missing(f.default_factory) else f.default if is_missing(
                f.default_factory) else f.default_factory() for f in datafields(self.data) if f.name in attrs}
        if self.cls.es.namedtype_sub:
            rv = self.cls.data._field_defaults
        elif self.cls.es.dicttype_sub or self.cls.es.slotstype_sub:
            rv = {key: inc[1] for key in attrs if (inc := self.include(key, self.data)) is not None}
        return rv | rv_data

    depth = property(
        lambda self: object.__getattribute__(self, '_depth'),
        lambda self, value: object.__setattr__(self, '_depth', value),
        lambda self: object.__setattr__(self, '_data', None)
    )

    @property
    def dir(self):
        return set(self.cls.dir + [i for i in dir(self.data) if self.key.include(i)])

    def es(self, data=None): return Es(data or self.data)

    def has_attr(self, name): return self.cls.has_attr(name=name) or hasattr(self.data, name)

    def has_method(self, name): return self.cls.has_method(name=name) or has_method(self.data, name)

    @property
    def has_reduce(self): return self.cls.has_reduce or has_reduce(self.data)

    @property
    def hash(self):
        return hash(tuple(map(lambda x: getset(self.data, x), Mro.hash_exclude.slotsinclude(self.data))))

    ignore = property(
        lambda self: object.__getattribute__(self, '_ignore'),
        lambda self, value: object.__setattr__(self, '_ignore', value),
        lambda self: object.__setattr__(self, '_ignore', False)
    )

    @property
    def ignore_attr(self): return

    def _include_attr(self, name, exclude=tuple()):
        ignore = {*Mro.ignore_attr.val(self.data), *(Mro.ignore_kwarg.val(self.data) if self.ignore else set()),
                  *exclude, *self.cls.initvar}
        return not any([not self.key.include(name), name in ignore, f'_{self.cls.name}' in name])

    def _include_exclude(self, data, key=True):
        import typing
        i = info(data)
        call = (Environs,)
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
    def initvarsdict(self):
        return getattr(self.data, '__init_vars__', dict())

    def is_attr(self, name): return self.cls.is_attr(name) or name in self().dir

    def is_coro(self, name): return name in self().coros

    def is_coro_pproperty(self, name): return name in self().coros_pproperty

    def is_coro_property(self, name): return name in self().coros_property

    key = property(
        lambda self: object.__getattribute__(self, '_key'),
        lambda self, value: object.__setattr__(self, '_key', value),
        lambda self: object.__setattr__(self, '_key', Attr.PRIVATE)
    )

    @property
    def keys(self):
        """
        Keys from kwargs to init class (not InitVars), exclude __ignore_kwarg__ and properties.

        Returns:
            list:
        """
        return sorted(list(self.kwargs.keys()))

    @property
    def kwargs(self):
        """
        Kwargs to init class with python objects no recursive, exclude __ignore_kwarg__ and properties.

        Includes InitVars.

        Example: Mongo binary.

        Returns:
            dict:
        """
        ignore = self.ignore
        self.ignore = True
        rv = {key: get(self.data, key) for key in self.attrs_cls} | \
             {key: value for key, value in self.initvarsdict.items()
              if key not in {*Mro.ignore_attr.val(self.data), *Mro.ignore_kwarg.val(self.data)}}
        self.ignore = ignore
        return rv

    @property
    def kwargs_dict(self) -> dict:
        """
        Kwargs recursive to init class with python objects as dict, asdict excluding __ignore_kwarg__ and properties.

        Example: Mongo asdict.

        Returns:
            dict:
        """
        ignore = self.ignore
        self.ignore = True
        rv = self.asdict()
        self.ignore = ignore
        return rv

    @property
    def module(self): return getmodule(self.data)

    @property
    def public(self):
        self.key = Attr.PUBLIC
        return self.asdict()

    @property
    def repr(self):
        attrs = Mro.repr_exclude.slotsinclude(self.data)
        attrs.update(self.cls.pproperty if Mro.repr_pproperty.first(self.data) else list())
        r = [f"{s}: {getset(self.data, s)}" for s in sorted(attrs) if s and not self.is_coro(s)]
        new = f',{NEWLINE if Mro.repr_newline.first(self.data) else " "}'
        return f'{self.cls.name}({new.join(r)})'

    @property
    def to_json(self, regenerate=True, indent=4, keys=True, max_depth=-1):
        return jsonpickle.encode(self.data, unpicklable=regenerate, indent=indent, keys=keys, max_depth=max_depth)

    def to_obj(self, keys=True): return jsonpickle.decode(self.data, keys=keys)

    @property
    def values(self):
        """
        Init python objects kwargs values no properties and not __ignore_kwarg__.

        Returns:
            list:
        """
        return list(self.kwargs.values())

    @property
    def values_dict(self):
        """
        Init python objects as dict kwargs values no properties and not __ignore_kwarg__.

        Returns:
            list:
        """
        return list(self.kwargs_dict.values())

    @property
    def vars(self):
        attrs = self.attrs
        return {key: inc[1] for key in attrs if (inc := self.include(key, self.data)) is not None}


def is_even(number): return Es(number).even


def iscoro(data):
    """
    Check all async options for object.

    Examples:
        >>> from rc import Es
        >>> from rc import iscoro
        >>> from rc import pretty_install
        >>>
        >>> pretty_install()
        >>>

    Args:
        data: any object.

    Returns:
        IsCoro.
    """


def in_dict(data, items=None, **kwargs):
    """
    Is Item in Dict?.

    Examples:
        >>> from rc import Es
        >>> from rc import in_dict
        >>> in_dict(globals(), {'True': True, 'False': False})
        True
        >>> in_dict(globals()['__builtins__'], {'True': True, 'False': False}, __name__='builtins')
        True
        >>> in_dict(globals(), {'True': True, 'False': False}, __name__='builtins')
        True
        >>> Es(Es.builtins).builtin
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


def map_reduce_even(iterable): return map_reduce(iterable, keyfunc=is_even)


def map_with_args(data, func, /, *args, pred=lambda x: True if x else False, split=' ', **kwargs):
    """
    Apply pred/filter to data and map with args and kwargs.

    Examples:
        >>> from rc import map_with_args
        >>> from rc import pretty_install
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
        >>> from rc import newprop
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
    # return property(
    #     lambda self:
    #     self.__getattribute__(f'_{name}', default=default(self) \
    #     if Es(default).instance(Callable, partial) else default),
    #     lambda self, value: self.__setattr__(f'_{name}', value),
    #     lambda self: self.__delattr__(f'_{name}')
    # )


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


class pproperty(property):
    """
    Print property.

    Examples:
        >>> from functools import cache
        >>> from rc import Es
        >>> from rc import pproperty
        >>> from rc import pretty_install
        >>>
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


def runinloop(data):
    """
    Run in loop if loop is running otherwise start loop and run.

    Examples:
        >>> from rc import Es
        >>> from rc import pretty_install
        >>> from rc import runinloop
        >>>
        >>> pretty_install()
        >>>

    Args:
        data: any object.

    Returns:
        IsCoro.
    """

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
        >>> from rc import Mro
        >>> from rc import to_camel
        >>>
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
        >>> from rc import pretty_install
        >>> from rc import to_iter
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
        >>> from rc import pretty_install
        >>> from rc import yield_if
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
        >>> from rc import pretty_install
        >>> from rc import yield_if
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
class TestBase(Base):
    classvar: ClassVar[int] = 1
    initvar: InitVar[int] = 1
    __slots__ = ('_hash', '_prop', '_repr', '_slot', )
    __hash_exclude__ = ('_slot', )
    __repr_exclude__ = ('_repr', )
    prop = Cls.propnew('prop')

    async def method_async(self):
        pass

    @classmethod
    def clsmethod(cls):
        pass

    @staticmethod
    def static():
        pass

    @pproperty
    def pprop(self):
        return 'pprop'

    @pproperty
    async def pprop_async(self):
        return 'pprop_async'


@dataclass
class TestData:
    dataclass_classvar: ClassVar[str] = 'dataclass_classvar'
    dataclass_default_factory: Union[dict, str] = datafield(default_factory=dict, init=False)
    dataclass_default_factory_init: Union[dict, str] = datafield(default_factory=dict)
    dataclass_default: str = datafield(default='dataclass_default', init=False)
    dataclass_default_init: str = datafield(default='dataclass_default_init')
    dataclass_initvar: InitVar[str] = 'dataclass_initvar'
    dataclass_str: str = 'dataclass_integer'

    def __post_init__(self, dataclass_initvar):
        pass


class TestDataDictMix(TestData):
    subclass_annotated_str: str = 'subclass_annotated_str'
    subclass_classvar: ClassVar[str] = 'subclass_classvar'
    subclass_str = 'subclass_str'

    def __init__(self, dataclass_initvar='dataclass_initvar_1', subclass_dynamic='subclass_dynamic'):
        super().__init__()
        super().__post_init__(dataclass_initvar=dataclass_initvar)
        self.subclass_dynamic = subclass_dynamic


class TestDataDictSlotMix(TestDataDictMix):
    __slots__ = ('_slot_property', 'slot', )

    def __init__(self, dataclass_initvar='dataclass_initvar_2', slot_property='slot_property', slot='slot'):
        super().__init__()
        super().__post_init__(dataclass_initvar=dataclass_initvar)
        self._slot_property = slot_property
        self.slot = slot

    @pproperty
    def slot_property(self):
        return self._slot_property
# </editor-fold>
