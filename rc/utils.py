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
    'POST_INIT_NAME',
    'PathLib',

    'BUILTIN_CLASSES',
    'FRAME_SYS_INIT',
    'FUNCTION_MODULE',
    'NEWLINE',
    'PYTHON_SYS',
    'PYTHON_SITE',

    'Alias',
    'console',
    'debug',
    'fmic',
    'fmicc',
    'ic',
    'icc',
    'POST_INIT_NAME',
    'pp',
    'print_exception',
    'RunningLoop',

    'pproperty',

    'Annotation',
    'AnnotationsType',
    'AsDictClassMethodType',
    'AsDictMethodType',
    'AsDictPropertyType',
    'AsDictStaticMethodType',
    'Attr',
    'Base',
    'Base1',
    'BoxKeys',
    'ChainRV',
    'Chain',
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
    'Name',
    'NamedType',
    'NamedAnnotationsType',
    'SlotsType',

    'aioloop',
    'allin',
    'annotations',
    'anyin',
    'Base1',
    'BoxKeys',
    'cmd',
    'cmdname',
    'current_task_name',
    'dict_sort',
    'dict_sort',
    'get',
    'info',
    'is_even',
    'join_newline',
    'map_reduce_even',
    'namedinit',
    'noexception',
    'prefixed',
    'repr_format',
    'startswith',
    'to_iter',
    'varname',

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

import ast
import re
import subprocess
import sys
import textwrap
import tokenize
from abc import ABCMeta
from abc import abstractmethod
from ast import AST
from ast import AsyncFor
from ast import AsyncFunctionDef
from ast import AsyncWith
from ast import Await
from ast import ClassDef
from ast import FunctionDef
from ast import get_source_segment
from ast import Import
from ast import ImportFrom
from ast import NodeVisitor
from ast import walk
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
from dataclasses import _POST_INIT_NAME
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
from inspect import stack
from operator import attrgetter
from pathlib import Path as PathLib
from subprocess import CompletedProcess
from types import BuiltinFunctionType
from types import CodeType
from types import FrameType
from types import FunctionType
from types import LambdaType
from types import ModuleType
from types import TracebackType
from typing import _alias
from typing import BinaryIO
from typing import Generator
from typing import get_args
from typing import get_origin
from typing import get_type_hints
from typing import IO
from typing import Iterable
from typing import Iterator
from typing import Literal
from typing import MutableMapping
from typing import NamedTuple
from typing import Optional
from typing import Protocol
from typing import runtime_checkable
from typing import TextIO
from typing import Union

from box import Box
from click import secho
from click.exceptions import Exit
from devtools import Debug
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
from more_itertools import always_iterable
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
console = Console(color_system='256')
debug = Debug(highlight=True)
fmic = IceCreamDebugger(prefix=str()).format
fmicc = IceCreamDebugger(prefix=str(), includeContext=True).format
ic = IceCreamDebugger(prefix=str())
icc = IceCreamDebugger(prefix=str(), includeContext=True)
POST_INIT_NAME = _POST_INIT_NAME
pp = console.print
print_exception = console.print_exception
pretty.install(console=console, expand_all=True)
# rich.traceback.install(console=console, extra_lines=5, show_locals=True)
RunningLoop = _RunningLoop


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


Annotation = namedtuple('Annotation', 'args cls hints key origin')


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
    __slots__: tuple
        slots

    Methods:
    --------
    __getattribute__(item, default=None)
        :class:`function`:  Sets ``None`` as default value is attr is not defined.
    get(cls, name, default=None)
        :class:`function`: Get attribute value.

    Examples:
    ---------
        >>> from rc import Base
        >>>
        >>> class Test(Base):
        ...     __slots__ = ('_hash', '_prop', '_repr', '_slot', )
        ...     __hash_exclude__ = ('_slot', )
        ...     __repr_exclude__ = ('_repr', )
        ...     prop = Base1.get_propnew('prop')
        >>>
        >>> test = Test()
    """
    __slots__ = ()

    def __getattribute__(self, name, default=None):
        """
        Sets attribute with default if it does not exists.

        Args:
            name: attr name.
            default: default value (default: None)

        Returns:
            Attribute value or sets with partial callable or sets default value
        """
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            object.__setattr__(self, name, default)
            return object.__getattribute__(self, name)

    def get(self, name, default=None):
        return self.__getattribute__(name, default=default)


class Base1:
    """
    Base1 Helper Class.

    Properties added with :meth:`rc.utils.Base1.get_propnew` must be added to :attr:`rc.utils.Base1.__slots__` with "_".

    Attributes:
    -----------
    __slots__: tuple
        slots
    __hash_exclude__: tuple
        mutables to be excluded for :func:`hash`
    __repr_exclude__: tuple
        mutables to be excluded for :func:`repr`
    get_clsname: str
        class name

    Methods:
    --------
    __getattribute__(item, default=None)
        Sets ``None`` as default value is attr is not defined.
    get_mroattr(cls, n='__slots__')
        :class:`staticmethod`: All values of atribute `cls.__slots__` in ``cls.__mro__``.
    get_mrohash()
        :class:`classmethod`: All values of atribute `cls.__hash__` in ``cls.__mro__``.
    get_mrorepr()
        :class:`classmethod`: All values of atribute `cls.__repr__` in ``cls.__mro__``.
    get_mroslots()
        :class:`classmethod`: All values of atribute `cls.__slots__` in ``cls.__mro__``.
    get_propnew(n, d=None)
        :class:`staticmethod`: Get a new :class:`property` with f'_{n}' as attribute name.
        It should be included in :attr:`rc.utils.Base1.__slots__`.
        If default is a partial instance, the returned value will be used. It is similar to a cache property:

    Examples:
    ---------
        >>> from rc import Base1
        >>>
        >>> class Test(Base1):
        ...     __slots__ = ('_hash', '_prop', '_repr', '_slot', )
        ...     __hash_exclude__ = ('_slot', )
        ...     __repr_exclude__ = ('_repr', )
        ...     prop = Base1.get_propnew('prop')
        >>>
        >>> test = Test()
        >>> test.cls_name
        'Test'
        >>> assert Test.get_mroattr(Test) == set(Test.__slots__) == test.get_mroslots()

        >>> assert Test.get_mroattr(Test, '__hash_exclude__') == set(Test.__hash_exclude__)
        >>> assert Test.__hash_exclude__ not in Test.get_mrohash()

        >>> assert Test.get_mroattr(Test, '__repr_exclude__') == set(Test.__repr_exclude__)
        >>> assert Test.__repr_exclude__[0] not in Test.get_mrorepr()
        >>> assert Test.__repr_exclude__[0] not in repr(test)
        >>> assert Test.__hash_exclude__[0] in repr(test)
        >>> assert test.cls_name in repr(test)

        >>> assert test.prop is None
        >>> test.prop = 2
        >>> test.prop
        2
        >>> del test.prop
        >>> assert test.prop is None

    """
    __slots__ = ()
    __hash_exclude__ = ()
    __repr_exclude__ = ()

    def __getattribute__(self, name: str, default=None):
        """
        Sets attribute with default if it does not exists.
        If default is a partial instance th returned value will be used. It is similar to a cache property:

        Args:
            name: attr name.
            default: default value (default: None)

        Returns:
            Attribute value or sets with partial callable or sets default value
        """
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            object.__setattr__(self, name, default() if isinstance(default, partial) else default)
            return object.__getattribute__(self, name)

    def __hash__(self):
        return hash(tuple(map(lambda x: self.__getattribute__(x), self.get_mrohash())))

    def __repr__(self):
        return f'{self.cls_name}({", ".join([f"{s}: {repr(getattr(self, s))}" for s in self.get_mrorepr()])})'

    def get(self, name, default=None):
        return self.__getattribute__(name, default=default)

    @property
    def cls_name(self):
        return self.__class__.__name__

    get_mroattr = staticmethod(lambda cls, n='__slots__': {a for i in cls.__mro__ for a in getattr(i, n, tuple())})
    get_mrohash = classmethod(lambda cls: cls.get_mroslots().difference(cls.get_mroattr(cls, n='__hash_exclude__')))
    get_mrorepr = classmethod(lambda cls:
                              sorted(cls.get_mroslots().difference(cls.get_mroattr(cls, n='__repr_exclude__'))))
    get_mroslots = classmethod(lambda cls: cls.get_mroattr(cls))
    get_propnew = staticmethod(lambda name, default=None: property(
        lambda self: self.__getattribute__(f'_{name}',
                                           default=default(self) if isinstance(default, partial) else default),
        lambda self, value: self.__setattr__(f'_{name}', value),
        lambda self: self.__delattr__(f'_{name}')
    ))


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
        lambda cls, C: cls is DataType and '__annotations__' in C.__dict__ and '__dataclass_fields__' in C.__dict__)


class DictType(metaclass=ABCMeta):
    """
    Dict Type.

    Examples:
        >>> class Dict: a = 1
        >>> class Slots: a = 1; __slots__ = ()
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
    datatype = property(lambda self: isinstance(self.data, DataType))
    datatype_sub = property(lambda self: issubclass(self.data, DataType))
    defaultdict = property(lambda self: isinstance(self.data, defaultdict))
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
    installed = property(lambda self: is_installed(self.data))
    instance = lambda self, *args: isinstance(self.data, args)
    int = property(lambda self: isinstance(self.data, int))
    io = property(lambda self: isinstance(self.data, IO))
    iterable = property(lambda self: isinstance(self.data, Iterable))
    iterator = property(lambda self: isinstance(self.data, Iterator))
    lambdatype = property(lambda self: isinstance(self.data, LambdaType))
    list = property(lambda self: isinstance(self.data, list))
    lst = property(lambda self: isinstance(self.data, (list, set, tuple)))
    method = property(
        lambda self:
        callable(self.data) and not type(self)(self.data).instance(classmethod, property, property, staticmethod))
    mlst = property(lambda self: isinstance(self.data, (MutableMapping, list, set, tuple)))
    mm = property(lambda self: isinstance(self.data, MutableMapping))
    moduletype = property(lambda self: isinstance(self.data, ModuleType))
    module_function = property(lambda self: is_module_function(self.data))
    noncomplex = property(lambda self: is_noncomplex(self.data))
    namedtype = property(lambda self: isinstance(self.data, NamedType))
    namedtype_sub = property(lambda self: issubclass(self.data, NamedType))
    named_annotationstype = property(lambda self: isinstance(self.data, NamedAnnotationsType))
    named_annotationstype_sub = property(lambda self: issubclass(self.data, NamedAnnotationsType))
    object = property(lambda self: is_object(self.data))
    pathlib = property(lambda self: isinstance(self.data, PathLib))
    picklable = lambda self, name: is_picklable(name, self.data)
    primitive = property(lambda self: is_primitive(self.data))
    prop = property(lambda self: isinstance(self.data, property))
    pproperty = property(lambda self: isinstance(self.data, pproperty))
    reducible = property(lambda self: is_reducible(self.data))
    reducible_sequence_subclass = property(lambda self: is_reducible_sequence_subclass(self.data))
    sequence = property(lambda self: is_sequence(self.data))
    sequence_subclass = property(lambda self: is_sequence_subclass(self.data))
    slotstype = property(lambda self: isinstance(self.data, SlotsType))
    slotstype_sub = property(lambda self: issubclass(self.data, SlotsType))
    staticmethod = property(lambda self: isinstance(self.data, staticmethod))
    source: property(lambda self: None)
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
        return cls is GetAttrType and \
               ('get' not in C.__dict__ or ('get' in C.__dict__ and not callable(C.__dict__['get'])))


@runtime_checkable
class GetSupport(Protocol):
    """An ABC with one abstract method get."""
    __slots__ = ()

    @abstractmethod
    def get(self, name, default=None):
        return self, name, default


class GetType(metaclass=ABCMeta):
    """
    Dict Type.

    Examples:
        >>> class Dict: a = 1
        >>> class Get: a = 1; get = lambda self, item: self.__getattribute__(item)
        >>> class Slots: a = 1; __slots__ = ()
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
    _hash_exclude0 = auto()
    _ignore_attr0 = auto()
    _ignore_str0 = auto()
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
    _repr_exclude0 = auto()
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
            Checks if has attr  with real name for private and public which conflicts with Enum.
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


class SlotsType(metaclass=ABCMeta):
    """
    Slots Type.

    Examples:
        >>> class Dict: a = 1
        >>> class Slots: a = 1; __slots__ = ()
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


def annotations(obj, index=1):
    """
    Formats obj annotations.

    Args:
        obj: obj.
        index: to extract globals and locals.

    Returns:
        Annotation: obj annotations.
    """
    frame = stack()[index].frame
    if a := noexception(TypeError, get_type_hints, obj, globalns=frame.f_globals, localns=frame.f_locals):
        return {name: Annotation(list(get_args(a[name])), cls, a, name, get_origin(a[name]))
                for name, cls in a.items()}
    return dict()


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


def get(data, name, default=None): return data.get(name, default) if Es(data).gettype else getattr(data, name, default)


class info:
    """
    Is Instance, etc. Helper Class

    Attributes:
    -----------
    data: Any
        object to provide information (default: None)
    depth: Optional[int]
        recursion depth (default: None)
    ignore: bool
        ignore properties (default: False)
    key: :class:`rc.Key`
        keys to include (default: :attr:`rc.Key.PRIVATE`)

    Examples:
        >>> from rc.utils import Base1
        >>> from rc.utils import info
        >>>
        >>> base = info(Base1)
        >>> assert 'cls_name' in base.cls_properties
        >>> i = info(info)
        >>> assert not set(info.__slots__).difference(i.cls_data)
        >>> assert len(base.cls_classmethods) == 3
        >>> assert len(base.cls_methods) == 1
        >>> assert len(base.cls_properties) == 1
        >>> assert len(base.cls_staticmethods) == 2
        >>> assert Base1.get.__name__ in base.cls_methods
        >>> assert len(base.cls_methods) == 1
        >>> assert len(base.cls_dir) == len(base.cls_classmethods) + len(base.cls_methods) + \
        len(base.cls_staticmethods) + len(base.cls_properties) + len(base.cls_data)
        >>> base_all = info(Base1, Attr.ALL)
        >>> assert len(base_all.cls_methods) == 19
        >>> base.get_importable_name
        'rc.utils.Base1'
        >>> base.cls_module_var
        'rc.utils'
        >>> base.cls_qual_var
        'Base1'
        >>> i.cls_attr_value('key')
        <member 'key' of 'info' objects>
        >>> i.get_module
        <module 'rc.utils' from '/Users/jose/bashrc/rc/utils.py'>
    """
    __slots__ = ('data', 'es', 'key', )

    def __init__(self, data=None, key=Attr.PRIVATE):
        self.data = data
        self.es = Es(self.data)
        self.key = key

    def __call__(self, data=None, key=None):
        self.data = data or self.data
        self.es = Es(self.data)
        self.key = key or self.key
        return self

    @property
    def cls(self):
        return self.data if self.es.type else type(self.data)

    @cache
    def cls_attr_value(self, name, default=None):
        return getattr(self.cls, name, default)

    @property
    @cache
    def cls_by_kind(self):
        return bucket(self.cls_classified, key=lambda x: x.kind if self.key.include(x.name) else 'e')

    @property
    @cache
    def cls_by_name(self):
        return {i.name: i for i in self.cls_classified if self.key.include(i.name)}

    @property
    @cache
    def cls_callables(self):
        return sorted(self.cls_classmethods + self.cls_methods + self.cls_staticmethods)

    @property
    @cache
    def cls_classified(self):
        return classify_class_attrs(self.cls)

    @property
    @cache
    def cls_classmethods(self):
        return list(map(Name.name_.getter, self.cls_by_kind['class method']))

    @property
    @cache
    def cls_data(self):
        return list(map(Name.name_.getter, self.cls_by_kind['data']))

    @property
    @cache
    def cls_dir(self):
        return [i for i in dir(self.cls) if self.key.include(i)]

    @property
    @cache
    def cls_methods(self):
        return list(map(Name.name_.getter, self.cls_by_kind['method']))

    @property
    def cls_module_var(self):
        return getattr(self.cls, '__module__', str())

    @property
    def cls_name(self):
        return self.cls.__name__

    @property
    @cache
    def cls_properties(self): return list(map(Name.name_.getter, self.cls_by_kind['property']))

    @property
    def cls_qual_var(self):
        return getattr(self.cls, '__qualname__', str())

    @property
    @cache
    def cls_staticmethods(self):
        return list(map(Name.name_.getter, self.cls_by_kind['static method']))

    get_dir = property(lambda self: list({self.cls_dir + self.get_dirinstance}))
    get_dirinstance = property(lambda self: [i for i in dir(self.data)if self.key.include(i)])
    get_importable_name = property(lambda self: importable_name(self.cls))
    get_module = property(lambda self: getmodule(self.data))
    get_mro = property(lambda self: self.cls.__mro__)
    get_mroattrins = lambda self, name='__ignore_attr__': {
        a for i in (info(), self.data) for a in {*getattr(i, name, list()), *Base1.get_mroattr(i.__class__, name)}}

    has_attr = lambda self, name='__slots__': hasattr(self.data, name)
    has_method = lambda self, name: has_method(self.data, name)
    has_reduce = property(lambda self: has_reduce(self.data))
    in_slot = lambda self, name='__slots__': name in Base1.get_mroattr(self.cls)

    @property
    def get_source(self):
        # if self.is_file
        # OSError
        return self

    @property
    def get_node(self):
        return self

    @property
    def get_file(self):
        return self

    @property
    def get_function(self):
        return self

    @property
    def get_lineno(self):
        return self

    @property
    def get_code(self):
        return self

    @property
    def get_package(self):
        return self

    @property
    def get_name(self):
        return self

    @property
    def get_var(self):
        return self


def is_even(number): return Es(number).even


def join_newline(data): return NEWLINE.join(data)


def map_reduce_even(iterable): return map_reduce(iterable, keyfunc=is_even)


def namedinit(cls, optional=True, **kwargs):
    """
    Init with defaults a NamedTuple.

    Examples:
        >>> from pathlib import Path
        >>> from typing import NamedTuple
        >>>
        >>> from rc import namedinit
        >>>
        >>> NoInitValue = NamedTuple('NoInitValue', var=str)

        >>> A = NamedTuple('A', module=str, path=Optional[Path], test=Optional[NoInitValue])
        >>> namedinit(A, optional=False)
        {'module': '', 'path': None, 'test': None}
        >>> namedinit(A)
        {'module': '', 'path': PosixPath('.'), 'test': None}
        >>> namedinit(A, test=NoInitValue('test'))
        {'module': '', 'path': PosixPath('.'), 'test': NoInitValue(var='test')}
        >>> namedinit(A, optional=False, test=NoInitValue('test'))
        {'module': '', 'path': None, 'test': NoInitValue(var='test')}

    Args:
        cls: NamedTuple cls.
        optional: True to use args[0] instead of None as default for Optional fallback to None if exception.
        **kwargs:

    Returns:
        cls: cls instance with default values.
    """
    rv = dict()
    for name, a in annotations(cls).items():
        value = None
        if v := kwargs.get(name):
            value = v
        elif a.origin == Literal:
            value = a.args[0]
        elif a.origin == Union and not optional:
            pass
        else:
            with suppress(Exception):
                value = (a.cls if a.origin is None else a.args[1] if a.args[0] is None else a.args[0])()
        rv[name] = value
    return rv


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


def to_iter(data, exclude=(str, bytes)):
    if isinstance(data, str):
        data = data.split(' ')
    return list(always_iterable(data, base_type=exclude))


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
        _stack = stack()
        func = _stack[index - 1].function
        index = index + 1 if func == POST_INIT_NAME else index
        if line := textwrap.dedent(_stack[index].code_context[0]):
            if var := re.sub(f'(.| ){func}.*', str(), line.split(' = ')[0].replace('assert ', str()).split(' ')[0]):
                return (var.lower() if lower else var).split(**split_sep(sep))[0]


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
