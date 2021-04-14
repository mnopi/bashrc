# -*- coding: utf-8 -*-
"""Utils Module."""
__all__ = (
    'PathLib',

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
    'pp',
    'print_exception',

    'AnnotationsType',
    'AsDictClassMethodType',
    'AsDictMethodType',
    'AsDictPropertyType',
    'AsDictStaticMethodType',
    'Base',
    'BoxKeys',
    'ChainRV',
    'Chain',
    'CmdError',
    'CmdAioError',
    'DataType',
    'DictType',
    'EnumDict',
    'Executor',
    'GetType',
    'Key',
    'Name',
    'NamedType',
    'NamedAnnotationsType',
    'SlotsType',

    'aioloop',
    'annotations',
    'Base',
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

import re
import subprocess
import sys
import textwrap
from abc import ABCMeta
from asyncio import current_task
from asyncio import get_running_loop
from asyncio import iscoroutine
from collections import ChainMap
from collections import defaultdict
from collections import namedtuple
from collections import OrderedDict
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import _POST_INIT_NAME
from enum import auto
from enum import Enum
from functools import cache
from functools import partial
from functools import singledispatch
from functools import singledispatchmethod
from inspect import classify_class_attrs
from inspect import FrameInfo
from inspect import getmodule
from inspect import isasyncgen
from inspect import isasyncgenfunction
from inspect import isawaitable
from inspect import iscoroutinefunction
from inspect import isgetsetdescriptor
from inspect import stack
from operator import attrgetter
from pathlib import Path as PathLib
from subprocess import CompletedProcess
from types import FrameType
from typing import _alias
from typing import Callable
from typing import Generator
from typing import get_args
from typing import get_origin
from typing import get_type_hints
from typing import Iterable
from typing import Iterator
from typing import Literal
from typing import MutableMapping
from typing import NamedTuple
from typing import Optional
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
from jsonpickle.util import is_module
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
from more_itertools import map_reduce
from rich import pretty
from rich.console import Console


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
pp = console.print
print_exception = console.print_exception
pretty.install(console=console, expand_all=True)
# rich.traceback.install(console=console, extra_lines=5, show_locals=True)


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
        >>> issubclass(named, AnnotationsType)
        False
        >>> isinstance(named(), AnnotationsType)
        False
        >>>
        >>> issubclass(Named, AnnotationsType)
        True
        >>> isinstance(Named(a='a'), AnnotationsType)
        True
    """
    __subclasshook__ = classmethod(lambda cls, C: cls is AnnotationsType and '__annotations__' in C.__dict__)


class AsDictClassMethodType(metaclass=ABCMeta):
    """
    AsDict Class Method Type.

    Examples:
        >>> from rc import AsDictClassMethodType
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
        >>> issubclass(AsDictClass, AsDictClassMethodType)
        True
        >>> isinstance(c, AsDictClassMethodType)
        True
        >>>
        >>> issubclass(AsDictMethod, AsDictClassMethodType)
        False
        >>> isinstance(m, AsDictClassMethodType)
        False
        >>>
        >>> issubclass(AsDictProperty, AsDictClassMethodType)
        False
        >>> isinstance(p, AsDictClassMethodType)
        False
        >>>
        >>> issubclass(AsDictStatic, AsDictClassMethodType)
        False
        >>> isinstance(s, AsDictClassMethodType)
        False
        """
    __subclasshook__ = classmethod(
        lambda cls, C:
        cls is AsDictClassMethodType and 'asdict' in C.__dict__ and info(C.__dict__['asdict']).is_classmethod)
    asdict = classmethod(lambda cls, *args, **kwargs: dict())


class AsDictMethodType(metaclass=ABCMeta):
    """AsDict Method Type."""
    __subclasshook__ = classmethod(
        lambda cls, C: cls is AsDictMethodType and 'asdict' in C.__dict__ and info(C.__dict__['asdict']).is_method)
    asdict = lambda self, *args, **kwargs: dict()


class AsDictPropertyType(metaclass=ABCMeta):
    """AsDict Property Type."""
    __subclasshook__ = classmethod(
        lambda cls, C: cls is AsDictPropertyType and 'asdict' in C.__dict__ and info(C.__dict__['asdict']).is_property)
    asdict = property(lambda self: dict())


class AsDictStaticMethodType(metaclass=ABCMeta):
    """AsDict Static Method Type."""
    __subclasshook__ = classmethod(
        lambda cls, C:
        cls is AsDictStaticMethodType and 'asdict' in C.__dict__ and info(C.__dict__['asdict']).is_staticmethod)
    asdict = staticmethod(lambda *args, **kwargs: dict())


class Base:
    """
    Base Helper Class.

    Properties added with :meth:`rc.utils.Base.get_propnew` must be added to :attr:`rc.utils.Base.__slots__` with "_".

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
        It should be included in :attr:`rc.utils.Base.__slots__`.
        If default is a partial instance, the returned value will be used. It is similar to a cache property:

    Examples:
    ---------
        >>> from rc import Base
        >>>
        >>> class Test(Base):
        >>>     __slots__ = ('_cache', '_hash', '_prop', '_repr', '_slot', )
        >>>     __hash_exclude__ = ('_slot', )
        >>>     __repr_exclude__ = ('_repr', )
        >>>     prop = Base.get_propnew('prop')
        >>>     cache =
        >>>
        >>> test = Test()
        >>> test.get_clsname
        'Test'
        >>> assert Test.get_mroattr(Test) == set(Test.__slots__) == test.get_mroslots()

        >>> assert Test.get_mroattr(Test, '__hash_exclude__') == set(Test.__hash_exclude__)
        >>> assert Test.__hash_exclude__ not in Test.get_mrohash()

        >>> assert Test.get_mroattr(Test, '__repr_exclude__') == set(Test.__repr_exclude__)
        >>> assert Test.__repr_exclude__[0] not in Test.get_mrorepr()
        >>> assert Test.__repr_exclude__[0] not in repr(test)
        >>> assert Test.__hash_exclude__[0] in repr(test)
        >>> assert test.get_clsname in repr(test)

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

        >>> from time import time
        >>> # noinspection PyAttributeOutsideInit
        >>> class A:
        >>>     @property
        >>>     def a(self):
        >>>         if hasattr(self, '_a'):
        >>>             return self._a
        >>>         self._a = time()

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
    __hash__ = lambda self: hash(tuple(map(lambda x: self.__getattribute__(x), self.get_mrohash())))
    __repr__ = lambda self: \
        f'{self.get_clsname}({", ".join([f"{s}: {repr(getattr(self, s))}" for s in self.get_mrorepr()])})'
    get = lambda self, name, default=None: self.__getattribute__(name, default=default)
    get_clsname = property(lambda self: self.__class__.__name__)
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
            if info(mapping).is_named_type:
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
        >>> issubclass(Data, DataType)
        True
        >>> isinstance(data, DataType)
        True
        >>> isinstance(Data, DataType)
        False
        >>>
        >>> issubclass(Dict, DataType)
        False
        >>> isinstance(d, DataType)
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
        >>> issubclass(Dict, DictType)
        True
        >>> isinstance(d, DictType)
        True
        >>>
        >>> issubclass(Slots, DictType)
        False
        >>> isinstance(s, DictType)
        False
    """
    __subclasshook__ = classmethod(lambda cls, C: cls is DictType and '__dict__' in C.__dict__)


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
        >>> issubclass(Dict, GetType)
        False
        >>> isinstance(d, GetType)
        False
        >>>
        >>> issubclass(Get, GetType)
        True
        >>> isinstance(g, GetType)
        True
        >>>
        >>> issubclass(dict, GetType)
        True
        >>> isinstance(dct, GetType)
        True
    """
    get = lambda self, name, default: getattr(self, name, default)
    __subclasshook__ = classmethod(
        lambda cls, C: cls is GetType and 'get' in C.__dict__ and callable(C.__dict__['get']))


class Key(Enum):
    """
    Include Key.

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
            Include Key.
    """
    ALL = auto()
    PRIVATE = '__'
    PUBLIC = '_'

    def include(self, obj) -> bool:
        """
        Include Key.

        Examples:
            >>> Key.PUBLIC.include('_hello')
            False
            >>> Key.PRIVATE.include('_hello')
            True
            >>> Key.ALL.include('__hello')
            True

        Args:
            obj: string

        Returns:
            True if key to be included.
        """
        if self is Key.ALL:
            return True
        return not obj.startswith(self.value)


class Name(Enum):
    __all = auto()
    __class = auto()
    __annotations = auto()
    __builtins = auto()
    __cached = auto()
    __contains = auto()
    __dataclass_fields = auto()
    __dataclass_params = auto()
    __delattr = auto()
    __dir = auto()
    __dict = auto()
    __doc = auto()
    __eq = auto()
    __file = auto()
    __getattribute = auto()
    __hash_exclude = auto()
    __ignore_attr = auto()
    __ignore_str = auto()
    __len = auto()
    __loader = auto()
    __members = auto()
    __module = auto()
    __mro = auto()
    __name = auto()
    __package = auto()
    __qualname = auto()
    __reduce = auto()
    __repr = auto()
    __repr_exclude = auto()
    __setattr = auto()
    __slots = auto()
    __spec = auto()
    __str = auto()
    _asdict = auto()
    add = auto()
    append = auto()
    asdict = auto()
    cls_ = auto()  # To avoid conflict with Name.cls
    clear = auto()
    copy = auto()
    count = auto()
    data = auto()
    endswith = auto()
    extend = auto()
    external = auto()
    f_code = auto()
    f_globals = auto()
    f_locals = auto()
    frame = auto()
    function = auto()
    get_ = auto()  # To avoid conflict with Name.get
    index = auto()
    item = auto()
    items = auto()
    keys = auto()
    kind = auto()
    lineno = auto()
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
    update = auto()
    value_ = auto()  # To avoid conflict with Enum.value
    values = auto()

    @singledispatchmethod
    def get(self, obj: GetType, default=None):
        if self is Name.function:
            return obj.get(self.__name.real, default)
        return obj.get(self.real, default)

    @get.register
    def _(self, obj: FrameInfo, default=None):
        if self is Name.f_code:
            return obj.frame.f_code
        if self is Name.function:
            return obj.function
        if self is Name.lineno:
            return obj.lineno
        if self is Name.__file:
            return PathLib(obj.filename)
        return default

    @get.register
    def _(self, obj: FrameType, default=None):
        if self is Name.f_code:
            return obj.f_code
        if self is Name.function:
            return obj.f_code.co_name
        if self is Name.lineno:
            return obj.f_lineno
        if self is Name.__file:
            return PathLib(obj.f_globals.get(self.real, default))

    @get.register
    def _(self, obj: DictType, default=None):
        try:
            return object.__getattribute__(obj, self.real)
        except AttributeError:
            return default

    @property
    @cache
    def getter(self): return attrgetter(self.real)

    def has(self, obj): return hasattr(obj, self.name)

    @classmethod
    @cache
    def attrs(cls):
        return map_reduce(cls.__members__, lambda x: x.startswith('__'), lambda x: cls._real(x))

    @classmethod
    @cache
    def private(cls): return cls.attrs()[True]

    @classmethod
    @cache
    def public(cls): return cls.attrs()[False]

    @classmethod
    @cache
    def _real(cls, name):
        return f'{name}__' if name.startswith('__') else name.removesuffix('_')

    @property
    def real(self):
        return self._real(self.name)


class NamedType(metaclass=ABCMeta):
    """
    named Type.

    Examples:
        >>> from collections import namedtuple
        >>>
        >>> named = namedtuple('named', 'a', defaults=('a', ))
        >>>
        >>> isinstance(named(), NamedType)
        True
        >>> issubclass(named, NamedType)
        True
        >>>
        >>> isinstance(named(), tuple)
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
        >>> isinstance(named(), NamedAnnotationsType)
        False
        >>> issubclass(named, NamedAnnotationsType)
        False
        >>>
        >>> isinstance(Named('a'), NamedAnnotationsType)
        True
        >>> issubclass(Named, NamedAnnotationsType)
        True
        >>>
        >>> isinstance(named(), tuple)
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
        >>> issubclass(Dict, SlotsType)
        False
        >>> isinstance(d, SlotsType)
        False
        >>>
        >>> issubclass(Slots, SlotsType)
        True
        >>> isinstance(s, SlotsType)
        True
    """
    __subclasshook__ = classmethod(lambda cls, C: cls is SlotsType and '__slots__' in C.__dict__)


def aioloop(): return noexception(RuntimeError, get_running_loop)


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
def _(data: list, key: Iterable = ('self', 'cls', )) -> Optional[list]:
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


def get(obj, name, d=None): return obj.get(name, d) if info(obj).is_get_type else getattr(obj, name, d)


class info(Base):
    """
    Is Instance, etc. Helper Class

    Attributes:
    -----------
    data: Any
        object to provide information (default: None)
    key: :class:`rc.Key`
        keys to include (default: :attr:`rc.Key.PRIVATE`)

    Examples:
        >>> from rc.utils import info
        >>>
    """
    __slots__ = ('_data', '_key', )

    def __init__(self, data=None, key=Key.PRIVATE): self.data = data; self.key = key
    def __call__(self, data=None, key=None): self.data = data or self.data; self.key = key or self.key; return self
    data = Base.get_propnew('data')
    key = Base.get_propnew('key')

    get_callables = property(lambda self: {
        i.name: i for i in self.get_inspect() if isinstance(i.object, Callable) and self.key.include(i.name)})
    get_classmethods = property(lambda self: {
        i.name: i for i in self.get_inspect() if 'class' in i.kind and self.key.include(i.name)})
    get_cls = property(lambda self: self.data if self.is_type else type(self.data))
    get_clsattr = lambda self, name, default=None: getattr(self.get_cls, name, default)
    get_clsmodule = property(lambda self: getattr(self.get_cls, '__module__', str()))
    get_clsname = property(lambda self: self.get_cls.__name__)
    get_properties = property(lambda self: {
        i.name: i for i in self.get_inspect() if 'property' in i.kind and self.key.include(i.name)})
    get_clsqual = property(lambda self: self.get_cls.__qualname__)
    get_dir = property(lambda self: list({self.get_dircls + self.get_dirinstance}))
    get_dircls = property(lambda self: [i for i in self.get_cls.__dir__() if self.key.include(i.name)])
    get_dirinstance = property(lambda self: [i for i in self.__dir__() if self.key.include(i.name)])
    get_inspect = property(lambda self: classify_class_attrs(self.cls))
    get_importable_name = property(lambda self: importable_name(self.get_cls))
    get_methods = property(lambda self: {
        i.name: i for i in self.get_inspect() if 'method' == i.kind and self.key.include(i.name)})
    get_module = property(lambda self: getmodule(self.data))
    get_mro = property(lambda self: self.get_cls.__mro__)
    get_mroattrins = lambda self, name='__ignore_attr__': {
        a for i in (info(), self.data) for a in {*getattr(i, name, list()), *Base.get_mroattr(i.__class__, name)}}
    get_staticmethods = property(lambda self: {
        i.name: i for i in self.get_inspect() if 'static' in i.kind and self.key.include(i.name)})
    has_attr = lambda self, name='__slots__': hasattr(self.data, name)
    has_method = lambda self, name: has_method(self.data, name)
    has_reduce = property(lambda self: has_reduce(self.data))
    in_slot = lambda self, name='__slots__': name in Base.get_mroattr(self.get_cls)
    is_annotations_type = property(lambda self: isinstance(self.data, AnnotationsType))
    is_annotations_type_sub = property(lambda self: issubclass(self.data, AnnotationsType))
    is_asdict_classmethod_type = property(lambda self: isinstance(self.data, AsDictClassMethodType))
    is_asdict_classmethod_type_sub = property(lambda self: issubclass(self.data, AsDictClassMethodType))
    is_asdict_method_type = property(lambda self: isinstance(self.data, AsDictMethodType))
    is_asdict_method_type_sub = property(lambda self: issubclass(self.data, AsDictMethodType))
    is_asdict_property_type = property(lambda self: isinstance(self.data, AsDictPropertyType))
    is_asdict_property_type_sub = property(lambda self: issubclass(self.data, AsDictPropertyType))
    is_asdict_staticmethod_type = property(lambda self: isinstance(self.data, AsDictStaticMethodType))
    is_asdict_staticmethod_type_sub = property(lambda self: issubclass(self.data, AsDictStaticMethodType))
    is_asyncgen = property(lambda self: isasyncgen(self.data))
    is_asyncgenfunc = property(lambda self: isasyncgenfunction(self.data))
    is_attr = lambda self, name: name in self.get_dir
    is_awaitable = property(lambda self: isawaitable(self.data))
    is_bool = property(lambda self: isinstance(self.data, int) and isinstance(self.data, bool))
    is_chain = property(lambda self: isinstance(self.data, Chain))
    is_chainmap = property(lambda self: isinstance(self.data, ChainMap))
    is_classmethod = property(lambda self: isinstance(self.data, classmethod))
    is_collections = property(lambda self: is_collections(self.data))
    is_coro = property(lambda self: any([self.is_asyncgen, self.is_asyncgenfunc, self.is_awaitable, self.is_coroutine,
                                         self.is_coroutinefunc]))
    is_coroutine = property(lambda self: iscoroutine(self.data))
    is_coroutinefunc = property(lambda self: iscoroutinefunction(self.data))
    is_data_type = property(lambda self: isinstance(self.data, DataType))
    is_data_type_sub = property(lambda self: issubclass(self.data, DataType))
    is_defaultdict = property(lambda self: isinstance(self.data, defaultdict))
    is_dict = property(lambda self: isinstance(self.data, dict))
    is_dict_type = property(lambda self: isinstance(self.data, DictType))
    is_dict_type_sub = property(lambda self: issubclass(self.data, DictType))
    is_dlst = property(lambda self: isinstance(self.data, (dict, list, set, tuple)))
    is_enum = property(lambda self: isinstance(self.data, Enum))
    is_enum_sub = property(lambda self: issubclass(self.data, Enum))
    is_enumdict = property(lambda self: isinstance(self.data, EnumDict))
    is_enumdict_sub = property(lambda self: issubclass(self.data, EnumDict))
    is_float = property(lambda self: isinstance(self.data, float))
    is_generator = property(lambda self: isinstance(self.data, Generator))
    is_get_type = property(lambda self: isinstance(self.data, GetType))
    is_get_type_sub = property(lambda self: issubclass(self.data, GetType))
    is_getsetdescriptor = lambda self, n: isgetsetdescriptor(self.get_clsattr(n)) if n else self.data
    is_hashable = property(lambda self: bool(noexception(TypeError, hash, self.data)))
    is_installed = property(lambda self: is_installed(self.data))
    is_instance = lambda self, *args: isinstance(self.data, args)
    is_int = property(lambda self: isinstance(self.data, int))
    is_iterable = property(lambda self: isinstance(self.data, Iterable))
    is_iterator = property(lambda self: isinstance(self.data, Iterator))
    is_list = property(lambda self: isinstance(self.data, list))
    is_lst = property(lambda self: isinstance(self.data, (list, set, tuple)))
    is_method = property(
        lambda self: callable(self.data) and not type(self)(self.data).is_instance(classmethod, property, staticmethod))
    is_mlst = property(lambda self: isinstance(self.data, (MutableMapping, list, set, tuple)))
    is_module = property(lambda self: is_module(self.data))
    is_module_function = property(lambda self: is_module_function(self.data))
    is_noncomplex = property(lambda self: is_noncomplex(self.data))
    is_named_type = property(lambda self: isinstance(self.data, NamedType))
    is_named_type_sub = property(lambda self: issubclass(self.data, NamedType))
    is_named_annotations_type = property(lambda self: isinstance(self.data, NamedAnnotationsType))
    is_named_annotations_type_sub = property(lambda self: issubclass(self.data, NamedAnnotationsType))
    is_object = property(lambda self: is_object(self.data))
    is_picklable = lambda self, name: is_picklable(name, self.data)
    is_primitive = property(lambda self: is_primitive(self.data))
    is_property = property(lambda self: isinstance(self.data, property))
    is_reducible = property(lambda self: is_reducible(self.data))
    is_reducible_sequence_subclass = property(lambda self: is_reducible_sequence_subclass(self.data))
    is_sequence = property(lambda self: is_sequence(self.data))
    is_sequence_subclass = property(lambda self: is_sequence_subclass(self.data))
    is_slots_type = property(lambda self: isinstance(self.data, SlotsType))
    is_slots_type_sub = property(lambda self: issubclass(self.data, SlotsType))
    is_staticmethod = property(lambda self: isinstance(self.data, staticmethod))
    is_tuple = property(lambda self: isinstance(self.data, tuple))
    is_type = property(lambda self: isinstance(self.data, type))
    is_unicode = property(lambda self: is_unicode(self.data))


def is_even(number: str): return not number % 2


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
        index = index + 1 if func == _POST_INIT_NAME else index
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
