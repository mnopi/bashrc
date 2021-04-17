# -*- coding: utf-8 -*-
"""Frame Module."""
__all__ = (
    'FRAME_INDEX',
    'FRAME_SYS_INIT',
    'FUNCTION_MODULE',
    'MODULE_MAIN',
    'PATHS_SYS',
    'PATHS_SYS_EXCL',
    'PATHS_EXCL',
    'FId',
    'File',
    'Function',
    'GlobalsModule',
    'Info',
    'IntervalBase',
    'Line',
    'Package',
    'Var',
    'GLOBALS_MODULE',
    'FrameId',
    'Frame',
    'MatchError',
    'Real',
    'Stack',
)

import asyncio
import importlib
import inspect
import sys
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
from ast import parse
from ast import walk
from asyncio import as_completed
from asyncio import create_task
from asyncio import ensure_future
from asyncio import gather
from asyncio import to_thread
from collections import namedtuple
from contextlib import suppress
from enum import Enum
from functools import cache
from operator import attrgetter
from operator import contains
from os import getenv
from sysconfig import get_paths
from typing import cast
from typing import NamedTuple
from typing import Optional

from icecream import ic
from intervaltree import Interval
from intervaltree import IntervalTree
from more_itertools import first_true

from .path import Path
from .path import PathIs
from .path import PathSuffix
from .utils import FRAME_SYS_INIT
from .utils import FUNCTION_MODULE
from .utils import Base
from .utils import delete
from .utils import dict_sort
from .utils import namedinit
from .utils import NEWLINE
from .utils import prefixed
from .utils import repr_format

FRAME_INDEX = 1
MODULE_MAIN = '__main__'
PATHS_SYS = {key: Path(value) for key, value in get_paths().items()}
PATHS_SYS_EXCL = [PATHS_SYS[_i] for _i in ['stdlib', 'purelib', 'include', 'platinclude', 'scripts']]
PATHS_EXCL = PATHS_SYS_EXCL + ([_i] if (_i := Path(getenv('PYCHARM_PLUGINS', '<None>'))).resolved.exists() else list())
FId = namedtuple('_FrameID', 'code decorator function parts real ASYNC')
File = namedtuple('File', 'include exists path')
Function = namedtuple('Function', 'ASYNC cls decorators module name qual')
GlobalsModule = namedtuple('GlobalsModule', 'all annotations builtins cached doc file loader name package repo pypi '
                                            'spec')
GLOBALS_MODULE = GlobalsModule(*[f'__{_i}__' for _i in GlobalsModule._fields])
Info = namedtuple('Info', 'ASYNC external internal module real')
IntervalBase = namedtuple('IntervalBase', 'begin end data')
IntervalType = Interval[IntervalBase]
Line = namedtuple('Line', 'ASYNC code lineno')
Package = NamedTuple('Package', init=Optional[Path], module=str, name=str, package=str, path=Optional[Path], prefix=str,
                     relative=Optional[Path], root=Optional[Path], var=Optional[GlobalsModule])
Var = namedtuple('Var', 'args globals locals')


class FrameId(Enum):
    ASYNCCONTEXTMANAGER = FId(code=str(), decorator=str(),
                              function='__aenter__', parts='contextlib.py', real=1, ASYNC=True)
    IMPORTLIB = FId(code=str(), decorator=str(),
                    function='_call_with_frames_removed', parts=f'<frozen {importlib._bootstrap.__name__}>',
                    real=5, ASYNC=False)
    RUN = FId(code=str(), decorator=str(),
              function='_run', parts='asyncio events.py', real=5, ASYNC=True)
    TO_THREAD = FId(code=str(), decorator=str(),
                    function='run', parts='concurrent futures thread.py', real=None, ASYNC=False)
    # FUNCDISPATCH = ('return funcs[Call().ASYNC]', 'wrapper bapy', 'core', 1)


class Frame(Base, NodeVisitor):
    __file = Path(__file__)
    __path = __file.path
    __package = __path.name

    __slots__ = ('exists', '_external', 'file', 'frame', 'include', 'internal', '_function', 'functions', 'lineno',
                 'lines', 'module', 'obj', 'source', 'stack')

    __hash_exclude__ = ()

    def __init__(self, file=None, frame=None, obj=None, source=str()):
        super().__init__()
        self.frame = frame
        self.file = Path(file, inspect.getsourcefile(self.frame or obj) or inspect.getfile(self.frame or obj))
        self.exists = self.file.resolved.exists()
        self.include = not (any(map(self.file.is_relative_to, PATHS_EXCL)) or not self.file.suffix)
        self.internal = self.file.parent == self.__file.parent and self.file.is_relative_to(self.__file.parent)
        self._external = False
        self._function = self.frame.f_code.co_name
        self.functions = IntervalTree()
        self.lineno = self.frame.f_lineno
        self.lines = dict()
        self.module = self._function == FUNCTION_MODULE
        self.obj = obj
        self.source = source or self.walk
        # self.stack = None

    def __repr__(self):
        return repr_format(self, 'file function info line package', clear=False, newline=True)

    @classmethod
    def distance(cls, lineno, value):
        distance = {lineno - item.begin: item.data for item in value}
        return Interval(min(distance.keys()), max(distance.keys()), distance)

    @property
    @cache
    def external(self):
        return self if self.info.external else None

    @property
    @cache
    def function(self):
        if self._module or not self.walk:
            return Function(False, str(), list(), self._module, self._function, self._function)

        if not (routines := self._functions[self._lineno]):
            raise MatchError(file=self.file, lineno=self._lineno, function=self._function)
        distance = self.distance(self._lineno, routines)
        distance_min = distance.data[distance.begin]
        if distance_min.name != self._function:
            raise MatchError(file=self.file, lineno=self._lineno, name=distance_min.name, function=self._function)
        return Function(isinstance(distance_min, AsyncFunctionDef), distance.data[distance.end].name,
                        [item.id for item in distance_min.decorator_list], self._module, self._function,
                        '.'.join([distance.data[item].name for item in sorted(distance.data, reverse=True)]))

    @property
    @cache
    def info(self):
        rv = None
        for i in FrameId:
            if i is FrameId.IMPORTLIB and self._lineno == 228:
                ic(i.value.function, self._function, i.value.function == self._function,
                   self._module, self._module is False,
                   self.file.exists, self.file.exists is False,
                   i.value.parts, self.file.path, self.file.path.text, self.file.path.has(i.value.parts))
            if i.value.function == self._function and \
                    any([(i.value.parts in self.file.path),
                         (self._module is False and self.file.exists is False and isinstance(i.value.parts, str) and
                          self.file.path.has(i.value.parts))]):
                rv = i
                break
        return Info(ASYNC=rv.value.ASYNC if rv else self.line.ASYNC or self.function.ASYNC,
                    external=self._external,
                    internal=self.internal,
                    module=self._module,
                    real=rv.value.real if rv else int() if self.file.include else None)

    @classmethod
    def interval(cls, node):
        if hasattr(node, 'lineno'):
            min_lineno = node.lineno
            max_lineno = node.lineno
            for node in walk(node):
                if hasattr(node, 'lineno'):
                    min_lineno = min(min_lineno, node.lineno)
                    max_lineno = max(max_lineno, node.lineno)
            return min_lineno, max_lineno + 1

    @property
    @cache
    def line(self):
        asy = False
        if self.walk:
            if not (line := self.lines[self._lineno]):
                raise MatchError(file=self.file, lineno=self._lineno)
            for node in line:
                code = get_source_segment(self.source, node)
                names = [as_completed, create_task, ensure_future, gather, asyncio]
                name = attrgetter('__name__')
                if isinstance(node, (AsyncFor, AsyncWith, Await,)) or \
                        (any([contains(code, name(item)) for item in names]) and name(to_thread) not in code):
                    asy = True
                    break
            return Line(
                code=str(max({get_source_segment(self.source, node) for node in line}, key=len)).split('\n'),
                lineno=self._lineno,
                ASYNC=asy,
            )
        return Line(asy, list(), self._lineno)

    @property
    @cache
    def package(self):
        if self.file.include and self.info.external:
            init = self.file.path.find_up(name='__init__.py').path
            path = init.parent if init else init.parent \
                if ((rv := self.file.path.find_up(PathIs.DIR, PathSuffix.GIT).path) and (rv / 'HEAD').exists()) \
                else self.file.path.parent

            root = path.parent
            relative = self.file.path.relative_to(root)
            var = GlobalsModule(*map(self.var.globals.get, GLOBALS_MODULE._asdict().values()))
            name = var.spec.name if var.spec else var.name \
                if var.name and var.name != MODULE_MAIN else \
                '.'.join([i.removesuffix(self.file.path.suffix) for i in self.file.path.relative_to(path.parent).parts])
            package = var.package or name.split('.')[0] or path.stem
            module = name.rpartition('.')[2]
            return Package(init=init, module=name.split('.')[-1] if '__init__' in module else module, name=path.stem,
                           package=package, path=path, prefix=prefixed(package), relative=relative, root=root,
                           var=var)
        return namedinit(Package, optional=False)

    @property
    @cache
    def var(self):
        locs = self._frame.f_locals.copy()
        arg, varargs, keywords = inspect.getargs(self._frame.f_code)
        return Var(args=delete({name: locs[name] for name in arg} | ({varargs: val} if (
            val := locs.get(varargs)) else dict()) | (kw if (kw := locs.get(keywords)) else dict())),
                   globals=self._frame.f_globals.copy(), locals=locs)

    @property
    @cache
    def walk(self):
        if self.file.exists:
            self.source = self.file.path.read_text_tokenize
            get = attrgetter('name')
            for node in walk(parse(self.source, filename=self.file.path.text)):
                if self.file.include \
                        and ((isinstance(node, Import)
                              and any([self.__package == cast(str, n).split('.')[0] for n in map(get, node.names)]))
                             or (isinstance(node, ImportFrom) and node.level == 0
                                 and node.module.split('.')[0] == self.__package)):
                    self._external = True
                elif isinstance(node, (FunctionDef, AsyncFunctionDef, ClassDef)):
                    start, end = self.interval(node)
                    self._functions[start:end] = node
                elif hasattr(node, 'lineno'):
                    if node.lineno not in self.lines:
                        self.lines |= {node.lineno: set()}
                    self.lines[node.lineno].add(node)
                self.lines = dict_sort(self.lines)
            return self.source


class MatchError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__('No match', *[f'{key}: {value}' for key, value in kwargs.items()], *args)


Real = NamedTuple('Real', caller=Frame, real=Optional[Frame])


class Stack(tuple[Frame]):
    real: Optional[Frame] = None

    def __new__(cls, init=False):
        fs = list()
        frame = FRAME_SYS_INIT if init else sys._getframe(1)
        while frame:
            fs.append(Frame(frame))
            frame = frame.f_back
        return tuple.__new__(Stack, fs)

    def __call__(self, index=FRAME_INDEX, real=False):
        caller = self[index]
        rv = None
        if real:
            with suppress(IndexError):
                rv = self[index + (caller.info.real or bool(None))]
        return Real(caller=caller, real=rv) if real else caller

    def __repr__(self):
        msg = f',{NEWLINE}'.join([f'[{self.index(item)}]: {self[self.index(item)]}' for item in self])
        return f'{self.__class__.__name__}({NEWLINE}{msg}{NEWLINE})'

    @classmethod
    @cache
    def init(cls):
        # noinspection PyArgumentList
        return cls(init=True)

    @classmethod
    @cache
    def main(cls):
        return cls.init()[0]

    @classmethod
    @cache
    def package(cls):
        return first_true(cls.init(), attrgetter('external'))
