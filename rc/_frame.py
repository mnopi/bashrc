from __future__ import annotations

import ast
import inspect
import sys
import sysconfig
from functools import cache
from typing import Any
from typing import Iterable
from typing import NamedTuple
from typing import Optional

from intervaltree import Interval
from intervaltree import IntervalTree

from ._path import *
from .enums import *
from .exceptions import *
from .utils import *

__all__ = (
    'CALL_INDEX',
    'FRAME_SYS_INIT',
    'FrameSysType',
    'SYS_PATHS',
    'SYS_PATHS_EXCLUDE',

    'Frame',
    'Stack',
)

CALL_INDEX = 1
FRAME_SYS_INIT = sys._getframe(0)
FrameSysType = type(FRAME_SYS_INIT)
SCRATCHES_INCLUDE_DEFAULT = False
SYS_PATHS = {key: Path(value) for key, value in sysconfig.get_paths().items()}
SYS_PATHS_EXCLUDE = [SYS_PATHS[item] for item in ['stdlib', 'purelib', 'include', 'platinclude', 'scripts']]

# TODO: Probar lo de func_code que esta en scratch_4.py
#   añadir los casos de llamada de async que había mirado
#   include file en el frame acabarlo.
#  ver que coño se hace ahora con si encuentro el caller, y cual es el real y en
#   func_code tengo que mirar el modulo!!!! para ver si es sync o async no como lo tengo hecho.
#   que hago con asyncio.run en modulo o con las funciones si son de main!!!


File = NamedTuple('File', INCLUDE=bool, exists=int, path=Path)
Function = NamedTuple('Function', ASYNC=bool, cls=str, decorators=list, module=bool, name=str, qual=str)
Info = NamedTuple('Info', ASYNC=bool, module=bool, real=Optional[int])
Line = NamedTuple('Line', ASYNC=bool, code=list[str], lineno=int)
Var = NamedTuple('Var', args=dict, globals=dict, locals=dict)

IntervalBase = NamedTuple('IntervalBase', begin=int, end=int, data=ast.AST)
IntervalType = Interval[IntervalBase]


class Frame:
    __slots__ = ('frame', 'functions', 'lines', 'module')

    def __hash__(self):
        return hash((self.frame, self.module))

    def __init__(self, frame: FrameSysType):
        self.frame: FrameSysType = frame
        self.functions: IntervalTree[IntervalType, ...] = IntervalTree()
        self.lines: dict[int, set[ast.AST, ...]] = dict()
        self.module: bool = self.frame.f_code.co_name == FUNCTION_MODULE

    def __repr__(self):
        return repr_format(self, 'file function info line', newline=True)

    @property
    @cache
    def code(self) -> Optional[list[str]]:
        if self.line:
            return

    @classmethod
    def distance(cls, lineno: int, value: Iterable[IntervalType, ...]) -> Interval:
        distance = {lineno - item.begin: item.data for item in value}
        return Interval(min(distance.keys()), max(distance.keys()), distance)

    @property
    @cache
    def file(self) -> File:
        p = Path(inspect.getsourcefile(self.frame) or inspect.getfile(self.frame))
        return File(exists=p.resolved.exists(), include=file_include(p.text), path=p)

    @classmethod
    def file_include(cls, file: Path, scratches: bool = False) -> bool:
        if not file.resolved.exists():
            return False
        if scratches and 'scratches' in file:
            return True
        for f in SYS_PATHS_EXCLUDE:
            if file.is_relative_to(f):
                return False
        return True

    @property
    @cache
    def function(self) -> Function:
        lineno = self.frame.f_lineno
        function = self.frame.f_code.co_name
        if self.module:
            return Function(cls=str(), decorators=[], module=self.module, name=function,
                            qual=function, sync=True)
        if self.source:
            if not (routines := self.functions[lineno]):
                raise MatchError(file=self.file, lineno=lineno, function=function)
            distance = self.distance(lineno, routines)
            distance_min = distance.data[distance.begin]
            if distance_min.name != function:
                raise MatchError(file=self.file, lineno=lineno, name=distance_min.name,
                                 function=function)
            return Function(cls=distance.data[distance.end].name,
                            decorators=[item.id for item in distance_min.decorator_list],
                            module=self.module,
                            name=function,
                            qual='.'.join([distance.data[item].name for item in sorted(distance.data, reverse=True)]),
                            sync=not isinstance(distance_min, ast.AsyncFunctionDef))
        return Function(str(), [], module=self.module, name=str(), qual=str(), sync=True)

    @property
    @cache
    def id(self) -> Optional[FrameID]:
        for i in FrameID:
            if i.value.function == self.frame.f_code.co_name and i.value.parts in self.file.path:
                return i

    @classmethod
    def include_file(cls):
        pass

    @classmethod
    def interval(cls, node: ast.AST) -> tuple[int, int]:
        if hasattr(node, 'lineno'):
            min_lineno = node.lineno
            max_lineno = node.lineno
            for node in ast.walk(node):
                if hasattr(node, 'lineno'):
                    min_lineno = min(min_lineno, node.lineno)
                    max_lineno = max(max_lineno, node.lineno)
            return min_lineno, max_lineno + 1

    @property
    @cache
    def line(self) -> Line:
        lineno = self.frame.f_lineno
        sync = True
        if self.source:
            if not (line := self.lines[lineno]):
                raise MatchError(file=self.file, lineno=lineno)
            for node in line:
                if isinstance(node, (ast.AsyncFor, ast.AsyncWith, ast.Await, )):
                    sync = False
                    break
            return Line(
                code=str(max({ast.get_source_segment(self.source, node) for node in line}, key=len)).split('\n'),
                lineno=lineno,
                sync=sync,
            )
        return Line(list(), lineno, sync)

    @property
    @cache
    def source(self) -> Optional[str]:
        if self.file.exists:
            source = self.file.path.read_text_tokenize
            for node in ast.walk(ast.parse(source, filename=self.file.path.text)):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start, end = self.interval(node)
                    self.functions[start:end] = node
                elif hasattr(node, 'lineno'):
                    if node.lineno not in self.lines:
                        self.lines |= {node.lineno: set()}
                    self.lines[node.lineno].add(node)
                self.lines = dict_sort(self.lines)
            return source

    @property
    @cache
    def info(self) -> Info:
        rv = None
        for i in FrameID:
            if i.value.function == self.frame.f_code.co_name and i.value.parts in self.file.path:
                rv = i
                break
        return Info(module=self.module,
                    real=rv.value.real if rv else int() if self.file.include else None,
                    sync=rv.value.ASYNC if rv else self.line.sync or self.function.sync)

    @property
    @cache
    def var(self) -> Var:
        locs = self.frame.f_locals.copy()
        arg, varargs, keywords = inspect.getargs(self.frame.f_code)
        return Var(args=del_key({name: locs[name] for name in arg} | ({varargs: val} if (
            val := locs.get(varargs)) else dict()) | (kw if (kw := locs.get(keywords)) else dict())),
                    globals=self.frame.f_globals.copy(), locals=locs)


class Stack(tuple[Frame]):

    def __new__(cls, init: bool = False):
        fs = list()
        frame = FRAME_SYS_INIT if init else sys._getframe(1)
        while frame:
            fs.append(Frame(frame))
            frame = frame.f_back
        return tuple.__new__(Stack, fs)

    def __call__(self, index: int = CALL_INDEX) -> Frame:
        return self[index]

    def __repr__(self):
        msg = f',{NEWLINE}'.join([f'[{self.index(item)}]: {self[self.index(item)]}' for item in self])
        return f'{self.__class__.__name__}({NEWLINE}{msg}{NEWLINE})'
