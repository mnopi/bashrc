from ast import AST
from ast import NodeVisitor
from enum import Enum
from functools import cache
from types import FrameType
from typing import Any
from typing import Iterable
from typing import NamedTuple
from typing import Optional
from typing import TypeVar
from typing import Union

from intervaltree import Interval
from intervaltree import IntervalTree

from .path import Path
from .path import PathLikeStr
from .utils import Base

_I = TypeVar('_I', bound='FrameId')

__all__: tuple
FRAME_INDEX: int
FRAME_SYS_INIT: FrameType
FUNCTION_MODULE : str
MODULE_MAIN: str
PATHS_SYS: dict[str, Path]
PATHS_SYS_EXCL: list[Path]
PATHS_EXCL: list[Path]

FId = NamedTuple('_FrameID', code=str, decorator=Union[list, str], function=str, parts=Union[list, str],
                      real=Optional[int], ASYNC=bool)
File = NamedTuple('File', include=bool, exists=int, path=Path)
Function = NamedTuple('Function', ASYNC=bool, cls=str, decorators=list, module=bool, name=str, qual=str)
GlobalsModule = NamedTuple('GlobalsModule', all=str, annotations=str, builtins=str, cached=str, doc=str, file=str,
                           loader=str, name=str, package=str, repo=str, pypi=str, spec=str)
GLOBALS_MODULE: GlobalsModule
Info = NamedTuple('Info', ASYNC=bool, external=bool, internal=bool, module=bool, real=Optional[int])
IntervalBase = NamedTuple('IntervalBase', begin=int, end=int, data=AST)
IntervalType: Interval[IntervalBase]
Line = NamedTuple('Line', ASYNC=bool, code=list[str], lineno=int)
Package = NamedTuple('Package', init=Optional[Path], module=str, name=str, package=str, path=Optional[Path], prefix=str,
                     relative=Optional[Path], root=Optional[Path], var=Optional[GlobalsModule])
Var = NamedTuple('Var', args=dict, globals=dict, locals=dict)
class FrameId(Enum):
    ASYNCCONTEXTMANAGER: _I
    IMPORTLIB: _I
    RUN: _I
    TO_THREAD: _I
class Frame(Base, NodeVisitor):
    __file: Path = ...
    __path: Path = ...
    __package: str = ...
    __slots__ = tuple[str]
    __hash_exclude__: tuple[str]
    exists: bool
    _external: bool
    file: Path
    frame: FrameType
    include: bool
    internal: bool
    _function: str
    functions: IntervalTree
    lineno: int
    lines: dict[int, set[AST, ...]]
    module: bool
    obj: Any
    source: str
    stack: Stack
    def __init__(self, file: Optional[Path], frame: Optional[FrameType] = ..., obj: Any = ...,
                 source: PathLikeStr = ...): ...
    def __repr__(self): ...
    @classmethod
    def distance(cls, lineno: int, value: Iterable[IntervalType, ...]) -> Interval: ...
    @property
    @cache
    def external(self) -> Frame: ...
    @property
    @cache
    def function(self) -> Function: ...
    @property
    @cache
    def info(self) -> Info: ...
    @classmethod
    def interval(cls, node: AST) -> tuple[int, int]: ...
    @property
    @cache
    def line(self) -> Line: ...
    @property
    @cache
    def package(self) -> Package: ...
    @property
    @cache
    def var(self) -> Var: ...
    @property
    @cache
    def walk(self) -> Optional[str]: ...
class MatchError(Exception):
    def __init__(self, *args, **kwargs) -> None: ...
Real = NamedTuple('Real', caller=Frame, real=Optional[Frame])
class Stack(tuple[Frame]):
    real: Optional[Frame]
    def __new__(cls, init: bool = ...) -> None: ...
    def __call__(self, index: int = ..., real: bool = ...) -> Union[Frame, Real]: ...
    def __repr__(self) -> str: ...
    @classmethod
    @cache
    def init(cls) -> Stack: ...
    @classmethod
    @cache
    def main(cls) -> Frame: ...
    @classmethod
    @cache
    def package(cls) -> Frame: ...
