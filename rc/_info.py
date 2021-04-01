# -*- coding: utf-8 -*-
"""Info Module."""
from __future__ import annotations

__all__ = (
    'ModuleSpec',
    'STACK',
    'info',
    'caller',
)

import importlib
import inspect
from asyncio import iscoroutine
from asyncio import iscoroutinefunction
from contextlib import suppress
from dataclasses import InitVar
from inspect import getargvalues
from inspect import getmodulename
from inspect import isasyncgen
from inspect import isasyncgenfunction
from inspect import isawaitable

from dataclasses import fields
from typing import Callable

from dataclasses import field

from dataclasses import dataclass
from inspect import FrameInfo
from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union


from ._path import *
from .enums import *
from .utils import *


ModuleSpec = importlib._bootstrap.ModuleSpec
STACK = inspect.stack()


@dataclass
class info(_base):
    """
    Inspect and Caller Class.

    inspect (__post_init__) uses: self.data, self.depth and self.switch.
    caller (__call__) uses: self.context, self.stack,
    """
    data: Optional[Union[FrameType, Any]] = None
    depth: Optional[int] = None
    ignore: bool = False
    swith: str = '__'

    bapy: Any = field(default=1, init=False)
    context: int = field(default=1, init=False)
    filtered: bool = field(default=False, init=False)
    _frame: Optional[FrameType] = field(default=None, init=False)
    index: int = field(default=2, init=False)
    lower: bool = field(default=True, init=False)
    prop: Optional[bool] = field(default=None, init=False)
    stack: List[FrameInfo] = field(default=None, init=False)

    init: InitVar[bool] = False

    def __post_init__(self, init: bool):
        if init and self.stack is None:
            self.stack = STACK
            self.bapy = self(index=0)
            self()

    def __call__(self, index: int = index, context: int = context, depth: Optional[int] = depth,
                 filtered: bool = filtered, ignore: bool = ignore, lower: bool = lower,
                 stack: Optional[List[FrameInfo]] = None, swith: str = swith) -> info:
        """
        Caller var name.

        Examples:

            .. code-block:: python
                caller = info()

                class A:

                    def __init__(self):

                        self.instance = varname()

                a = A()

                var = caller(1, name=True)

        Args:
            index: index.
            context: stack context.
            depth: depth.
            filtered: filter globs and locs.
            ignore: ignore.
            lower: var name lower.
            stack: stack for real caller.
            swith: swith.

        Returns:
            Optional[str, info]:
        """
        self.index = index
        self.context = context
        self.depth = depth
        self.filtered = filtered
        self.ignore = ignore
        self.lower = lower
        self.swith = swith
        if not self.stack or stack:
            self.stack = stack or inspect.stack(self.context)
        if context := self.locs.get('stack_context'):
            self.context = context
            self.stack = inspect.stack(self.context)
        return self

    @property
    def args(self) -> Optional[dict]:
        if self.instance(FrameType):
            argvalues = getargvalues(self.data)
            args = {name: argvalues.locals[name] for name in argvalues.args} | (
                {argvalues.varargs: val} if (val := argvalues.locals.get(argvalues.varargs)) else dict()) | (
                       kw if (kw := argvalues.locals.get(argvalues.keywords)) else dict())
            return self.new(args).del_key()

    @property
    def asyncgen(self) -> bool:
        return isasyncgen(self.data)

    @property
    def asyncgenfunction(self) -> bool:
        return isasyncgenfunction(self.data)

    @property
    def awaitable(self) -> bool:
        return isawaitable(self.data)

    @property
    def code(self) -> list:
        try:
            return self.frame.code_context
        except AttributeError:
            return list()

    @property
    def coro(self) -> Optional[bool]:
        # noinspection PyArgumentList
        return type(self)(f).coro if (f := self.func) else None

    @property
    def coroutine(self) -> bool:
        return iscoroutine(self.data)

    @property
    def coroutinefunction(self) -> bool:
        return iscoroutinefunction(self.data)

    def del_key(self, key: Iterable = ('self', 'cls', )) -> Union[dict, list]:
        rv = self.data
        key = self.new(key).iter
        if self.dict:
            rv = self.data.copy()
            for item in key:
                with suppress(KeyError):
                    del rv[item]
        elif self.list:
            for item in key:
                with suppress(ValueError):
                    self.data.remove(item)
            rv = self.data
        return rv

    @property
    def file(self) -> Optional[Path]:
        try:
            return Path(self.frame.filename).resolved
        except AttributeError:
            pass

    @property
    def frame(self) -> Optional[FrameInfo]:
        if self.stack:
            try:
                self._frame = self.stack[self.index]
            except IndexError:
                self.index -= 1
                self._frame = self.stack[self.index]
            self.data = self._frame.frame
            return self._frame

    @property
    def func(self) -> Optional[Union[Callable, property]]:
        v = self.globs | self.locs
        for item in ['self', 'cls']:
            if (obj := v.get(item)) and self.function in dir(obj):
                if item == 'self':
                    cls = getattr(obj, '__class__')
                    if (func := getattr(cls, self.function)) and isinstance(func, property):
                        self.prop = True
                        return func
                return getattr(obj, self.function)
        return v.get(self.function)

    @property
    def function(self) -> Optional[str]:
        try:
            return self.frame.function
        except AttributeError:
            pass

    @property
    def globs(self) -> dict:
        try:
            return self.data.f_globals.copy()
        except AttributeError:
            return dict()

    def get(self, name: str, /, *args, default: Any = None) -> Any:
        for arg in map(type(self), args) or self.data:
            if frameinfo := arg.instance((FrameInfo, FrameType)):
                frame = arg.data.frame if frameinfo else arg.data
                for i in ['f_locals', 'f_globals']:
                    value = getattr(frame, i)
                    if name in value:
                        return value.get(name, default)
            elif arg.mutablemapping:
                if name in arg.data:
                    return arg.data.get(name, default)
            elif arg.hasattr(name):
                return getattr(arg.data, name, default)
        return default

    @property
    def git(self) -> Optional[GitTop]:
        pass

    @property
    def gittop(self) -> Optional[GitTop]:
        file = self.file
        # noinspection PyArgumentList
        return file.gittop if file else GitTop()

    @property
    def id(self) -> Optional[CallerID]:
        try:
            for i in CallerID:
                if all([i.value[0] in self.line, i.value[1] == self.function, i.value[2] in str(self.file),
                        i.value[3] in str(self.file)]):
                    return i
        except (IndexError, TypeError):
            pass

    @property
    def imported(self) -> bool:
        return self.index == 1

    @property
    def installed(self) -> Optional[Path]:
        file = self.file
        # noinspection PyArgumentList
        return file.installed if file else None

    def instance(self, cls: Union[tuple, Any]) -> bool:
        return isinstance(self.data, cls)

    @property
    def iscoro(self) -> bool:
        return any([self.asyncgen, self.asyncgenfunction, self.awaitable, self.coroutine, self.coroutinefunction])

    @property
    def iter(self) -> Iterable:
        return to_iter(self.data)

    @property
    def line(self) -> Optional[str]:
        try:
            return self.code[0] if self.context == 1 else ''.join(self.code)
        except (TypeError, IndexError):
            pass

    @property
    def lineno(self) -> Optional[int]:
        try:
            return self.frame.lineno
        except AttributeError:
            return None

    @property
    def list(self) -> bool:
        return self.instance(list)

    @property
    def locs(self) -> dict:
        try:
            return self.data.f_locals.copy()
        except AttributeError:
            return dict()

    @property
    def modname(self) -> Optional[GitTop]:
        name = self.name
        file = self.file
        return name.rpartition('.')[2] if name else getmodulename(file.text) if file else None
        pass

    @property
    def name(self) -> Optional[str]:
        name = self.globs.get('__name__')
        spec = self.spec
        file = self.file
        return spec.name if spec else name if name and name != MODULE_MAIN else \
            '.'.join([i.removesuffix(file.suffix) for i in file.relative_to(self.path.parent).parts])

    def new(self, data: Any = None, /, **kwargs) -> info:
        return info(**{f.name: getattr(self, f.name) for f in fields(self)} | dict(
            data=self.data if data is None else data) | kwargs)

    @property
    def package(self) -> Optional[str]:
        name = self.name
        path = self.path
        return self.globs.get('__package__') or (name.split('.')[0] if name else path.name if path else None)

    @property
    def path(self) -> Optional[Path]:
        return rv.parent if (rv := self.file.find_up(name='__init__.py').path) else self.file.parent

    @property
    def prefix(self) -> Optional[str]:
        return prefixed(self.package)

    @property
    def qual(self) -> Optional[str]:
        return getattr(self.func, f'__qualname__', None)

    @property
    def real(self) -> info:
        # noinspection PyArgumentList
        return self.__call__(index=self.index + i[4], s=self.stack) if (i := self.id) else None

    @property
    def spec(self) -> Optional[ModuleSpec]:
        return self.globs.get('__spec__')

    @property
    def spec_origin(self) -> Optional[Path]:
        spec = self.spec
        file = self.file
        return spec.origin if spec else file.parent if file else None

    @property
    def sync(self) -> Optional[bool]:
        sync = True
        if l := self.line:
            sync = not any([self.id in [CallerID.RUN],
                            'asyncio.run' in l, 'as_completed' in l, 'await' in l, 'ensure_future' in l, 'async' in l,
                            'gather' in l, 'create_task' in l])
        return sync

    @property
    def task(self) -> Optional[str]:
        return current_task_name()

    @property
    def dict(self) -> bool:
        return isinstance(self.data, dict)

# TODO: falta el bapy con el file, con len = 1
# TODO: meter los asdict y las pydantic
# TODO: cambiar las clases que hereden de base y poner el asdict en base.
# TODO: lo de los paquetes de module_from_obj, import etc.


caller = info()
