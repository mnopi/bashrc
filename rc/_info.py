# -*- coding: utf-8 -*-
"""Info Module."""
from __future__ import annotations

from dataclasses import Field
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields

from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Union


from ._path import *
from .enums import *
from .utils import *
from .vars import *

__all__ = (
    'SYS_PATHS',

    'Frame',

    'info',
    # 'caller',
    # 'package',
)


SYS_PATHS = Box({key: Path(value) for key, value in get_paths().items()})


Frame = NamedTuple('Frame', file=Path, globs=dict, init=Path, locs=dict, module=str, name=str, package=str, path=Path,
                   relative=Path, root=Path, spec=Optional[ModuleSpec])


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

    context: Union[int, Field] = field(default=1, init=False)
    filtered: Union[bool, Field] = field(default=False, init=False)
    index: Union[int, Field] = field(default=2, init=False)
    lower: Union[bool, Field] = field(default=True, init=False)
    # main: Frame = Path.package()
    prop: Union[Optional[bool], Field] = field(default=None, init=False)
    stack: Union[Optional[List[FrameInfo]], Field] = field(default=None, init=False)

    def __call__(self, index: int = index.default, context: int = context.default, depth: Optional[int] = depth,
                 filtered: bool = filtered.default, ignore: bool = ignore, lower: bool = lower.default,
                 stack: Optional[List[FrameInfo]] = stack.default, swith: str = swith) -> info:
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
        ic(index)
        self.index = index
        self.context = context
        self.depth = depth
        self.filtered = filtered
        self.ignore = ignore
        self.lower = lower
        self.swith = swith

        self.stack = stack or inspect.stack(self.context)

        if context := self.locs.get('stack_context'):
            self.context = context
            self.stack = inspect.stack(self.context)
        return self

    def __call__1(self, index: int = index.default, context: int = context.default, depth: Optional[int] = depth,
                 filtered: bool = filtered.default, ignore: bool = ignore, lower: bool = lower.default,
                 stack: Optional[List[FrameInfo]] = stack.default, swith: str = swith) -> info:
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
        ic(index)
        self.index = index
        self.context = context
        self.depth = depth
        self.filtered = filtered
        self.ignore = ignore
        self.lower = lower
        self.swith = swith

        self.stack = stack or inspect.stack(self.context)

        if context := self.locs.get('stack_context'):
            self.context = context
            self.stack = inspect.stack(self.context)
        return self

    def __hash__(self):
        return hash((self.index, self.context, self.depth, self.filtered, self.ignore, self.lower, self.swith, ))

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
            return self.frame().code_context
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
        return del_key(self.data, key)


    @cache
    def frame(self) -> Frame:
        if self.stack:
            try:
                return self.stack[self.index]
            except IndexError:
                self.index -= 1
                return self.stack[self.index]

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
            return self.frame().function
        except AttributeError:
            pass

    @property
    def globs(self) -> dict:
        try:
            return self.frame().frame.f_globals.copy()
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
        return file.git if file else GitTop()

    @classmethod
    def init(cls):
        index = 0
        count = len(STACK)
        ic(STACK[0].filename)
        found = False
        if count > 1:
            index = 1
            ic(index, len(STACK[index:]))
            for frame in STACK[index:]:
                index += 1
                file_path = Path(frame.filename)
                spec = frame.frame.f_globals.get('__spec__')
                ic(file_path, index)
                if all([getattr(spec, 'has_location', None),
                        frame.index == 0,
                        not file_path.is_relative_to(_main.path),
                        not _main.path.is_relative_to(file_path.parent),  # setup.py
                        frame.function == FUNCTION_MODULE,
                        'PyCharm' not in file_path,
                        not file_path.installedbin,
                        file_path.suffix,
                        not file_path.is_relative_to(SYS_PATHS.stdlib),
                        not file_path.is_relative_to(SYS_PATHS.purelib),
                        not file_path.is_relative_to(SYS_PATHS.include),
                        not file_path.is_relative_to(SYS_PATHS.platinclude),
                        not file_path.is_relative_to(SYS_PATHS.scripts),
                        ]):
                    found = True
                    ic(file_path, found)
                    break
        ic(STACK[0 if count == 1 or not found else index], 0 if count == 1 or not found else index)
        return cls()(index=0 if count == 1 or not found else index, stack=STACK)

    @classmethod
    def _init(cls, init: bool = True):
        index = 0
        count = len(STACK)
        ic(STACK[0].filename)
        found = False
        if count > 1:
            index = 1
            ic(index, len(STACK[index:]))
            for frame in STACK[index:]:
                index += 1
                file_path = Path(frame.filename)
                spec = frame.frame.f_globals.get('__spec__')
                ic(file_path, index)
                if all([all([getattr(spec, 'has_location', None),
                        frame.index == 0,
                        not file_path.is_relative_to(_main.path),
                        not _main.path.is_relative_to(file_path.parent),  # setup.py
                        frame.function == FUNCTION_MODULE,
                        'PyCharm' not in file_path]) if init else True,
                        not file_path.installedbin,
                        file_path.suffix,
                        not file_path.is_relative_to(SYS_PATHS.stdlib),
                        not file_path.is_relative_to(SYS_PATHS.purelib),
                        not file_path.is_relative_to(SYS_PATHS.include),
                        not file_path.is_relative_to(SYS_PATHS.platinclude),
                        not file_path.is_relative_to(SYS_PATHS.scripts),
                        ]):
                    found = True
                    ic(file_path, found)
                    break
        ic(STACK[0 if count == 1 or not found else index], 0 if count == 1 or not found else index)
        return cls()(index=0 if count == 1 or not found else index, stack=STACK)

    @property
    def installed(self) -> Optional[Path]:
        file = self.file
        # noinspection PyArgumentList
        return file.installed if file else None

    def instance(self, cls: Union[tuple, Any]) -> bool:
        return isinstance(self.data, cls)

    @property
    def instance_name(self,) -> str:
        return varname()

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
            return self.frame().lineno
        except AttributeError:
            return None

    @property
    def list(self) -> bool:
        return self.instance(list)

    def new(self, data: Any = None, /, **kwargs) -> info:
        return info(**{f.name: getattr(self, f.name) for f in fields(self)} | dict(
            data=self.data if data is None else data) | kwargs)

    @property
    def qual(self) -> Optional[str]:
        return getattr(self.func, f'__qualname__', None)

    @property
    def repo(self) -> Optional[str]:
        return self.gittop.name or self.repo_var

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
# TODO: ver que hago con el nombre del paquete.
# TODO: la variable de REPO_VAR y el usuario y el repo url meterlo en el _info.

# caller = info()
# _main = info.main()
# package = info.init()
