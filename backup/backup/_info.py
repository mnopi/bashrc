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


    def new(self, data: Any = None, /, **kwargs) -> info:
        return info(**{f.name: getattr(self, f.name) for f in fields(self)} | dict(
            data=self.data if data is None else data) | kwargs)


# TODO: falta el bapy con el file, con len = 1
# TODO: meter los asdict y las pydantic
# TODO: cambiar las clases que hereden de base y poner el asdict en base.
# TODO: lo de los paquetes de module_from_obj, import etc.
# TODO: ver que hago con el nombre del paquete.
# TODO: la variable de REPO_VAR y el usuario y el repo url meterlo en el _info.

# caller = info()
# _main = info.main()
# package = info.init()
