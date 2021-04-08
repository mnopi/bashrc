# -*- coding: utf-8 -*-
"""Utils Module."""
__all__ = (
    'Annotation',
    'aioloop',
    'annotations',
    '_base',
    'basemodel',
    'cmd',
    'cmdname',
    'current_task_name',
    'del_key',
    'dict_sort',
    'join_new',
    'namedinit',
    'prefixed',
    'repr_format',
    'slots',
    'sudo',
    'to_iter'
)

from asyncio.events import _RunningLoop
from collections import OrderedDict
from typing import Any
from typing import Callable
from typing import get_args
from typing import get_origin
from typing import get_type_hints
from typing import Iterable
from typing import Literal
from typing import NamedTuple
from typing import Optional
from typing import Type
from typing import Union

from .exceptions import *
from .vars import *

Annotation = NamedTuple('Annotation', args=list[Type, ...], cls=Type, hints=dict, key=str, origin=Type)


def aioloop() -> Optional[_RunningLoop]:
    try:
        return get_running_loop()
    except RuntimeError:
        return None


def annotations(obj: Any) -> Optional[dict[str, Annotation]]:
    frame = inspect.stack()[1].frame
    with suppress(TypeError):
        a = get_type_hints(obj, globalns=frame.f_globals, localns=frame.f_locals)
        return {name: Annotation(list(get_args(a[name])), cls, a, name, get_origin(a[name]), )
                for name, cls in a.items()}


class _base:
    @classmethod
    @cache
    def clsname(cls, lower: bool = False) -> str:
        return cls.__name__.lower() if lower else cls.__name__

    @classmethod
    @cache
    def clsqual(cls, lower: bool = False) -> str:
        return cls.__qualname__.lower() if lower else cls.__qualname__

    def debug(self):
        debug(self)

    def fmic(self):
        fmic(self)

    def fmicc(self):
        fmicc(self)

    def ic(self):
        ic(self)

    def icc(self):
        icc(self)


class basemodel(BaseModel, _base):
    pass


def cmd(command: Iterable, exc: bool = False, lines: bool = True, shell: bool = True,
        py: bool = False, pysite: bool = True) -> Union[CompletedProcess, int, list, str]:
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

    Raises:
        CmdError:

    Returns:
        Union[Cmd, int, list, str]:
    """
    if py:
        m = '-m'
        if isinstance(command, str) and command.startswith('/'):
            m = str()
        command = f'{PYTHON_SITE if pysite else PYTHON_SYS} {m} {command}'
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


def cmdname(func: Callable, sep: str = '_') -> str:
    """Command name."""
    return func.__name__.split(**split_sep(sep))[0]


def current_task_name() -> str:
    return current_task().get_name() if aioloop() else str()


def del_key(data: Union[dict, list], key: Iterable = ('self', 'cls',)) -> Union[dict, list]:
    rv = data
    key = to_iter(key)
    if isinstance(data, dict):
        rv = data.copy()
        for item in key:
            with suppress(KeyError):
                del rv[item]
    elif isinstance(data, list):
        for item in key:
            with suppress(ValueError):
                data.remove(item)
        rv = data
    return rv


def dict_sort(data: dict, ordered: bool = False, reverse: bool = False) -> Union[dict, OrderedDict]:
    """
    Order a dict based on keys.

    Args:
        data: dict to be ordered.
        ordered: OrderedDict.
        reverse: reverse.

    Returns:
        Union[dict, collections.OrderedDict]:
    """
    rv = {key: data[key] for key in sorted(data.keys(), reverse=reverse)}
    if ordered:
        return OrderedDict(rv)
    return rv.copy()


class fromkeys(Box):
    def __init__(self, keys: Iterable, lower: bool = False):
        super().__init__({item: item.lower() if lower else item for item in to_iter(keys)})


def join_new(data: list) -> str:
    return NEWLINE.join(data)


def namedinit(cls: Type[NamedTuple], optional: bool = True, **kwargs):
    """
    Init with defaults a NamedTuple.

    Args:
        cls: cls.
        optional: True to use args[0] instead of None as default for Optional fallback to None if exception.
        **kwargs:

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

    Returns:
        cls:
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


def prefixed(name: str) -> str:
    try:
        return f'{name.upper()}_'
    except AttributeError:
        pass


def repr_format(obj: Any, attrs: Iterable, clear: bool = True, newline: bool = False):
    cls = obj.__class__
    if clear:
        for item in dir(cls):
            if (attr := getattr(cls, item, None)) and (c := getattr(attr, 'cache_clear', None)):
                # noinspection PyUnboundLocalVariable
                c()
    new = NEWLINE if newline else str()
    msg = f',{new if newline else " "}'.join([f"{arg}: {repr(getattr(obj, arg))}" for arg in to_iter(attrs)])
    return f'{cls.__name__}({new}{msg}{new})'


class slots:
    """
    Slots Repr Helper Class
    """
    __slots__ = ('_slots',)

    def __init__(self):
        self._slots = sorted({attr for item in inspect.getmro(self.__class__)
                              for attr in getattr(item, "__slots__", list())
                              if attr != slots.__slots__[0]})
        for attr in self._slots:
            self.__setattr__(attr, None)

    @recursive_repr()
    def __repr__(self):
        values = {name: getattr(self, name) for name in self._slots}
        return f'{self.__class__.__name__}({", ".join([slot + ":" + repr(value) for slot, value in values.items()])})'


def sudo(command: str, su: bool = False) -> str:
    return command if SUDO or not su else f'sudo {command}'


def to_iter(data: Any) -> Iterable:
    if isinstance(data, str):
        return data.split(' ')
    elif isinstance(data, Iterable):
        return data
    else:
        return [data]
