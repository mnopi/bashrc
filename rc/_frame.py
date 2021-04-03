import sys
from inspect import findsource
from inspect import FrameInfo
from inspect import getargs
from inspect import getfile
from inspect import getsourcefile
from inspect import isframe
from inspect import istraceback
from textwrap import dedent
from typing import Optional

from ._path import *
from .utils import *

__all__ = (
    'SYS_FRAME_INIT',
    'SysFrameType',
)

SYS_FRAME_INIT = sys._getframe(1)

SysFrameType = type(SYS_FRAME_INIT)

# TODO: Probar lo de func_code que esta en scratch_4.py
#   añadir los casos de llamada de async que había mirado
#   include file en el frame acabarlo.
#  ver que coño se hace ahora con si encuentro el caller, y cual es el real y en
#   func_code tengo que mirar el modulo!!!! para ver si es sync o async no como lo tengo hecho.
#   que hago con asyncio.run en modulo o con las funciones si son de main!!!



class Frame(slots):

    __slots__ = ('args', 'argsc', 'code', 'code_context', 'context', 'file', 'frame', 'function', 'globs', 'include_file', 'index', 'lineno', 'locs', )

    def __init__(self, context: int, frame: SysFrameType):
        self.context: int = context
        self.frame: SysFrameType = frame
        self.file: Path = Path(getsourcefile(self.frame) or getfile(self.frame))
        self.function: str = self.frame.f_code.co_name
        self.globs: dict = self.frame.f_globals.copy()
        self.lineno: int = self.frame.f_lineno
        self.locs: dict = self.frame.f_locals.copy()
        self.include_file: bool = include_file(Path.text)

        start = self.lineno - 1 - context // 2
        try:
            lines, _ = findsource(self.frame)
        except OSError:
            self.code_context = self.index = None
        else:
            start = max(0, min(start, len(lines) - context))
            self.code_context: list[str] = lines[start:start + context]
            self.index: int = self.lineno - 1 - start
            self.code: str = '\n'.join((map(dedent, self.code_context)))

        args, varargs, varkw = getargs(frame.f_code)

        if self.instance(FrameType):
            # argvalues = getargvalues(self.data)
            self.args = {name: self.locs[name] for name in args} | (
                {args.varargs: val} if (val := self.locs.get(args.varargs)) else dict()) | (
                       kw if (kw := self.locs.get(args.keywords)) else dict())
            return self.new(args).del_key()

def frames(context: int = 1, init: bool = False) -> list[Frame]:
    fs = list()
    frame = SYS_FRAME_INIT if init else sys._getframe(1)
    while frame:
        fs.append(Frame(context, frame))
        frame = frame.f_back
    return fs
