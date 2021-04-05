# from __future__ import annotations

# import ast
# import sys
# from functools import cached_property
# from inspect import findsource
# from inspect import FrameInfo
# from inspect import getargs
# from inspect import getfile
# from inspect import getsourcefile
# from inspect import isframe
# from inspect import istraceback
# from textwrap import dedent
# from typing import ClassVar
# from typing import Optional

from interval import Interval
from intervaltree import IntervalTree
from typing import NamedTuple

# from ._path import *
# from ._ast import *
from .enums import *
# from .utils import *

#
# class Frame1(slots):
#
#     __slots__ = (
#         '_frame',
#         'ast',
#                  'function', 'funcmodule', 'id', 'include_file', 'lineno',
#                  'real', 'source', )
#
#     def __hash__(self):
#         return hash((self.file, self.function, self.lineno, ))
#
#     def __init__(self, frame: SysFrameType):
#         super().__init__()
#         # self.context: int = context
#         self._frame = frame
#         # self.code: str = str()
#         # self.code_context: list = list()
#         # co_firstlineno is the first line of the function withouth decorators (start + decorators)
#         # self.firstlineno: int = self.frame.f_code.co_firstlineno
#         # self.index: Optional[int] = None
#         self.lineno: int = self.frame.f_lineno
#         # (source, start): start is start line number of function including decorators.
#         self.source: tuple[list, int] = findsource(frame)
#         self.ast: Ast = Ast(self.file, function=self.function, lineno=self.lineno)

#         # start = self.lineno - 1 - context // 2
#         # try:
#         #     lines, _ = findsource(self.frame)
#         # except OSError:
#         #     pass
#         # else:
#         #     start = max(0, min(start, len(lines) - context))
#         #     self.code_context: list[str] = [l.rstrip() for l in lines[start:start + context]]
#         #     self.index: int = self.lineno - 1 - start
#         #     self.code: str = '\n'.join((map(dedent, self.code_context)))
#
#         self.id: Optional[CallerID] = None
#         self.real: Optional[int] = int()
#         # for i in CallerID:
#         #     if all([i.value[0] in self.code_context, i.value[1] == self.function, i.value[2] in self.file]):
#         #         self.id = i
#         #         self.real = i.value[3]
#         #         break
#

