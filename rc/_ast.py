from __future__ import annotations

import ast
import tokenize
from operator import attrgetter
from typing import NamedTuple
from typing import Optional
from typing import Union

from intervaltree import Interval
from intervaltree import IntervalTree

from ._path import *
from .utils import *

__all__ = (
    'AST',
    'Expr',
    'NodeVisitor',

    'Ast'
)

Expr = ast.Expr
Import = ast.Import
ImportFrom = ast.ImportFrom
Module = ast.Module
Name = ast.Name
NodeVisitor = ast.NodeVisitor


class Ast(slots, NodeVisitor):
    __slots__ = (
        'clsname',
        'code',
        'decorators',
        'funcsync',
        'lines',
        'matchline',
        'matchroutine',
        'parsed',
        'qual',
        'source',
        'sync',
        'routines',
    )

    def __hash__(self):
        return hash((self.parsed, ))

    def __init__(self, file: Union[Path, str], function: str = None, lineno: int = None):
        super().__init__()
        if not file.exists():
            return
        self.source = file.read_text_tokenize
        self.parsed: AST = ast.parse(self.source, filename=file)
        self.lines: dict[int, set[AST]] = dict()
        self.treeother: IntervalTree[IntervalType, ...] = IntervalTree()
        self.routines: IntervalTree[IntervalType, ...] = IntervalTree()
        for node in ast.walk(self.parsed):
            if isinstance(node, (FunctionDef, AsyncFunctionDef, ClassDef)):
                start, end = self.interval(node)
                self.routines[start:end] = node
            if hasattr(node, 'lineno'):
                if node.lineno not in self.lines:
                    self.lines |= {node.lineno: set()}
                self.lines[node.lineno].add(node)
            self.lines = dict_sort(self.lines)
        self(function=function, lineno=lineno)

    def __call__(self, function: str = None, lineno: int = None) -> Ast:
        func = None
        if not lineno:
            return self
        for node in self.lines[lineno]:
            if isinstance(node, Call):
                ic(lineno, node.func, node)
                # TODO:
                #       Si la funcion es async entonces tiene que ir sync si es to_thread
                #           (o el modulo es threading, )
                #       Si la funcion es sync tiene que ir al sync de no ser que tenga asyncio.run
                #       Si la funcion es async y no es to_thread (o sea, que viniera de events, o del mismo,
                #               entonces async) - ver executor!
                # TODO: TENGO que meter la LINEA en el FRAME.
                # TODO: tengo que marcar si esta dentro de funcion y la funcion es sync o async.
                # TODO: Acabar con el qualname de clase y de la funcion.
                # TODO: as_completed.
                # TODO: lista de decoradores.
            elif isinstance(node, Await):
                ic(lineno, node.value, node, )
            elif isinstance(node, AsyncFor):
                ic(lineno, node.target, node.body, node, )
            elif isinstance(node, AsyncWith):
                ic(lineno, node.items, node.body, node, )
        if match := (self.routines[lineno] or self.lines.get(lineno)):
            self.matchroutine: Optional[Union[tuple[IntervalType, ...], tuple[AST, ...]]] = match
            line = isinstance(self.matchroutine[0], AST)
            if not line:
                distance = self.distance(lineno, match)
                distance_min = distance.data[distance.begin]
                self.clsname: Optional[str] = str() if line else distance.data[distance.end].name
                func: Optional[str] = FUNCTION_MODULE if line else distance_min.name
                self.qual: Optional[str] = '.'.join([value.name for value in distance.data.values()])
                self.funcsync: Optional[bool] = not isinstance(distance_min, AsyncFunctionDef)
                self.decorators = list() if line else [item.id for item in distance_min.decorator_list]
            self.matchline = match if line else tuple(self.lines.get(lineno))
            if not self.matchline:
                raise RuntimeError(f'Did not find a match for function [{self.qual} | {lineno}]: {repr(self)}')
            # TODO: Aqui hacer lo de si la llamada es sync o async
            for node in self.matchline:
                aw = isinstance(node, Await)
                afor = isinstance(node, AsyncFor)
                awith = isinstance(node, AsyncWith)
                a = any([aw, afor, awith])
                thread = False
                coroutine = a and not any([isinstance(node, Call)])
                # gather y el run joder!!!.
                # Tengo que mirar el body para ver si parte la linea o no en el caso de lineas largas,
                # porque entonces tengo que subir la linea no solo para coroutines sino para todo !!!
                """
                names = ['as_completed', 'create_task', 'wrap_future']
                if coroutine and aw and self.function and self.function != FUNCTION_MODULE and not line:
                    top = self.matchroutine[-1].min
                    index = lineno -1
                    l = tuple(self.lines.get(lineno))
                    while index > top:

                        if u

                while
                thread = if aw
                completed =
                if 'to_thread' in
                self.sync = True or False
                """
                ic(self.lines[lineno])
                if isinstance(node, Call):
                    ic(lineno, node.func, node)
                    # TODO:
                    #       Si la funcion es async entonces tiene que ir sync si es to_thread
                    #           (o el modulo es threading, )
                    #       Si la funcion es sync tiene que ir al sync de no ser que tenga asyncio.run
                    #       Si la funcion es async y no es to_thread (o sea, que viniera de events, o del mismo,
                    #               entonces async) - ver executor!
                    # TODO: TENGO que meter la LINEA en el FRAME.
                    # TODO: tengo que marcar si esta dentro de funcion y la funcion es sync o async.
                    # TODO: Acabar con el qualname de clase y de la funcion.
                    # TODO: as_completed.
                    # TODO: lista de decoradores.
                elif isinstance(node, Await):
                    ic(lineno, node.value, node, )
                elif isinstance(node, AsyncFor):
                    ic(lineno, node.target, node.body, node, )
                elif isinstance(node, AsyncWith):
                    ic(lineno, node.items, node.body, node, )


    @classmethod
    def distance(cls, lineno: int, value: tuple[IntervalType, ...]) -> Interval:
        distance = {lineno - item.begin: item.data for item in value}
        distance = {key: distance[key] for key in reversed(distance.keys())}
        return Interval(min(distance.keys()), max(distance.keys()), distance)

    @classmethod
    def interval(cls, node) -> tuple[int, int]:
        if hasattr(node, 'lineno'):
            min_lineno = node.lineno
            max_lineno = node.lineno
            for node in ast.walk(node):
                if hasattr(node, 'lineno'):
                    min_lineno = min(min_lineno, node.lineno)
                    max_lineno = max(max_lineno, node.lineno)
            return min_lineno, max_lineno + 1
