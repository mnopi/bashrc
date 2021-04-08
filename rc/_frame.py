from __future__ import annotations

from typing import cast
from typing import Iterable
from typing import NamedTuple
from typing import Optional
from typing import Union


from ._path import *
from .enums import *
from .exceptions import *
from .utils import *
from .vars import *

__all__ = (
    'File',
    'Function',
    'Info',
    'Line',
    'Package',
    'Var',
    'Frame',
    'Stack',
)


# TODO: Probar lo de func_code que esta en scratch_4.py
#   añadir los casos de llamada de async que había mirado
#   include file en el frame acabarlo.
#  ver que coño se hace ahora con si encuentro el caller, y cual es el real y en
#   func_code tengo que mirar el modulo!!!! para ver si es sync o async no como lo tengo hecho.
#   que hago con asyncio.run en modulo o con las funciones si son de main!!!

File = NamedTuple('File', include=bool, exists=int, path=Path)
Function = NamedTuple('Function', ASYNC=bool, cls=str, decorators=list, module=bool, name=str, qual=str)
Info = NamedTuple('Info', ASYNC=bool, external=bool, module=bool, real=Optional[int])
Line = NamedTuple('Line', ASYNC=bool, code=list[str], lineno=int)
Package = NamedTuple('Package', init=Optional[Path], module=str, name=str, package=str, path=Optional[Path], prefix=str,
                     relative=Optional[Path], root=Optional[Path], var=Optional[ModuleVars])
Var = NamedTuple('Var', args=dict, globals=dict, locals=dict)

IntervalBase = NamedTuple('IntervalBase', begin=int, end=int, data=ast.AST)
IntervalType = Interval[IntervalBase]


class Frame:
    __file: Path = Path(__file__)
    __path: Path = __file.path
    __package: str = __path.name

    __slots__ = ('_external', '_frame', '_function', '_functions', '_lineno', '_lines', '_module', '_source')

    def __hash__(self):
        return hash((self._external, self._frame, self._function, self._module, self._source))

    def __init__(self, frame: FrameSysType):
        self._external: bool = False
        self._frame: FrameSysType = frame
        self._function: str = self._frame.f_code.co_name
        self._functions: IntervalTree[IntervalType, ...] = IntervalTree()
        self._lineno: int = self._frame.f_lineno
        self._lines: dict[int, set[ast.AST, ...]] = dict()
        self._module: bool = self._function == FUNCTION_MODULE

    def __repr__(self):
        return repr_format(self, 'file function info line package', clear=False, newline=True)

    @classmethod
    def distance(cls, lineno: int, value: Iterable[IntervalType, ...]) -> Interval:
        distance = {lineno - item.begin: item.data for item in value}
        return Interval(min(distance.keys()), max(distance.keys()), distance)

    @property
    @cache
    def external(self) -> Frame:
        return self if self.info.external else None

    @property
    @cache
    def file(self) -> File:
        f = Path(inspect.getsourcefile(self._frame) or inspect.getfile(self._frame))
        incl = False
        if exists := self.__file.resolved.exists():
            incl = not (any(map(f.is_relative_to, PATHS_EXCLUDE)) or
                        any([not f.suffix, f.is_relative_to(self.__file.parent), f.parent == self.__file.parent]))
        return File(exists=exists, include=incl, path=f)

    @property
    @cache
    def function(self) -> Function:
        if self._module or not self.source:
            return Function(False, str(), list(), self._module, self._function, self._function)

        if not (routines := self._functions[self._lineno]):
            raise MatchError(file=self.file, lineno=self._lineno, function=self._function)
        distance = self.distance(self._lineno, routines)
        distance_min = distance.data[distance.begin]
        if distance_min.name != self._function:
            raise MatchError(file=self.file, lineno=self._lineno, name=distance_min.name, function=self._function)
        return Function(isinstance(distance_min, ast.AsyncFunctionDef), distance.data[distance.end].name,
                        [item.id for item in distance_min.decorator_list], self._module, self._function,
                        '.'.join([distance.data[item].name for item in sorted(distance.data, reverse=True)]))

    @property
    @cache
    def info(self) -> Info:
        rv = None
        for i in FrameID:
            if i is FrameID.IMPORTLIB and self._lineno == 228:
                icc(i.value.function, self._function, i.value.function == self._function,
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
                    module=self._module,
                    real=rv.value.real if rv else int() if self.file.include else None)

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
        asy = False
        if self.source:
            if not (line := self._lines[self._lineno]):
                raise MatchError(file=self.file, lineno=self._lineno)
            for node in line:
                code = ast.get_source_segment(self.source, node)
                names = [asyncio.as_completed, asyncio.create_task, asyncio.ensure_future, asyncio.gather, asyncio]
                name = attrgetter('__name__')
                if isinstance(node, (ast.AsyncFor, ast.AsyncWith, ast.Await,)) or \
                        (any([contains(code, name(item)) for item in names]) and name(asyncio.to_thread) not in code):
                    asy = True
                    break
            return Line(
                code=str(max({ast.get_source_segment(self.source, node) for node in line}, key=len)).split('\n'),
                lineno=self._lineno,
                ASYNC=asy,
            )
        return Line(asy, list(), self._lineno)

    @property
    @cache
    def package(self) -> Package:
        if (self.file.include and self.info.external) or self.file.path == self.__file:
            init = self.file.path.find_up(name='__init__.py').path
            path = init.parent if init else init.parent \
                if ((rv := self.file.path.find_up(PathIs.DIR, PathSuffix.GIT).path) and (rv / 'HEAD').exists()) \
                else self.file.path.parent

            root = path.parent
            relative = self.file.path.relative_to(root)
            var = ModuleVars(*map(self.var.globals.get, MODULE_VARS._asdict().values()))
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
    def source(self) -> Optional[str]:
        if self.file.exists:
            source = self.file.path.read_text_tokenize
            get = attrgetter('name')
            for node in ast.walk(ast.parse(source, filename=self.file.path.text)):
                if self.file.include \
                        and ((isinstance(node, ast.Import)
                              and any([self.__package == cast(str, n).split('.')[0] for n in map(get, node.names)]))
                             or (isinstance(node, ast.ImportFrom) and node.level == 0
                                 and node.module.split('.')[0] == self.__package)):
                    self._external = True
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start, end = self.interval(node)
                    self._functions[start:end] = node
                elif hasattr(node, 'lineno'):
                    if node.lineno not in self._lines:
                        self._lines |= {node.lineno: set()}
                    self._lines[node.lineno].add(node)
                self._lines = dict_sort(self._lines)
            return source

    @property
    @cache
    def var(self) -> Var:
        locs = self._frame.f_locals.copy()
        arg, varargs, keywords = inspect.getargs(self._frame.f_code)
        return Var(args=del_key({name: locs[name] for name in arg} | ({varargs: val} if (
            val := locs.get(varargs)) else dict()) | (kw if (kw := locs.get(keywords)) else dict())),
                   globals=self._frame.f_globals.copy(), locals=locs)


Real = NamedTuple('Real', caller=Frame, real=Optional[Frame])


class Stack(tuple[Frame]):
    real: Optional[Frame] = None

    def __new__(cls, init: bool = False):
        fs = list()
        frame = FRAME_SYS_INIT if init else sys._getframe(1)
        while frame:
            fs.append(Frame(frame))
            frame = frame.f_back
        return tuple.__new__(Stack, fs)

    def __call__(self, index: int = FRAME_INDEX, real: bool = False) -> Union[Frame, Real]:
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
    def init(cls) -> Stack:
        # noinspection PyArgumentList
        return cls(init=True)

    @classmethod
    @cache
    def main(cls) -> Frame:
        return cls.init()[0]

    @classmethod
    @cache
    def package(cls) -> Frame:
        return first_true(cls.init(), attrgetter('external'))
