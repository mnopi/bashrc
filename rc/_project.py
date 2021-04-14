# __all__ = (
#     'APPCONTEXT',
#     'CLI',
#     'TESTS',
#     'Project',
# )

from dataclasses import dataclass
from dataclasses import field
from dataclasses import Field
from enum import Enum
from typing import Union

from typer import Typer


class Bump(str, Enum):
    MAJOR = 'major'
    MINOR = 'minor'
    PATCH = 'patch'
    PRERELEASE = 'prerelease'
    BUILD = 'build'
# from ._info import package
# from ._info import _main
# from ._info import info
# from .utils import *
#
# APPCONTEXT = dict(help_option_names=['-h', '--help'], color=True)
# CLI = varname(1)
# TESTS = varname(1)
# ic(TESTS)
#
# @dataclass
# class Project(_base):
#     init: Union[Field, info] = field(default=package, init=False)
#     main: info = field(default=_main, init=False)
#     cli: Typer = Typer(name=init.default.package, context_settings=APPCONTEXT)
#
#
# project = Project()
