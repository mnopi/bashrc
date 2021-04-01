__all__ = (
    'APPCONTEXT',
    'Project',
)

from dataclasses import dataclass

from typer import Typer

from ._info import info


APPCONTEXT = dict(help_option_names=['-h', '--help'], color=True)


@dataclass
class Project(info):

    def __post_init__(self, init: bool):
        super().__post_init__(init)
        self.cli = Typer(name=self.package, context_settings=APPCONTEXT)


project = Project(init=True)
