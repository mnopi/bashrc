# -*- coding: utf-8 -*-
"""Bashrc Package."""
import os
import pathlib
import shutil
from importlib.metadata import version as __version__
from typing import Text, Literal

import distro
import typer

command = 'rc'
Int = bool

app = typer.Typer()

package = pathlib.Path(__file__).parent.resolve()
project = package.parent

scripts = package / 'scripts'
scripts_relative = [str(item.relative_to(project)) for item in scripts.iterdir()]

readme = project / 'README.md'
description = package.name
if readme.is_file():
    try:
        description = str(readme).splitlines()[0].split('#')[1]
    except IndexError:
        pass

requirements = project / 'requirements.txt'
if requirements.is_file():
    install_requires = requirements.read_text().splitlines()
else:
    install_requires = list()

atlas_url = f'mongodb+srv://{os.environ["USER"]}:' \
      f'{os.environ["NFERX_ATLAS_PASSWORD"]}@pen.ydo6l.mongodb.net/pen?retryWrites=true&w=majority'

__all__ = ['package', 'project', 'scripts', 'scripts_relative', 'readme', 'description']

for global_var, global_value in os.environ.items():
    # noinspection PyStatementEffect
    globals()[global_var] = global_value
    __all__.append(global_var)


class Option:
    """APP/CLI Option."""
    Function = Literal['version', ]  # type: ignore

    @staticmethod
    def version():
        return ['patch', 'minor', 'major', ]

    @staticmethod
    def option(function: Function = Function.__args__[0], msg: Text = None, default: Int = 0) -> typer.Option:
        """
        APP/CLI Option.

        Examples:
            >>> assert '<typer.models.OptionInfo' in str(Option.option())
            >>> Option.option(Option.Function.__args__[0], 'Part of version to increase') #doctest: +ELLIPSIS
            <...

        Args:
            function: completion function name.
            msg: cli help message for option.
            default: index for default choice.

        Returns:
            typer.Option:
        """
        # noinspection PyCallByClass
        attribute = getattr(Option, function)
        return typer.Option(attribute()[default], help=msg if msg else function.capitalize(), autocompletion=attribute)


dist = distro.LinuxDistribution().info()['id']


@app.command()
def secrets():
    """Secrets Update."""
    global dist
    if dist == 'darwin':
        os.system(f'secrets-push.sh')
    elif dist == 'Kali':
        os.system(f'secrets-pull.sh')


@app.command()
def up(bump: Text = Option.option(Option.Function.__args__[0], 'Part of version to increase')):
    """
    Project Upgrade.

    Args:
        bump: Part of version to increase.
    """
    global dist
    if dist == 'darwin':
        os.system(f'{command} --version && bashrc-upload.sh {bump} && {command} --version  && source ~/.bashrc')
    elif dist == 'Kali':
        os.system(f'{command} --version && bashrc-upgrade.sh && {command} --version && source ~/.bashrc')


def version_callback(value: bool):
    if value:
        typer.echo(f"{__version__(package.name)}")
        raise typer.Exit()


# noinspection PyUnusedLocal
@app.callback()
def main(version: bool = typer.Option(None, "--version", callback=version_callback, is_eager=True)):
    # Do other global stuff, handle other global options here
    return
