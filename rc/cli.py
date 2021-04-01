# -*- coding: utf-8 -*-
"""CLI Module."""
__all__ = (
    'project',
    'project_cmd',
    'version_cmd',
)
import typer

from ._project import project


@project.cli.command(name=project.clsname(True))
def project_cmd(ctx: typer.Context):
    """Project info."""
    print(ctx.command.name)
    # ic(dataasdict(project))


@project.cli.command(name='version')
def version_cmd(ctx: typer.Context):
    """Version info."""
    print(ctx.command.name)
