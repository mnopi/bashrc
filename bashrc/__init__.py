# -*- coding: utf-8 -*-
"""Bashrc Package."""
import os
import pathlib

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

__all__ = ['package', 'project', 'scripts', 'scripts_relative', 'readme', 'description']

for global_var, global_value in os.environ.items():
    # noinspection PyStatementEffect
    globals()[global_var] = global_value
    __all__.append(global_var)
