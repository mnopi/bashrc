#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The setup script."""
import pathlib

from setuptools import setup, find_packages

# noinspection PyUnresolvedReferences
from bashrc import GITHUB_EMAIL, REALNAME, GITHUB_URL

project = pathlib.Path(__file__).parent
scripts = pathlib.Path() / f'{project.name}/scripts'
readme = project / 'README.md'
description = project.name
if readme.is_file():
    try:
        description = str(readme).splitlines()[0].split('#')[1]
    except IndexError:
        pass

setup(
    author=REALNAME,
    author_email=GITHUB_EMAIL,
    description=description,
    include_package_data=True,
    install_requires=[
        'bump2version',
        'pip',
        'setuptools',
        'twine',
        'wheel',
    ],
    name=project.name,
    package_data={
        project.name: [f'{project.name}/scripts/*'],
    },
    packages=find_packages(include=[project.name]),
    scripts=[item for item in scripts.iterdir()],
    setup_requires=[],
    tests_require=[],
    url=f'{GITHUB_URL}/',
    version='0.1.2',
    zip_safe=False,
)
