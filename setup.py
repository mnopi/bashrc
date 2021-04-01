#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The setup script."""
from setuptools import setup, find_packages

# noinspection PyUnresolvedReferences
import os
import pathlib


project = pathlib.Path(__file__).parent.resolve()

scripts = project / 'scripts'
scripts_relative = [str(item.relative_to(project)) for item in scripts.iterdir()]
#
# readme = project / 'README.md'
# description = package.name
# if readme.is_file():
#     try:
#         description = str(readme).splitlines()[0].split('#')[1]
#     except IndexError:
#         pass

requirements = project / 'requirements.txt'
if requirements.is_file():
    install_requires = requirements.read_text().splitlines()
else:
    install_requires = list()

packages = find_packages()
packages.remove('tests')

setup(
    # author=REALNAME,
    # author_email=GITHUB_EMAIL,
    # description=description,
    # entry_points={
    #     'console_scripts': [
    #         f'{command} = {package.name}:app',
    #     ],
    # },
    # include_package_data=True,
    install_requires=install_requires,
    name=project.name,
    # package_data={
    #     project.name: [f'{project.name}/scripts/*'],
    # },
    packages=find_packages(),
    scripts=scripts_relative,
    # setup_requires=[],
    # tests_require=[],
    # url=f'{GITHUB_URL}/',
    version='0.31.63',
    zip_safe=False,
)
