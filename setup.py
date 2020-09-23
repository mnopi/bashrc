#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The setup script."""
from setuptools import setup, find_packages

# noinspection PyUnresolvedReferences
from bashrc import package, project, scripts_relative, readme, description, install_requires, GITHUB_EMAIL, \
    REALNAME, GITHUB_URL, command

setup(
    author=REALNAME,
    author_email=GITHUB_EMAIL,
    description=description,
    entry_points={
        'console_scripts': [
            f'{command} = {package.name}:app',
        ],
    },
    include_package_data=True,
    install_requires=install_requires,
    name=package.name,
    package_data={
        project.name: [f'{project.name}/scripts/*'],
    },
    packages=find_packages(include=[package.name]),
    scripts=scripts_relative,
    setup_requires=[],
    tests_require=[],
    url=f'{GITHUB_URL}/',
    version='0.31.38',
    zip_safe=False,
)
