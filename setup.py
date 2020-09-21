#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The setup script."""
from setuptools import setup, find_packages

# noinspection PyUnresolvedReferences
from bashrc import package, project, scripts_relative, readme, description, install_requires, GITHUB_EMAIL, \
    REALNAME, GITHUB_URL

setup(
    author=REALNAME,
    author_email=GITHUB_EMAIL,
    description=description,
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
    version='0.1.150',
    zip_safe=False,
)
