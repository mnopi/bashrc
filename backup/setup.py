#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The setup script."""
from setuptools import setup
from icecream import ic
# path = Path(__file__).parent
# sys.path.append(str(path))
from rc import Distro

distro = Distro()
ic(distro)
setup(**distro.setup())

if __name__ == '__main__':
    setup()
