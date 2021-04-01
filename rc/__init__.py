# -*- coding: utf-8 -*-
"""Bashrc Package."""
import logging
import os
from logging import getLogger

import colorama
import urllib3

from ._git import *
from ._info import *
from ._path import *
from ._project import *
from ._user import *
from .cli import *
from .echo import *
from .enumdict import *
from .enums import *
from .exceptions import *
from .utils import *

colorama.init()
getLogger('paramiko').setLevel(logging.NOTSET)
urllib3.disable_warnings()
os.environ['PYTHONWARNINGS'] = 'ignore'

__all__ = \
    _git.__all__ + \
    _info.__all__ + \
    _path.__all__ + \
    _project.__all__ + \
    _user.__all__ + \
    cli.__all__ + \
    echo.__all__ + \
    enumdict.__all__ + \
    enums.__all__ + \
    exceptions.__all__ + \
    utils.__all__
