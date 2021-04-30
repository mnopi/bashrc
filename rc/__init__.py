# -*- coding: utf-8 -*-
"""Bashrc Frame."""
import logging
import os

import colorama
import urllib3

from .utils import *

colorama.init()
logging.getLogger('paramiko').setLevel(logging.NOTSET)
urllib3.disable_warnings()
os.environ['PYTHONWARNINGS'] = 'ignore'

# __all__ = \
#     frame.__all__ + \
#     path.__all__ + \
#     utils.__all__

__all__ = utils.__all__
