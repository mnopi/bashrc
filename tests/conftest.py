# -*- coding: utf-8 -*-
""" Module."""
import inspect

import pytest
from rc import *

p = Path(__file__)
_stack = inspect.stack()
ic.enabled = all([len(_stack) >= 7, p.parent.text in _stack[6].filename,
                  f'from {inspect.getmodulename(__file__)}' in _stack[6].code_context[0]])


@pytest.fixture
def cwd():
    return p.cwd()
