# -*- coding: utf-8 -*-
"""Exceptions Module."""
__all__ = (
    'RcError',
    'AioCmdError',
    'CmdError',
)

from subprocess import CompletedProcess


class RcError(Exception):
    """Base class for all package exceptions."""


class CmdError(RcError):
    """Thrown if execution of cmd command fails with non-zero status code."""

    def __init__(self, rv: CompletedProcess):
        command = rv.args
        rc = rv.returncode
        stderr = rv.stderr
        stdout = rv.stdout
        super(CmdError, self).__init__(f'{command=}', f'{rc=}', f'{stderr=}', f'{stdout=}')


class AioCmdError(CmdError):
    """Thrown if execution of aiocmd command fails with non-zero status code."""

    def __init__(self, rv: CompletedProcess):
        super(AioCmdError, self).__init__(rv)
