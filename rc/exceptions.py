# -*- coding: utf-8 -*-
"""Exceptions Module."""
__all__ = (
    'Error',
    'AioCmdError',
    'CmdError',
    'MatchError',
)

from subprocess import CompletedProcess


class Error(Exception):
    """Base class for all package exceptions."""


class CmdError(Error):
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


class MatchError(Error):
    def __init__(self, *args, **kwargs):
        super(Error, self).__init__('No match', *[f'{key}: {value}' for key, value in kwargs.items()], *args)
