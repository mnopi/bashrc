# -*- coding: utf-8 -*-
"""Echo Module."""
__all__ = (
    'black',
    'blue',
    'cyan',
    'green',
    'magenta',
    'red',
    'white',
    'yellow',
    'bblack',
    'bblue',
    'bcyan',
    'bgreen',
    'bmagenta',
    'bred',
    'bwhite',
    'byellow',
)

from typing import Any
from typing import Optional

from typer import Exit
from typer import secho


def black(msg: Any, bold: bool = False, nl: bool = True, underline: bool = False,
          blink: bool = False, err: bool = False, reset: bool = True, rc: Optional[int] = None) -> None:
    """
    Black.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='black', err=err, reset=reset)
    if rc is not None:
        Exit(rc)


def blue(msg: Any, bold: bool = False, nl: bool = True, underline: bool = False,
         blink: bool = False, err: bool = False, reset: bool = True, rc: Optional[int] = None) -> None:
    """
    Blue.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='blue', err=err, reset=reset)
    if rc is not None:
        Exit(rc)


def cyan(msg: Any, bold: bool = False, nl: bool = True, underline: bool = False,
         blink: bool = False, err: bool = False, reset: bool = True, rc: Optional[int] = None) -> None:
    """
    Cyan.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='cyan', err=err, reset=reset)
    if rc is not None:
        Exit(rc)


def green(msg: Any, bold: bool = False, nl: bool = True, underline: bool = False,
          blink: bool = False, err: bool = False, reset: bool = True, rc: Optional[int] = None) -> None:
    """
    Green.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='green', err=err, reset=reset)
    if rc is not None:
        Exit(rc)


def magenta(msg: Any, bold: bool = False, nl: bool = True, underline: bool = False,
            blink: bool = False, err: bool = False, reset: bool = True, rc: Optional[int] = None) -> None:
    """
    Magenta.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='magenta', err=err, reset=reset)
    if rc is not None:
        Exit(rc)


def red(msg: Any, bold: bool = False, nl: bool = True, underline: bool = False,
        blink: bool = False, err: bool = True, reset: bool = True, rc: Optional[int] = None) -> None:
    """
    Red.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='red', err=err, reset=reset)
    if rc is not None:
        Exit(rc)


def white(msg: Any, bold: bool = False, nl: bool = True, underline: bool = False,
          blink: bool = False, err: bool = False, reset: bool = True, rc: Optional[int] = None) -> None:
    """
    White.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='white', err=err, reset=reset)
    if rc is not None:
        Exit(rc)


def yellow(msg: Any, bold: bool = False, nl: bool = True, underline: bool = False,
           blink: bool = False, err: bool = True, reset: bool = True, rc: Optional[int] = None) -> None:
    """
    Yellow.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='yellow', err=err, reset=reset)
    if rc is not None:
        Exit(rc)


def bblack(msg: Any, bold: bool = False, nl: bool = True, underline: bool = False,
           blink: bool = False, err: bool = False, reset: bool = True, rc: Optional[int] = None) -> None:
    """
    Black.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='bright_black', err=err,
          reset=reset)
    if rc is not None:
        Exit(rc)


def bblue(msg: Any, bold: bool = False, nl: bool = True, underline: bool = False,
          blink: bool = False, err: bool = False, reset: bool = True, rc: Optional[int] = None) -> None:
    """
    Bblue.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='bright_blue', err=err,
          reset=reset)
    if rc is not None:
        Exit(rc)


def bcyan(msg: Any, bold: bool = False, nl: bool = True, underline: bool = False,
          blink: bool = False, err: bool = False, reset: bool = True, rc: Optional[int] = None) -> None:
    """
    Bcyan.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='bright_cyan', err=err,
          reset=reset)
    if rc is not None:
        Exit(rc)


def bgreen(msg: Any, bold: bool = False, nl: bool = True, underline: bool = False,
           blink: bool = False, err: bool = False, reset: bool = True, rc: Optional[int] = None) -> None:
    """
    Bgreen.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='bright_green', err=err,
          reset=reset)
    if rc is not None:
        Exit(rc)


def bmagenta(msg: Any, bold: bool = False, nl: bool = True, underline: bool = False,
             blink: bool = False, err: bool = False, reset: bool = True, rc: Optional[int] = None) -> None:
    """
    Bmagenta.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='bright_magenta', err=err,
          reset=reset)
    if rc is not None:
        Exit(rc)


def bred(msg: Any, bold: bool = False, nl: bool = True, underline: bool = False,
         blink: bool = False, err: bool = False, reset: bool = True, rc: Optional[int] = None) -> None:
    """
    Bred.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='bright_red', err=err,
          reset=reset)
    if rc is not None:
        Exit(rc)


def bwhite(msg: Any, bold: bool = False, nl: bool = True, underline: bool = False,
           blink: bool = False, err: bool = False, reset: bool = True, rc: Optional[int] = None) -> None:
    """
    Bwhite.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='bright_white', err=err,
          reset=reset)
    if rc is not None:
        Exit(rc)


def byellow(msg: Any, bold: bool = False, nl: bool = True, underline: bool = False,
            blink: bool = False, err: bool = False, reset: bool = True, rc: Optional[int] = None) -> None:
    """
    Byellow.

    Args:
        msg: msg
        bold: bold
        nl: nl
        underline: underline
        blink: blink
        err: err
        reset: reset
        rc: rc
    """
    secho(f'{msg}', bold=bold, nl=nl, underline=underline, blink=blink, color=True, fg='bright_yellow', err=err,
          reset=reset)
    if rc is not None:
        Exit(rc)
