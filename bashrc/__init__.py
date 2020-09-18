# coding=utf-8
"""Bashrc Package."""
import os

__all__ = []
for global_var, global_value in os.environ.items():
    # noinspection PyStatementEffect
    globals()[global_var] = global_value
    __all__.append(global_var)

print(__all__)
