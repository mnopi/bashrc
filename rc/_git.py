from __future__ import annotations

__all__ = (
    'Git',
)

from dataclasses import dataclass

from git import Repo


@dataclass
class Git(Repo):
    pass
