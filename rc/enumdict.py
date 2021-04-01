"""Enumdict Module."""
__all__ = (
    'Alias',
    'Enum',
    'EnumType',
)

import enum
import inspect
import typing

Alias = typing._alias


class Enum(enum.Enum):

    @staticmethod
    def _check_methods(C, *methods):
        # collections.abc._check_methods
        mro = C.__mro__
        for method in methods:
            for B in mro:
                if method in B.__dict__:
                    if B.__dict__[method] is None:
                        return NotImplemented
                    break
            else:
                return NotImplemented
        return True

    @classmethod
    def asdict(cls):
        return {key: value._value_ for key, value in cls.__members__.items()}

    @classmethod
    def attrs(cls):
        return list(cls.__members__)

    @staticmethod
    def auto():
        return enum.auto()

    @classmethod
    def default(cls):
        return cls._member_map_[cls._member_names_[0]]

    @classmethod
    def default_attr(cls):
        return cls.attrs()[0]

    @classmethod
    def default_dict(cls):
        return {cls.default_attr(): cls.default_value()}

    @classmethod
    def default_value(cls):
        return cls[cls.default_attr()]

    @property
    def describe(self):
        """
        Returns:
            tuple:
        """
        # self is the member here
        return self.name, self.value

    @property
    def lower(self):
        return self.name.lower()

    @property
    def dot(self):
        return self.value if self.name == 'NO' else f'.{self.name.lower()}'

    def prefix(self, prefix):
        return f'{prefix}_{self.name}'

    @classmethod
    def values(cls):
        return list(cls.asdict().values())

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Enum:
            attrs = [C] + ['asdict', 'attrs', 'auto', 'default', 'default_attr', 'default_dict', 'default_value',
                           'describe', 'lower', 'dot', 'prefix', 'values', '_generate_next_value_', '_missing_',
                           'name', 'value'] + inspect.getmembers(C)
            return cls._check_methods(*attrs)
        return NotImplemented


EnumType = Alias(Enum, 1, name=Enum.__name__)
