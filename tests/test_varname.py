# coding=utf-8
from dataclasses import dataclass

from rc import varname


def function() -> str:
    return varname()


class ClassTest:
    def __init__(self):
        self.name = varname()

    @property
    def prop(self):
        return varname()

    # noinspection PyMethodMayBeStatic
    def method(self):
        return varname()


@dataclass
class DataClassTest:
    def __post_init__(self):
        self.name = varname()


name = varname(1)
Function = function()
classtest = ClassTest()
method = classtest.method()
prop = classtest.prop
dataclasstest = DataClassTest()


def test_var():
    assert name == 'name'


def test_function():
    assert Function == function.__name__.lower()


def test_class():
    assert classtest.name == ClassTest.__name__.lower()


def test_method():
    assert classtest.method() == ClassTest.__name__.lower()
    assert method == 'method'


def test_property():
    assert classtest.prop == ClassTest.__name__.lower()
    assert prop == 'prop'


def test_dataclass():
    assert dataclasstest.name == DataClassTest.__name__.lower()
