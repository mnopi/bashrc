import os
from pathlib import Path

from rc import cachemodule
from rc import conf
from rc import envbash
from rc import EnvironOS

conf_envbash = conf['defaults'][envbash.__name__]
tests_envbash = Path(__file__).parent / conf_envbash
keys = [f'TEST_{_i}' for _i in ['MACOS', 'LOGNAME', 'LOGNAMEHOME', 'ROOTHOME', 'LOGGEDINUSER',
                                'LOGNAMEREALNAME', 'MULTILINE']]


def check(env):
    for i in keys:
        assert env.get(i) is not None
        del env[i]


def test_envbash():
    rv = envbash()
    assert cachemodule[envbash][(Path.cwd(), None)] in [tests_envbash, Path(conf_envbash)]
    assert isinstance(rv, EnvironOS)
    assert id(rv) == id(os.environ)
    check(rv)


def test_envbash_into():
    into = {}
    rv = envbash(into=into)
    assert cachemodule[envbash][(Path.cwd(), None)] in [tests_envbash, Path(conf_envbash)]
    assert not isinstance(rv, EnvironOS)
    assert id(into) == id(rv)
    check(rv)


def test_envbash_new():
    rv = envbash(new=True)
    assert cachemodule[envbash][(Path.cwd(), None)] in [tests_envbash, Path(conf_envbash)]
    assert not isinstance(rv, EnvironOS)
    rv_copy = rv.copy()
    check(rv)

    for i in os.environ:
        assert i in rv_copy if i in keys else i not in rv_copy


def test_envbash_override():
    rv = envbash(override=False)
    assert cachemodule[envbash][(Path.cwd(), None)] in [tests_envbash, Path(conf_envbash)]
    assert not isinstance(rv, EnvironOS)
    assert id(rv) != id(os.environ)
    check(rv)


test_envbash()
test_envbash_into()
test_envbash_new()
test_envbash_override()
