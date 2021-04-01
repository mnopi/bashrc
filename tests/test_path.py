# -*- coding: utf-8 -*-
from contextlib import suppress
from shutil import which

import pytest

import rc
from rc import *


def test_c():
    new = Path('/usr/local').resolved
    pwd = Path.cwd()
    with new.cd as previous:
        assert new == Path.cwd()
        assert previous == pwd
    assert pwd == Path.cwd()


def test_call_rm(tmp_path):
    tmp = Path(tmp_path)
    name = 'dir'
    p = tmp(name)
    assert p.is_dir()
    p.rm()
    assert not p.is_dir()
    name = 'file'
    p = tmp(name, FILE_DEFAULT)
    assert p.is_file()
    p.rm()
    assert not p.is_file()
    assert Path('/tmp/a/a/a/a')().is_dir()


def test_cd(cwd):
    previous = Path(cwd)
    local = previous.c_('/usr/local')
    usr = local.parent
    assert usr.text == usr.str == str(usr.resolved)
    assert Path.cwd() == usr.cwd()
    assert 'local' in local
    assert local.has('usr local')
    assert not local.has('usr root')
    assert local.c_() == previous.resolved


def test_gittop():
    assert Path.cwd().gittop() == Path.gittop(Path(__file__).parent.name)
    tmp = Path.gittop('/tmp')
    assert all([tmp.name is None, tmp.origin is None, tmp.path is None])


def test_install():
    assert Path(__file__).installed is None
    assert Path(rc.__file__).installed is None
    assert Path(which(pytest.__name__)).installed
    assert Path(pytest.__file__).installed


def test_template(tmp_path):
    tmp = Path(tmp_path)
    p = tmp('templates')
    filename = 'sudoers'
    f = p(f'{filename}{PathSuffix.J2.dot}', FILE_DEFAULT)
    name = User().name
    template = 'Defaults: {{ name }} !logfile, !syslog'
    value = f'Defaults: {name} !logfile, !syslog'
    f.write_text(template)
    assert p.j2(stream=False)[filename] == value
    p.j2(dest=p)
    assert p(filename, FILE_DEFAULT).read_text() == value


def run():
    with Path.tmp() as p:
        test_call_rm(p)
        test_template(p)
    test_c()
    test_cd(Path.cwd())
    test_gittop()
    test_install()


with suppress(ModuleNotFoundError):
    from conftest import ic
    if ic.enabled:
        run()
