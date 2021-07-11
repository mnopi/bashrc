import asyncio
import sys
import tempfile
from pathlib import Path

import pytest

import rc
from rc import Env
from rc import Pname
from rc import top

env = Env()

rc_spec = rc.__spec__
rc_file, rc_name, rc_path = Path(rc_spec.origin), rc_spec.name, Path(rc_spec.submodule_search_locations[0])
rc_root = rc_path.parent


def test_env():
    assert env.top.file != env.top.path
    assert env.top.path == Path(__file__).parent
    assert env.top.git_dir == rc.env.top.git_dir
    assert env.top.init_py == env.top.path / Pname.INIT_.py()
    assert env.top.installed is None
    assert env.top.name != Path(__file__).stem
    assert env.top.name == Path(__file__).parent.name
    assert env.top.prefix == f'{env.top.name.upper()}_'
    assert env.top.pyproject_toml == rc.env.top.pyproject_toml
    assert env.top.root == rc.env.top.root
    assert env.top.setup_cfg == rc.env.top.setup_cfg
    assert env.top.setup_py == rc.env.top.setup_py


def test_top_asyncio():
    asyncio_spec = asyncio.__spec__
    asyncio_file, asyncio_name = Path(asyncio_spec.origin), asyncio_spec.name
    asyncio_path = Path(asyncio_spec.submodule_search_locations[0])
    asyncio_root = asyncio_path.parent

    asyncio_top = top(asyncio.__file__)
    assert asyncio_top.file == asyncio_file
    assert asyncio_top.git_dir is None
    assert asyncio_top.init_py == asyncio_file
    assert asyncio_top.installed is True
    assert asyncio_top.name == asyncio_name
    assert asyncio_top.path == asyncio_path
    assert asyncio_top.prefix == f'{asyncio_name.upper()}_'
    assert asyncio_top.pyproject_toml is None
    assert asyncio_top.root == asyncio_root
    assert asyncio_top.root.name == Path(sys.executable).resolve().name
    assert asyncio_top.setup_cfg is None
    assert asyncio_top.setup_py is None


def test_top_pytest():
    pytest_spec = pytest.__spec__
    pytest_file, pytest_name = Path(pytest_spec.origin), pytest_spec.name
    pytest_path = Path(pytest_spec.submodule_search_locations[0])
    pytest_root = pytest_path.parent

    pytest_top = top(pytest.__file__)
    assert pytest_top.file == pytest_file
    assert pytest_top.git_dir is None
    assert pytest_top.init_py == pytest_file
    assert pytest_top.installed is True
    assert pytest_top.name == pytest_name
    assert pytest_top.path == pytest_path
    assert pytest_top.prefix == f'{pytest_name.upper()}_'
    assert pytest_top.pyproject_toml is None
    assert pytest_top.root == pytest_root
    assert pytest_top.root.name == 'site-packages'
    assert pytest_top.setup_cfg is None
    assert pytest_top.setup_py is None


def test_top_tmp_module():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tmp_file = tmp_path / 'tmp.py'
        tmp_file.touch()
        tmp_name = tmp_file.stem
        tmp_root = tmp_path
        tmp_path = tmp_file

        tmp_top = top(tmp_file)
        assert tmp_top.file == tmp_file
        assert tmp_top.git_dir is None
        assert tmp_top.init_py is None
        assert tmp_top.installed is None
        assert tmp_top.name == tmp_name
        assert tmp_top.path == tmp_path
        assert tmp_top.prefix == f'{tmp_name.upper()}_'
        assert tmp_top.pyproject_toml is None
        assert tmp_top.root == tmp_root
        assert tmp_top.root == tmp_top.path.parent
        assert tmp_top.setup_cfg is None
        assert tmp_top.setup_py is None


def test_rc():
    assert rc.env.top.file == rc_file
    assert rc.env.top.git_dir == rc_root / Pname.GIT.dot
    assert rc.env.top.init_py == rc_file
    assert rc.env.top.installed is None
    assert rc.env.top.name == rc_name
    assert rc.env.top.path == rc_path
    assert rc.env.top.prefix == f'{rc_name.upper()}_'
    assert rc.env.top.pyproject_toml == rc_root / Pname.PYPROJECT.toml()
    assert rc.env.top.root == rc_root
