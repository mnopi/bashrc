import itertools
from pathlib import Path

import rc
from rc import Action
from rc import conf
from rc import Env
from rc import getlog
from rc import LOG
from rc import LogCall
from rc import Pname
from rc import trace

file = Path(__file__)
env_file = file.parent / Pname.ENV.dot
jlogfile = LOG / Pname.NONE.json(file.parent.name)
logfile = LOG / Pname.NONE.log(file.parent.name)
jlogfile_rc = LOG / Pname.NONE.json('rc')
logfile_rc = LOG / Pname.NONE.log('rc')
env = Env()
log = getlog(std_add=('name', 'line', ))


def rc_env():
    assert rc.env.prefix == 'RC_'
    assert rc.env.path == env_file
    assert rc.env.top.jlogfile == jlogfile_rc
    assert rc.env.top.logfile == logfile_rc


def rc_log():
    assert rc.log.top_name == 'rc'
    assert rc.log.cached is True and rc.log.deepcopied is True
    assert repr(rc.log._core.handlers_count) == repr(itertools.count(3))
    assert rc.log._core.handlers[0].levelno == conf['log']['number']['ERROR']  # `RC_LEVEL_STD` in .env.
    assert rc.log._core.handlers[0]._name == '<stderr>'
    assert rc.log._core.handlers[1].levelno == conf['log']['number'][conf['log']['default']['level_file']]
    assert Path(rc.log._core.handlers[1]._name).name == "rc.log'"
    rc.log.info('rc')


def rc_logger():
    # 1 because getlog() with copy=False ut it has been deepcopy since it is the first time for rc.
    assert repr(rc.logger._core.handlers_count) == repr(itertools.count(1))
    assert rc.logger._core.handlers[0].levelno == 10
    assert rc.logger._core.handlers[0]._name == '<stderr>'


def log_env():
    assert env.prefix == 'TESTS_'
    assert env.path == env_file
    assert env.top.jlogfile == jlogfile
    assert env.top.logfile == logfile


def log_log():
    assert log.top_name == 'tests'
    assert repr(log._core.handlers_count) == repr(itertools.count(3))
    assert log._core.handlers[0].levelno == conf['log']['number'][conf['log']['default']['level_std']]
    assert log._core.handlers[0]._name == '<stderr>'
    assert log._core.handlers[1].levelno == conf['log']['number']['TRACE']  # `TEST_LOG_LEVEL_STD` in .env.
    assert Path(log._core.handlers[1]._name).name == "tests.log'"
    log.info('tests')


@trace(l=getlog(std_default=True, copy=True), add_end=('test', ), released=LogCall.INFO, start=LogCall.ERROR)
def log_trace_loguru_default():
    name = log_trace_loguru_default.__name__
    log.info(name)
    log_trace_loguru_default.test = name
    log_trace_loguru_default.action = Action.RELEASED


@trace(l=getlog(std_add=('name', 'module',), copy=True), add_end=('test', ), released=LogCall.SUCCESS,
       start=LogCall.ERROR)
def log_trace_std_add_name():
    name = log_trace_std_add_name.__name__
    log.info(name)
    log_trace_std_add_name.test = name
    log_trace_std_add_name.action = Action.RELEASED


# noinspection PyUnusedLocal
@trace(add_end=('test',), add_start=True, released=LogCall.SUCCESS, start=LogCall.WARNING)
def log_trace_globals(arg1, kwarg_1='kwarg_1'):
    name = log_trace_globals.__name__
    log.info(name)
    log_trace_globals.test = name
    log_trace_globals.action = Action.RELEASED


def check_jlogfile():
    jlogload = env.top.jlogload(chain=False)
    assert jlogload[0]['module'] == '__init__'
    assert jlogload[0]['name'] == 'log'
    jlogload = env.top.jlogload()
    assert jlogload['module'] == '__init__'
    assert jlogload['name'] == 'log'


if __name__ == '__main__':
    rc_env()
    rc_log()
    rc_logger()
    log_env()
    log_log()
    log_trace_loguru_default()
    log_trace_std_add_name()
    log_trace_globals(1)
    check_jlogfile()
