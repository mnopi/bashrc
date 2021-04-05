from rc import FUNCTION_MODULE
from rc import icc
from rc import Path
from rc import Stack
import tests.data.frame
from tests.data.frame import A
from tests.data.frame import context
from tests.data.frame import thread
from tests.data.frame import thread2
from tests.data.frame import *


def test_data_c():
    assert Stack(init=True) != A.init
    assert (A.init[0].file.exists and A.init[0].file.include) is True
    assert (A.init[1].file.exists and A.init[1].file.include) is False
    assert (Stack(init=True)[1].file.exists and Stack(init=True)[1].file.include) is False

    assert len(A.c_1) > 2
    assert A.c_1[0].file.path == Path(tests.data.frame.__file__)
    assert A.c_1[0].file.exists is True
    assert A.c_1[0].file.include is True
    assert A.c_1[0].function.cls == A.__name__
    assert classmethod.__name__ in A.c_1[0].function.decorators
    assert A.c_1[0].function.module is False
    assert A.c_1[0].function.qual == A.c.__qualname__
    assert A.c_1[0].function.sync is True
    assert A.c_1[0].function.name == A.c.__name__
    assert A.c_1[0].var.args == dict()

    assert A.c_1[1].file.path == Path(tests.data.frame.__file__)
    assert A.c_1[1].file.exists is True
    assert A.c_1[1].file.include is True
    assert A.c_1[1].function.name == FUNCTION_MODULE
    assert A.c_1[1].var.args == dict()


def test_data_asyncgen():
    assert context.no_sync.caller == context.no_sync.stack[CALL_INDEX]
    assert context.no_sync.stack[0].file.INCLUDE is True
    assert context.no_sync.stack[0].function.decorators == list()
    assert context.no_sync.stack[0].function.name == asynccontext_call_async.__name__
    assert context.no_sync.stack[0].info.real == 0
    assert context.no_sync.stack[0].info.ASYNC is True
    assert context.no_sync.stack[0].line.ASYNC is True
    assert context.no_sync.stack[CALL_INDEX].file.INCLUDE is True
    assert context.no_sync.stack[CALL_INDEX].function.name == asynccontext.__name__
    assert context.no_sync.stack[CALL_INDEX].info.real == 0
    assert context.no_sync.stack[CALL_INDEX].info.ASYNC is True
    icc(context)
    # print()
    # icc(thread)
    # print()
    # icc(thread2)


test_data_c()
test_data_asyncgen()
