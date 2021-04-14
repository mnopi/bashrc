import tests.data.frame
from tests.data.frame import *


def test_data_c():
    assert Stack(init=True) != A.init
    assert (A.init[0].file.exists and A.init[0].file.include) is True
    assert A.init[0].info.external and A.init[1].info.external is False
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
    assert A.c_1[0].function.ASYNC is False
    assert A.c_1[0].function.name == A.c.__name__
    assert A.c_1[0].var.args == dict()

    assert A.c_1[1].file.path == Path(tests.data.frame.__file__)
    assert A.c_1[1].file.exists is True
    assert A.c_1[1].file.include is True
    assert A.c_1[1].function.name == FUNCTION_MODULE
    assert A.c_1[1].var.args == dict()


def test_context():
    assert context.no_sync.caller == context.no_sync.stack[FRAME_INDEX]
    assert context.no_sync.stack[0].file.exists is True
    assert context.no_sync.stack[0].file.include is True
    assert context.no_sync.stack[0].function.ASYNC is True
    assert context.no_sync.stack[0].function.decorators == list()
    assert context.no_sync.stack[0].function.name == asynccontext_call_async.__name__
    assert context.no_sync.stack[0].info.ASYNC is True
    assert context.no_sync.stack[0].info.real == 0
    assert context.no_sync.stack[0].line.ASYNC is False
    assert context.no_sync.stack[FRAME_INDEX].file.exists is True
    assert context.no_sync.stack[FRAME_INDEX].file.include is True
    assert context.no_sync.stack[FRAME_INDEX].function.ASYNC is True
    assert context.no_sync.stack[FRAME_INDEX].function.decorators == [asynccontextmanager.__name__]
    assert context.no_sync.stack[FRAME_INDEX].function.name == asynccontext.__name__
    assert context.no_sync.stack[FRAME_INDEX].info.ASYNC is True
    assert context.no_sync.stack[FRAME_INDEX].info.real == 0
    assert context.no_sync.stack[FRAME_INDEX].line.ASYNC is False


def test_thread():
    # icc(thread)
    assert thread[0].function.decorators == [staticmethod.__qualname__]
    assert thread[0].function.qual == A.s.__qualname__
    assert thread[0].info.ASYNC is False
    assert thread[FRAME_INDEX].info.real is None
    caller = thread()
    real = thread(real=True)
    assert caller == real.caller == real.real


def test_completed():
    assert completed[0].function.ASYNC is True
    assert completed[0].function.decorators == [staticmethod.__qualname__]
    assert completed[0].function.qual == A.d.__qualname__
    assert completed[0].info.ASYNC is True
    assert completed[0].line.ASYNC is False
    icc(completed)
    caller = completed()  # 1
    assert caller.info.ASYNC is True
    assert caller.path.has(asyncio.__name__)

    real = completed(real=True)
    index = completed.index(real)  # 6
    assert real.real.info.module is True
    assert completed[FRAME_INDEX].info.real is None
    caller = completed()
    real = completed(real=True)
    # icc(caller, real)
    # real.real.info.module is True
    # assert caller == real.caller == real.real


def test_foryield():
    pass
    # icc(foryield)
    # assert thread[0].function.qual == A.s.__qualname__
    # assert thread[0].info.ASYNC is False
    # assert thread[FRAME_INDEX].info.real is None
    # caller = thread()
    # real = thread(real=True)
    # icc(caller, real)
    # assert caller == real.caller == real.real


def test_prop_to_thread():
    pass
    # icc(prop_to_thread)
    # assert thread[0].function.qual == A.s.__qualname__
    # assert thread[0].info.ASYNC is False
    # assert thread[FRAME_INDEX].info.real is None
    # caller = thread()
    # real = thread(real=True)
    # icc(caller, real)
    # assert caller == real.caller == real.real


def test_task():
    pass
    # icc(task)
    # assert thread[0].function.qual == A.s.__qualname__
    # assert thread[0].info.ASYNC is False
    # assert thread[FRAME_INDEX].info.real is None
    # caller = thread()
    # real = thread(real=True)
    # icc(caller, real)
    # assert caller == real.caller == real.real


def test_ensure():
    pass
    # icc(ensure)
    # assert thread[0].function.qual == A.s.__qualname__
    # assert thread[0].info.ASYNC is False
    # assert thread[FRAME_INDEX].info.real is None
    # caller = thread()
    # real = thread(real=True)
    # icc(caller, real)
    # assert caller == real.caller == real.real


def test_gather():
    # gather = asyncio.run(asyncdef(gth=True), debug=False)

    pass
    # icc(gather)
    # assert thread[0].function.qual == A.s.__qualname__
    # assert thread[0].info.ASYNC is False
    # assert thread[FRAME_INDEX].info.real is None
    # caller = thread()
    # real = thread(real=True)
    # icc(caller, real)
    # assert caller == real.caller == real.real


test_data_c()
test_context()
test_thread()
test_completed()
test_foryield()
test_prop_to_thread()
test_task()
test_ensure()
test_gather()
