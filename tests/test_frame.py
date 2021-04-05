from rc import FUNCTION_MODULE
from rc import icc
from rc import Path
from rc import Stack
import tests.data.frame
from tests.data.frame import A
from tests.data.frame import context
from tests.data.frame import thread
from tests.data.frame import thread2


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
    icc(A.c_1[0].function.qual)
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
    icc(context)
    print()
    icc(thread)
    print()
    icc(thread2)


test_data_c()
test_data_asyncgen()
