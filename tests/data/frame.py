from dataclasses import dataclass
from typing import AsyncIterable
from typing import Optional
from typing import Union
from types import Me
from rc import *
from rc.enums import Executor

@dataclass
class CallerStack:
    caller: Frame
    stack: Stack


@dataclass
class NoSyncSync:
    no_sync: CallerStack
    sync: CallerStack


class A:
    init: Stack = Stack(init=True)
    c_1: Optional[Stack] = None

    @classmethod
    def c(cls):
        cls.c_1 = Stack()
        cls.s()
        pass

    # noinspection PyMethodMayBeStatic
    def m(self) -> Stack:
        return Stack()

    @property
    def p(self):
        return 'p'

    @staticmethod
    def s() -> Stack:
        rv = Stack()
        return rv

    @async_property
    async def a(self) -> Stack:
        b = await to_thread(self.m)
        return b

    # noinspection PyPropertyDefinition
    @property
    async def n(self):
        pass

    @staticmethod
    def z():
        pass

    @staticmethod
    async def d() -> Stack:
        s = Stack()
        return s

    @classmethod
    async def e(cls):
        pass

    async def f(self):
        pass


def func():
    pass


async def asynccontext_call_async() -> CallerStack:
    s = Stack()
    rv = CallerStack(caller=s(), stack=s)
    return rv


def asynccontext_call_sync() -> CallerStack:
    s = Stack()
    rv = CallerStack(caller=s(), stack=s)
    return rv


@asynccontextmanager
async def asynccontext() -> NoSyncSync:
    no_sync = await asynccontext_call_async()
    sync = await Executor.NONE.run(asynccontext_call_sync)
    rv = NoSyncSync(no_sync=no_sync, sync=sync)
    try:
        yield rv
    finally:
        pass


async def rv_async() -> Stack:
    return Stack()


async def asyncyield() -> AsyncIterable:
    a = 1
    while a:
        a -= 1
        yield rv_async


async def asyncdef(ctx: bool = False, thrd: bool = False,
                   cmp: bool = False, fyld: bool = False, pth: bool = False, tsk: bool = False,
                   ens: bool = False, gth: bool = False) -> Union[NoSyncSync, Stack]:
    if ctx:
        async with \
                asynccontext() as rv:
            return rv

    if thrd:
        return \
            await to_thread(A.s)

    if cmp:
        result = list()
        for coro in as_completed([A.d(),
                                  A.d()]):
            r = await coro
            result.append(r)
        return result[0]

    if fyld:
        async for a in \
                asyncyield():
            return await a()

    if pth:
        return await A().a

    if tsk:
        # No se puede crear tarea si es asyncyield
        t = create_task(
            rv_async()
        )
        return await t

    if ens:
        return await asyncio.ensure_future(
            rv_async()
        )

    if gth:
        results = await asyncio.gather(rv_async())
        return results[0]


A.c()

context = asyncio.run(asyncdef(ctx=True), debug=False)
thread = asyncio.run(asyncdef(thrd=True), debug=False)
completed = asyncio.run(asyncdef(cmp=True), debug=False)
foryield = asyncio.run(asyncdef(fyld=True), debug=False)
prop_to_thread = asyncio.run(asyncdef(pth=True), debug=False)
task = asyncio.run(asyncdef(tsk=True), debug=False)
ensure = asyncio.run(asyncdef(ens=True), debug=False)
