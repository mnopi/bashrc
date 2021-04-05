import asyncio
import inspect
from asyncio import as_completed
from asyncio import create_task
from asyncio import to_thread
from contextlib import asynccontextmanager
from typing import Optional

from async_property import async_property

from rc import *


class A:
    init: Stack = Stack(init=True)
    c_1: Optional[Stack] = None

    @classmethod
    def c(cls):
        cls.c_1 = Stack()
        cls.s()
        pass

    # noinspection PyMethodMayBeStatic
    def m(self):
        return 'm'

    @property
    def p(self):
        return 'p'

    @staticmethod
    def s():
        rv = Stack()
        return rv

    @async_property
    async def a(self):
        # noinspection PyUnusedLocal
        b = await to_thread(self.m)
        await to_thread(self.m)
        return await to_thread(self.m)

    # noinspection PyPropertyDefinition
    @property
    async def n(self):
        pass

    @staticmethod
    def z():
        pass

    @staticmethod
    async def d():
        pass

    @classmethod
    async def e(cls):
        pass

    async def f(self):
        pass


def func():
    pass


async def asynccontext_call_async():
    s = Stack()
    return s


def asynccontext_call_sync():
    s = Stack()
    return s


@asynccontextmanager
async def asynccontext():
    no_sync = await asynccontext_call_async()
    sync = await to_thread(asynccontext_call_sync)
    try:
        yield no_sync, sync
    finally:
        pass


async def asyncyield():
    a = [1, 2]
    while a:
        yield a.pop()


async def asyncdef2():
    a = [1, 2]
    while a:
        return a.pop()


async def asyncdef(ctx: bool = False, thrd1: bool = False, thrd2: bool = False):
    result = list()
    if ctx:
        async with \
                asynccontext() as rv:
            return rv

    if thrd1:
        rv = await \
            to_thread(A.s)
        return rv

    if thrd2:
        return \
            await \
            to_thread(A.s)

    for coro in as_completed([A.d(),
                              A.d()]):
        r = await coro
        result.append(r)

    # noinspection PyUnusedLocal
    async for a in asyncyield():
        pass
    # noinspection PyUnusedLocal
    async for a in \
            asyncyield():
        pass
    # noinspection PyUnusedLocal
    a = await A().a
    # No se puede crear tarea si es asyncyield
    # noinspection PyUnusedLocal
    task = create_task(
        asyncdef2()
    )

    task1 = create_task(
        asyncdef2()
    )

    await task1

    await asyncio.ensure_future(
        asyncdef2()
    )
    await asyncio.gather(
        asyncdef2()
    )

A.c()

context = asyncio.run(asyncdef(ctx=True), debug=False)
thread = asyncio.run(asyncdef(thrd1=True), debug=False)
thread2 = asyncio.run(asyncdef(thrd2=True), debug=False)
