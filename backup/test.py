# <editor-fold desc="Test">

class TestBase(Base):
    classvar: ClassVar[int] = 1
    initvar: InitVar[int] = 1
    __slots__ = ('_hash', '_prop', '_repr', '_slot',)
    __hash_exclude__ = ('_slot',)
    __repr_exclude__ = ('_repr',)
    prop = newprop()

    async def method_async(self):
        pass

    @classmethod
    def clsmethod(cls):
        pass

    @staticmethod
    def static():
        pass

    @pproperty
    def pprop(self):
        return 'pprop'

    @pproperty
    async def pprop_async(self):
        return 'pprop_async'


# </editor-fold>
