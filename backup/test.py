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


@dataclass
class TestData:
    __data = varname(1)
    __dataclass_classvar__: ClassVar[str] = '__dataclass_classvar__'
    __dataclass_classvar: ClassVar[str] = '__dataclass_classvar'
    __dataclass_default_factory: Union[dict, str] = datafield(default_factory=dict, init=False)
    __dataclass_default_factory_init: Union[dict, str] = datafield(default_factory=dict)
    dataclass_classvar: ClassVar[str] = 'dataclass_classvar'
    dataclass_default_factory: Union[dict, str] = datafield(default_factory=dict, init=False)
    dataclass_default_factory_init: Union[dict, str] = datafield(default_factory=dict)
    dataclass_default: str = datafield(default='dataclass_default', init=False)
    dataclass_default_init: str = datafield(default='dataclass_default_init')
    dataclass_initvar: InitVar[str] = 'dataclass_initvar'
    dataclass_str: str = 'dataclass_integer'

    def __post_init__(self, dataclass_initvar): pass

    __class_getitem__ = classmethod(GenericAlias)


class TestDataDictMix(TestData):
    subclass_annotated_str: str = 'subclass_annotated_str'
    subclass_classvar: ClassVar[str] = 'subclass_classvar'
    subclass_str = 'subclass_str'

    def __init__(self, dataclass_initvar='dataclass_initvar_1', subclass_dynamic='subclass_dynamic'):
        super().__init__()
        super().__post_init__(dataclass_initvar=dataclass_initvar)
        self.subclass_dynamic = subclass_dynamic


class TestDataDictSlotMix(TestDataDictMix):
    __slots__ = ('_slot_property', 'slot',)

    # Add init=True dataclass attrs if it subclassed and not @dataclass
    def __init__(self, dataclass_initvar='dataclass_initvar_2', slot_property='slot_property', slot='slot'):
        super().__init__()
        super().__post_init__(dataclass_initvar=dataclass_initvar)
        self._slot_property = slot_property
        self.slot = slot

    @pproperty
    def slot_property(self):
        return self._slot_property
# </editor-fold>
