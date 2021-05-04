class Base:
    """
    Base Helper Class.

    Attributes:
    -----------
        __slots__: tuple[str]
            slots (default: tuple()).
        __hash_exclude__: tuple[str]
            Exclude slot attr for hash (default: tuple()).
        __ignore_attr__: tuple[str]
            Exclude instance attribute (default: tuple()).
        __ignore_copy__: tuple[Type, ...]
            True or class for repr instead of nested asdict and deepcopy. No deepcopy (default: tuple()).
        __ignore_kwarg__: tuple[str]
            Exclude attr from kwargs (default: tuple()).
        __ignore_str__: tuple[Type, ...]
            Use str value for object (default: tuple()).
        __repr_exclude__: tuple[str]
            Exclude slot attr for repr (default: tuple()).
        __repr_newline__: bool
            New line per attr in repr (default: True).
        __repr_pproperty__: bool
            Include :class:`pproperty` in repr (default: True).

    Methods:
    --------
        __getattribute__(item, default=None)
            :class:`function`:  Sets ``None`` as default value is attr is not defined.
        __hash__(self)
            :class:`function`: hash
        __repr__(self)
            :class:`function`: repr
        get(cls, name, default=None)
            :class:`function`: Get attribute value.
        info(self, key=Access.PRIVATE)
            :class:`function`: :class:`info`

    Examples:
    ---------
        >>> from rc import Base
        >>> from rc import Cls
        >>> from rc import pproperty
        >>> from rc import pretty_install
        >>> from rc import TestBase
        >>>
        >>> pretty_install()
        >>> test = TestBase()
        >>>
        >>> sorted(Get.hash_exclude.val(test))
        ['_slot']
        >>> sorted(Get.ignore_attr.val(test))
        []
        >>> Get.ignore_copy.val(test).difference(Get.ignore_copy.default)
        set()
        >>> sorted(Get.ignore_kwarg.val(test))
        []
        >>> Get.ignore_str.val(test).difference(Get.ignore_str.default)
        set()
        >>> sorted(Get.repr_exclude.val(test))
        ['_repr']
        >>> sorted(Get.slots.val(test))
        ['_hash', '_prop', '_repr', '_slot']
        >>> Get.repr_newline.mro_first_attr_value_in_obj(test)
        True
        >>> Get.repr_pproperty.mro_first_attr_value_in_obj(test)
        True
        >>> Get.slot(test, '_hash')
        True
        >>>
        >>> test.info.cls.name
        'TestBase'
        >>> repr(test)  # doctest: +ELLIPSIS
        'TestBase(_hash: None,\\n_prop: None,\\n_slot: None,\\npprop: pprop)'
        >>> assert test.__repr_exclude__[0] not in repr(test)
        >>> test.prop
        >>> test.prop = 2
        >>> test.prop
        2
        >>> del test.prop
        >>> test.prop
        >>> assert hash((test._hash, test._prop, test._repr)) == hash(test)
        >>> set(test.__slots__).difference(test.info.cls.data_attrs)
        set()
        >>> sorted(test.info.cls.data_attrs)
        ['_hash', '_prop', '_repr', '_slot', 'classvar', 'initvar']
        >>> '__slots__' in sorted(test.info().cls.data_attrs)  # info() __call__(key=Access.ALL)
        True
        >>> test.get.__name__ in test.info.cls.method
        True
        >>> test.get.__name__ in test.info.cls.callable
        True
        >>> 'prop' in test.info.cls.prop
        True
        >>> 'pprop' in test.info.cls.pproperty
        True
        >>> test.info.cls.importable_name  # doctest: +ELLIPSIS
        '....TestBase'
        >>> test.info.cls.importable_name == f'{test.info.cls.modname}.{test.info.cls.name}'
        True
        >>> test.info.cls.qualname
        'TestBase'
        >>> test.info.cls.attr_value('pprop')  # doctest: +ELLIPSIS
        <....pproperty object at 0x...>
        >>> test.info.attr_value('pprop')
        'pprop'
        >>> test.info.module  # doctest: +ELLIPSIS
        <module '...' from '/Users/jose/....py'>
        >>> assert sorted(test.info.dir) == sorted(test.info.cls.dir)
        >>> sorted(test.info.cls.memberdescriptor)
        ['_hash', '_prop', '_repr', '_slot']
        >>> sorted(test.info.cls.memberdescriptor) == sorted(test.__slots__)
        True
        >>> test.info.cls.methoddescriptor  # doctest: +ELLIPSIS
        [
            '__delattr__',
            ...,
            '__subclasshook__',
            'clsmethod',
            'static'
        ]
        >>> sorted(test.info.cls.method)
        ['_info', 'get', 'method_async']
        >>> sorted(test.info.cls().method)  # doctest: +ELLIPSIS
        [
            '__delattr__',
            ...,
            '__str__',
            '_info',
            'get',
            'method_async'
        ]
        >>> sorted(test.info.cls().callable) == \
        sorted(list(test.info.cls().method) + list(test.info.cls().classmethod) + list(test.info.cls().staticmethod))
        True
        >>> test.info.cls.setter
        ['prop']
        >>> test.info.cls.deleter
        ['prop']
        >>> test.info.cls.is_attr('_hash')
        True
        >>> test.info.cls.is_data('_hash')
        True
        >>> test.info.cls.is_deleter('prop')
        True
        >>> test.info.cls.is_memberdescriptor('_hash')
        True
        >>> test.info.cls.is_method('__repr__')
        True
        >>> test.info.cls.is_methoddescriptor('__repr__')
        False
        >>> test.info.cls.is_methoddescriptor('__str__')
        True
        >>> test.info.cls.is_pproperty('pprop')
        True
        >>> test.info.cls.is_property('prop')
        True
        >>> test.info.cls.is_routine('prop')
        False
        >>> test.info.cls.is_setter('prop')
        True
        >>> test.info.cls.is_attr('classvar')
        True
        >>> test.info.is_attr('classvar')
        True
        >>> sorted(test.info.coros)
        ['method_async', 'pprop_async']
        >>> test.info.coros_pproperty
        ['pprop_async']
        >>> test.info.coros_property
        ['pprop_async']
        >>> test.info.is_coro('pprop_async')
        True
        >>> test.info.is_coro('method_async')
        True
        >>> test.info.is_coro_pproperty('pprop_async')
        True
        >>> test.info.is_coro_property('pprop_async')
        True
        >>> test.info.cls.classvar
        ['classvar']
        >>> test.info.cls.initvar
        ['initvar']
        >>> test.info.cls.is_classvar('classvar')
        True
        >>> test.info.cls.is_initvar('initvar')
        True
    """
    __slots__ = tuple()
    __hash_exclude__ = tuple()
    """Exclude slot attr for hash (default: tuple())."""
    __ignore_attr__ = tuple()
    """Exclude instance attribute (default: tuple())."""
    __ignore_copy__ = tuple()
    """True or class for repr instead of nested asdict and deepcopy. No deepcopy (default: tuple())"""
    __ignore_kwarg__ = tuple()
    """Exclude attr from kwargs (default: tuple())."""
    __ignore_str__ = tuple()
    """Use str value for object (default: tuple())."""
    __repr_exclude__ = tuple()
    """Exclude slot attr for repr (default: tuple())."""
    __repr_newline__ = True
    """New line per attr in repr (default: True)."""
    __repr_pproperty__ = True
    """Include :class:`pproperty` in repr (default: True)."""

    def __getattribute__(self, name, default=None):
        """
        Sets attribute:
            - with None if it does not exists and involked with getattr()
                # getattr does not pass default to __getattribute__
            - with default if called directly.

        Examples:
            >>> from rc import Base
            >>> class Dict(Base): pass
            >>> class Slots(Base): __slots__ = ('a', )
            >>>
            >>> d = Dict()
            >>> s = Slots()
            >>> getattr(d, 'a')
            >>> d.a
            >>> getattr(s, 'a')
            >>> s.a
            >>>
            >>> d = Dict()
            >>> s = Slots()
            >>> getattr(d, 'a', 2)  # getattr does not pass default to __getattribute__
            >>> d.a
            >>> getattr(s, 'a', 2)  # getattr does not pass default to __getattribute__
            >>> s.a
            >>>
            >>> d = Dict()
            >>> s = Slots()
            >>> d.__getattribute__('a', 2)
            2
            >>> d.a
            2
            >>> s.__getattribute__('a', 2)
            2
            >>> s.a
            2
            >>>
            >>> class Dict(Base): a = 1
            >>> class Slots(Base):
            ...     __slots__ = ('a', )
            ...     def __init__(self):
            ...         self.a = 1
            >>> d = Dict()
            >>> s = Slots()
            >>> getattr(d, 'a')
            1
            >>> getattr(s, 'a')
            1
            >>> getattr(d, 'a', 2)
            1
            >>> getattr(s, 'a', 2)
            1

        Args:
            name: attr name.
            default: default value (default: None).

        Returns:
            Attribute value or sets default value and returns.
        """
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            object.__setattr__(self, name, default)
            return object.__getattribute__(self, name)

    def __hash__(self): return self.info.hash

    def __repr__(self): return self.info.repr

    def get(self, name, default=None):
        """
        Sets attribute:
            - with None if it does not exists and involked with getattr()
                # getattr does not pass default to __getattribute__
            - with default if called directly.

        Examples:
            >>> from rc import Base
            >>> class Dict(Base): pass
            >>> class Slots(Base): __slots__ = ('a', )
            >>>
            >>> d = Dict()
            >>> s = Slots()
            >>> d.get('a')
            >>> d.a
            >>> s.get('a')
            >>> s.a
            >>>
            >>> d = Dict()
            >>> s = Slots()
            >>> getattr(d, 'a', 2)  # getattr does not pass default to __getattribute__
            >>> d.a
            >>> getattr(s, 'a', 2)  # getattr does not pass default to __getattribute__
            >>> s.a
            >>>
            >>> d = Dict()
            >>> s = Slots()
            >>> d.get('a', 2)
            2
            >>> d.a
            2
            >>> s.get('a', 2)
            2
            >>> s.a
            2
            >>>
            >>> class Dict(Base): a = 1
            >>> class Slots(Base):
            ...     __slots__ = ('a', )
            ...     def __init__(self):
            ...         self.a = 1
            >>> d = Dict()
            >>> s = Slots()
            >>> d.get('a')
            1
            >>> s.get('a')
            1
            >>> d.get('a', 2)
            1
            >>> s.get('a', 2)
            1
            >>>
            >>> class Dict(Base, dict): pass
            >>> d = Dict()
            >>> d.get('a', 2)
            2
            >>> d.a  # dict not super().__init__(dict(a=2)

        Args:
            name: attr name.
            default: default value (default: None).

        Returns:
            Attribute value or sets default value and returns.
        """
        if hasattr(self, '__getitem__'):
            if self.__getitem__ is not None:
                try:
                    rv = self.__getitem__(name)
                except KeyError:
                    self[name] = default
                    rv = self.__getitem__(name)
                return rv
        return getset(self, name, default)

    def _info(self, depth=None, ignore=False, key=Attr.PRIVATE): return self.info(depth=depth, ignore=ignore, key=key)

    @property
    def info(self): return info(self)
