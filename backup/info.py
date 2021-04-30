class info:
    """
    Is Instance, etc. Helper Class

    Attributes:
    -----------
    __slots__: tuple
        slots (default: tuple()).
    __ignore_attr__: tuple
        Exclude instance attribute (default: tuple()).
    __ignore_copy__: tuple
        True or class for repr instead of nested asdict and deepcopy. No deepcopy (default: tuple()).
    __ignore_kwarg__: tuple
        Exclude attr from kwargs (default: tuple()).
    __ignore_str__: tuple
        Use str value for object (default: tuple()).
    _data: Any
        object to provide information (default: None)
    _depth: Optional[int]
        recursion depth (default: None)
    _ignore: bool
        ignore properties and kwargs :class:`info.__ignore_kwargs__` (default: False)
    _key: :class:`Key`
        keys to include (default: :attr:`rc.Key.PRIVATE`)
    cls: :class:`Cls`
        :class:`Cls` (default: Cls(_data, _ignore, _ley))
    es: :class:`Es`
        :class:`Es` (default: Es(data))
   """
    __slots__ = ('_data', '_depth', '_ignore', '_key',)
    __ignore_attr__ = tuple()
    """Exclude instance attribute (default: tuple())."""
    __ignore_copy__ = tuple()
    """True or class for repr instead of nested asdict and deepcopy. No deepcopy (default: tuple())"""
    __ignore_kwarg__ = tuple()
    """Exclude attr from kwargs (default: tuple())."""
    __ignore_str__ = tuple()
    """Use str value for object (default: tuple())."""

    def __init__(self, data, depth=None, ignore=False, key=Attr.PRIVATE):
        self.data = data
        self.depth = depth
        self.ignore = ignore
        self.key = key

    def __call__(self, depth=None, ignore=False, key=Attr.ALL):
        self.depth = depth or self.depth
        self.ignore = ignore or self.ignore
        self.key = key or self.key
        return self

    def annotations(self, stack=2):
        """
        Object Annotations.

        Examples:
            >>> from dataclasses import dataclass
            >>> from dataclasses import InitVar
            >>> from typing import ClassVar
            >>> from rc import info
            >>>
            >>> @dataclass
            ... class Test:
            ...     a: int = 1
            ...     initvar: InitVar[int] = 1
            ...     classvar: ClassVar[int] = 2
            ...     def __post_init__(self, initvar: int):
            ...         self.a = initvar
            >>> ann = info(Test).annotations()
            >>> ann['a'].cls, ann['a'].default
            (<class 'int'>, 0)
            >>> ann['initvar'].cls, ann['initvar'].initvar, ann['initvar'].default
            (<class 'int'>, True, 0)
            >>> ann['classvar'].cls, ann['classvar'].classvar, ann['classvar'].default
            (<class 'int'>, True, 0)

        Args:
            stack: stack index to extract globals and locals (default: 2) or frame.

        Returns:
            Object Annotations.
        """
        return annotations(self.data, stack=stack)

    # noinspection PyUnusedLocal
    def asdict(self, count: int = ..., defaults: bool = ...):
        """
        Dict excluding.

        Returns:
            dict:
        """
        # convert = self.depth is None or self.depth > 1
        # # if self.es().enumenuminstance:
        # #     self.data = {self.data.name: self.data.value}
        # # elif self.es().namedtuple:
        # #     self.data = self.data._asdict().copy()
        # # elif self.instance(self.getmroinsattr('__ignore_str__')) and convert:
        # #     self.data = str(self.data)
        # # elif self.instance(Semaphore) and convert:
        # #     self.data = dict(locked=self.data.locked(), value=self.data._value)
        # # elif self.instance(GitSymbolicReference) and convert:
        # #     self.data = dict(repo=self.data.repo, path=self.data.path)
        # # elif self.instance(Remote) and convert:
        # #     self.data = dict(repo=self.data.repo, name=self.data.name)
        # # elif self.instance(Environs) and convert:
        # #     self.data = self.data.dump()
        # # elif self.instance(Logger) and convert:
        # #     self.data = dict(name=self.data.name, level=self.data.level)
        # # if self.enumenumcls:
        # #     self.data = {key: value._value_ for key, value in self.getcls.__members__.items()}
        # # elif self.chainmap and convert:
        # #     self.data.rv = ChainRV.FIRST
        # #     self.data = dict(self.data).copy()
        # if any([self.dataclass, self.dataclass_instance, self.dict_cls, self.dict_instance, self.slots_cls,
        #           self.slots_instance]):
        #     self.data = self.defaults if defaults else self.defaults | self.vars
        # # elif self.mutablemapping and convert:
        # #     self.data = dict(self.data).copy()
        # if self.mlst:
        #     rv = dict() if (mm := self.mutablemapping) else list()
        #     for key in self.data:
        #         value = get(self.data, key) if mm else key
        #         if value:
        #             if (inc := self.include(key, self.data if mm else None)) is None:
        #                 continue
        #             else:
        #                 value = inc[1]
        #                 if self.depth is None or count < self.depth:
        #                     value = self.new(value).asdict(count=count + 1, defaults=defaults)
        #         rv.update({key: value}) if mm else rv.append(value)
        #     return rv if mm else type(self.data)(rv)
        # if (inc := self.include(self.data)) is not None:
        #     if self.getsetdescriptor() or self.iscoro or isinstance(inc[1], \
        #     (*self.getmroinsattr('__ignore_copy__'),)) \
        #             or (self.depth is not None and self.depth > 1):
        #         return inc[1]
        #     try:
        #         return deepcopy(inc[1])
        #     except TypeError as exception:
        #         if "cannot pickle '_thread.lock' object" == str(exception):
        #             return inc[1]
        return self.data

    def attr_value(self, name, default=None): return getattr(self.data, name, default)

    @property
    def attrs(self) -> list:
        """
        Attrs including properties if not self.ignore.

        Excludes:
            __ignore_attr__
            __ignore_kwarg__ if not self.ignore.

        Returns:
            list:
        """
        return sorted([attr for attr in {*self.attrs_cls, *self.cls.memberdescriptor,
                                         *(vars(self.data) if self.es().datatype or self.es().datatype_sub else []),
                                         *(self.cls.prop if not self.ignore else [])}
                       if self._include_attr(attr, self.cls.callable) and attr not in self.cls.setter])

    @property
    def attrs_cls(self):
        attrs = {item for item in self.cls.dir if
                 self._include_attr(item) and item in self.cls.data_attrs and item}
        if self.cls.es.datatype_sub:
            _ = {attrs.add(item.name) for item in datafields(self.data) if self._include_attr(item.name)}
        return sorted(list(attrs))

    @property
    def cls(self): return Cls(data=self.data, ignore=self.ignore, key=self.key)

    @property
    def coro(self): return [i.name for i in self.cls.classified if Es(i.object).coro] + self.coros_property

    @property
    def coro_pproperty(self): return [i.name for i in self.cls.classified if Es(i.object).pproperty and
                                      Es(object.__getattribute__(self.data, i.name)).coro]

    @property
    def coro_prop(self): return [i.name for i in self.cls.classified if Es(i.object).prop and
                                 Es(object.__getattribute__(self.data, i.name)).coro]

    data = property(
        lambda self: object.__getattribute__(self, '_data'),
        lambda self, value: object.__setattr__(self, '_data', value),
        lambda self: object.__setattr__(self, '_data', None)
    )

    @property
    def defaults(self):
        """Class defaults."""

        def is_missing(default: str) -> bool:
            return isinstance(default, MISSING_TYPE)

        rv = dict()
        rv_data = dict()
        attrs = self.attrs_cls
        if self.cls.es.datatype_sub:
            rv_data = {f.name: f.default if is_missing(
                f.default) and is_missing(f.default_factory) else f.default if is_missing(
                f.default_factory) else f.default_factory() for f in datafields(self.data) if f.name in attrs}
        if self.cls.es.namedtype_sub:
            rv = self.cls.data._field_defaults
        elif self.cls.es.dicttype_sub or self.cls.es.slotstype_sub:
            rv = {key: inc[1] for key in attrs if (inc := self.include(key, self.data)) is not None}
        return rv | rv_data

    depth = property(
        lambda self: object.__getattribute__(self, '_depth'),
        lambda self, value: object.__setattr__(self, '_depth', value),
        lambda self: object.__setattr__(self, '_data', None)
    )

    @property
    def dir(self):
        return set(self.cls.dir + [i for i in dir(self.data) if self.key.include(i)])

    def es(self, data=None): return Es(data or self.data)

    def has_attr(self, name): return self.cls.has_attr(name=name) or hasattr(self.data, name)

    def has_method(self, name): return self.cls.has_method(name=name) or has_method(self.data, name)

    @property
    def has_reduce(self): return self.cls.has_reduce or has_reduce(self.data)

    @property
    def hash(self):
        return hash(tuple(map(lambda x: getset(self.data, x), Mro.hash_exclude.slotsinclude(self.data))))

    ignore = property(
        lambda self: object.__getattribute__(self, '_ignore'),
        lambda self, value: object.__setattr__(self, '_ignore', value),
        lambda self: object.__setattr__(self, '_ignore', False)
    )

    @property
    def ignore_attr(self): return

    def _include_attr(self, name, exclude=tuple()):
        ignore = {*Mro.ignore_attr.val(self.data), *(Mro.ignore_kwarg.val(self.data) if self.ignore else set()),
                  *exclude, *self.cls.initvar}
        return not any([not self.key.include(name), name in ignore, f'_{self.cls.name}' in name])

    def _include_exclude(self, data, key=True):
        import typing
        i = info(data)
        call = (Environs,)
        return any([i.module == typing, i.module == _abc, i.es().moduletype,
                    False if i.cls.data in call else i.es().callable, i.es().type,
                    self.key.include(data) if key else False])

    def include(self, key=None, data=None):
        es = Es(data)
        if (not es.mm and Cls(data).is_memberdescriptor(key) and key not in Mro.ignore_attr.val(data)) \
                or not self._include_exclude(key):
            if not es.none:
                if (value := get(self.data, key)) and self._include_exclude(value, key=False):
                    return None
                return key, value
            return key, key
        return None

    @property
    def initvarsdict(self):
        return getattr(self.data, '__init_vars__', dict())

    def is_attr(self, name): return self.cls.is_attr(name) or name in self().dir

    def is_coro(self, name): return name in self().coros

    def is_coro_pproperty(self, name): return name in self().coros_pproperty

    def is_coro_property(self, name): return name in self().coros_property

    key = property(
        lambda self: object.__getattribute__(self, '_key'),
        lambda self, value: object.__setattr__(self, '_key', value),
        lambda self: object.__setattr__(self, '_key', Attr.PRIVATE)
    )

    @property
    def keys(self):
        """
        Keys from kwargs to init class (not InitVars), exclude __ignore_kwarg__ and properties.

        Returns:
            list:
        """
        return sorted(list(self.kwargs.keys()))

    @property
    def kwargs(self):
        """
        Kwargs to init class with python objects no recursive, exclude __ignore_kwarg__ and properties.

        Includes InitVars.

        Example: Mongo binary.

        Returns:
            dict:
        """
        ignore = self.ignore
        self.ignore = True
        rv = {key: get(self.data, key) for key in self.attrs_cls} | \
             {key: value for key, value in self.initvarsdict.items()
              if key not in {*Mro.ignore_attr.val(self.data), *Mro.ignore_kwarg.val(self.data)}}
        self.ignore = ignore
        return rv

    @property
    def kwargs_dict(self) -> dict:
        """
        Kwargs recursive to init class with python objects as dict, asdict excluding __ignore_kwarg__ and properties.

        Example: Mongo asdict.

        Returns:
            dict:
        """
        ignore = self.ignore
        self.ignore = True
        rv = self.asdict()
        self.ignore = ignore
        return rv

    @property
    def module(self): return getmodule(self.data)

    @property
    def public(self):
        self.key = Attr.PUBLIC
        return self.asdict()

    @property
    def repr(self):
        attrs = Mro.repr_exclude.slotsinclude(self.data)
        attrs.update(self.cls.pproperty if Mro.repr_pproperty.first(self.data) else list())
        r = [f"{s}: {getset(self.data, s)}" for s in sorted(attrs) if s and not self.is_coro(s)]
        new = f',{NEWLINE if Mro.repr_newline.first(self.data) else " "}'
        return f'{self.cls.name}({new.join(r)})'

    @property
    def to_json(self, regenerate=True, indent=4, keys=True, max_depth=-1):
        return jsonpickle.encode(self.data, unpicklable=regenerate, indent=indent, keys=keys, max_depth=max_depth)

    def to_obj(self, keys=True): return jsonpickle.decode(self.data, keys=keys)

    @property
    def values(self):
        """
        Init python objects kwargs values no properties and not __ignore_kwarg__.

        Returns:
            list:
        """
        return list(self.kwargs.values())

    @property
    def values_dict(self):
        """
        Init python objects as dict kwargs values no properties and not __ignore_kwarg__.

        Returns:
            list:
        """
        return list(self.kwargs_dict.values())

    @property
    def vars(self):
        attrs = self.attrs
        return {key: inc[1] for key in attrs if (inc := self.include(key, self.data)) is not None}
