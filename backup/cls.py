class Cls:
    """Class Helper Class."""
    __slots__ = ('args', 'asdict', 'attr', 'builtin', 'cache', 'cached_property', 'classified', 'classmethod',
                 'classvar',
                 'coro', 'data',
                 'defaults', 'dir',
                 'dynamicclassattribute',
                 'es', 'factory',
                 'fields', 'ignore', 'initvar', 'key', 'kwargs',
                 'method', 'mro', 'name',
                 'pproperty', 'prop', 'property_any', 'slots', 'staticmethod')
    # __slots__ = ('asyncgen', 'asyncgeneratortype', 'asyncgenfunc', 'attrs', 'awaitable', 'builtinfunctiontype',
    #              'callable', 'classmethod', 'classmethoddescriptortype', 'coroutine',
    #              'coroutinefunc', 'coroutinetype', 'data',
    #              'datafactory', 'datafield',
    #              'deleter', 'defined_class', 'defined_obj', 'defaults', 'dir', 'dynamicclassattribute',
    #              'es', 'functiontype', 'generator', 'generatortype', 'getsetdescriptor',
    #              'ignore', 'initvar', 'key', 'lambdatype',
    #              'mappingproxytype', 'memberdescriptor', 'method', 'methoddescriptor', 'methoddescriptortype',
    #              'methodtype', 'methodwrappertype',
    #              'mro', 'names',
    #              'none', 'routine', 'setter', 'source'
    #              'wrapperdescriptortype', )
    kind_compose = ('annotation', 'data_attrs', 'datafactory_dict', 'datafield_dict', 'slots/memberdescriptor')

    def __init__(self, data, ignore=False, key=Attr.PRIVATE):
        """
        Class Helper init.

        Examples:
            >>> from rc import Cls
            >>> from rc import pretty_install
            >>> from rc import TestData
            >>>
            >>> pretty_install()
            >>>
            >>> test = Cls(TestDataDictSlotMix)
            >>>
            #
            # Data Class Fields
            #
            >>> list(test.datafield)
            [
                'dataclass_classvar',
                'dataclass_default',
                'dataclass_default_factory',
                'dataclass_default_factory_init',
                'dataclass_default_init',
                'dataclass_initvar',
                'dataclass_str'
            ]
            >>> test.datafield  # doctest: +ELLIPSIS
            {
                'dataclass_classvar': Field(name='dataclass_classvar',...),
                'dataclass_default': Field(name='dataclass_default',...),
                'dataclass_default_factory': Field(name='dataclass_default_factory',...),
                'dataclass_default_factory_init': Field(name='dataclass_default_factory_init',...),
                'dataclass_default_init': Field(name='dataclass_default_init',...),
                'dataclass_initvar': Field(name='dataclass_initvar',...),
                'dataclass_str': Field(name='dataclass_str',...)
            }
            >>>
            #
            # Data Class Fields - default_factory - ['dataclass_default_factory', 'dataclass_default_factory_init']
            #
            >>> list(test.datafactory)
            ['dataclass_default_factory', 'dataclass_default_factory_init']
            >>> test.datafactory
            {'dataclass_default_factory': {}, 'dataclass_default_factory_init': {}}
            >>> 'dataclass_default_factory' in TestDataDictSlotMix.__dataclass_fields__
            True
            >>> 'dataclass_default_factory' in dir(TestDataDictSlotMix)
            False
            >>> 'dataclass_default_factory' in vars(TestDataDictSlotMix)
            False
            >>> 'dataclass_default_factory' in vars(TestDataDictSlotMix())
            True
            >>> vars(TestDataDictSlotMix()).get('dataclass_default_factory')
            {}
            >>> getattr(TestDataDictSlotMix(), 'dataclass_default_factory')
            {}
            >>> 'dataclass_default_factory_init' in TestDataDictSlotMix.__dataclass_fields__
            True
            >>> 'dataclass_default_factory_init' in dir(TestDataDictSlotMix)
            False
            >>> 'dataclass_default_factory_init' in vars(TestDataDictSlotMix)
            False
            >>> 'dataclass_default_factory_init' in vars(TestDataDictSlotMix())
            True
            >>> vars(TestDataDictSlotMix()).get('dataclass_default_factory_init')
            {}
            >>> getattr(TestDataDictSlotMix(), 'dataclass_default_factory_init')
            {}
            >>> vars(TestDataDictSlotMix())['dataclass_default_factory_init'] == \
            test.datafactory['dataclass_default_factory_init']
            True
            >>>
            #
            # Data Class Fields - ['dataclass_classvar']
            #
            >>> list(test.classvar)
            ['dataclass_classvar']
            >>> test.classvar  # doctest: +ELLIPSIS
            {
                'dataclass_classvar': Field(name='dataclass_classvar',...)
            }
            >>> 'dataclass_classvar' in TestDataDictSlotMix.__dataclass_fields__
            True
            >>> 'dataclass_classvar' in dir(TestDataDictSlotMix)
            True
            >>> 'dataclass_classvar' in vars(TestDataDictSlotMix)
            False
            >>> 'dataclass_classvar' in vars(TestDataDictSlotMix())
            False
            >>> vars(TestDataDictSlotMix()).get('dataclass_classvar')
            >>> getattr(TestDataDictSlotMix(), 'dataclass_classvar')
            'dataclass_classvar'
            >>>
            #
            # Data Class Fields - ['dataclass_initvar']
            #
            >>> list(test.initvar)
            ['dataclass_initvar']
            >>> test.initvar  # doctest: +ELLIPSIS
            {
                'dataclass_initvar': Field(name='dataclass_initvar',...)
            }
            >>> 'dataclass_initvar' in TestDataDictSlotMix.__dataclass_fields__
            True
            >>> 'dataclass_initvar' in dir(TestDataDictSlotMix)
            True
            >>> 'dataclass_initvar' in vars(TestDataDictSlotMix)
            False
            >>> 'dataclass_initvar' in vars(TestDataDictSlotMix())
            False
            >>> vars(TestDataDictSlotMix()).get('dataclass_initvar')
            >>> getattr(TestDataDictSlotMix(), 'dataclass_initvar')
            'dataclass_initvar'
            >>>
            #
            # Dict - ClassVar - ['subclass_classvar']
            #

x

        Args:
            data: Class to provide information.
            ignore: ignore properties and kwargs :class:`Base.__ignore_kwargs__` (default: False)
            key: keys to include (default: :attr:`rc.Key.PRIVATE`)

        Returns:
            Cls Instance.
        """
        effect(lambda x: self.__setattr__(x, dict()), self.__slots__)
        self.data = data if isinstance(data, type) else type(data)
        self.es = Es(self.data)
        self.ignore = ignore
        self.key = key
        self.classified = dict_sort({i.name: i for i in classify_class_attrs(self.data) if self.key.include(i.name)})
        self.fields = dict(filter(lambda x: self.key.include(x[0]), dict_sort(self.data.__dataclass_fields__).items(

        ))) \
            if self.es.datatype_sub else dict()
        factories = filter(lambda x: Es(x[1]).datafactory, self.fields.items())
        if self.es.datatype_sub:
            for i in sorted(self.fields):
                if self.key.include(i):
                    v = self.fields[i]
                    es = Es(v)
                    self.datafield |= {i: v}
                    if es.classvar:
                        self.classvar |= {i: v}
                    elif es.datafactory:
                        self.factory |= {i: v.default_factory()}
                    elif es.initvar:
                        self.initvar |= {i: v}

    def __call__(self, ignore=False, key=Attr.ALL):
        """
        Updates instance with ignore adn key (default: Attr.ALL)

        Examples:
            >>> from rc import Cls
            >>> from rc import pretty_install
            >>>
            >>> pretty_install()
            >>>
            >>> cls = Cls(dict())
            >>> cls.is_callable(Mro.getitem.value)
            False
            >>> cls().callable
            >>> cls().is_callable(Mro.getitem.value)
            True

        Args:
            ignore: ignore properties and kwargs :class:`Base.__ignore_kwargs__` (default: False)            key:
            key: keys to include (default: :attr:`rc.Key.PRIVATE`)

        Returns:
            Updated Cls Instance.
        """
        return self.__init__(data=self.data, ignore=ignore, key=key)

    @functools.cache
    def _annotations(self, stack=2):
        return annotations(self.data, stack=stack)

    @property
    def _asyncgen(self):
        return self.kind[self.asyncgen.__name__]

    @property
    def _asyncgenfunc(self):
        return self.kind[self.asyncgenfunc.__name__]

    @functools.cache
    def _attr_value(self, name, default=None): return getattr(self.data, name, default)

    @property
    def _awaitable(self):
        return self.kind[self.awaitable.__name__]

    @property
    def _builtinfunctiontype(self):
        return self.kind[self.builtinfunctiontype.__name__]

    @property
    def _by_kind(self): return bucket(self.classified, key=lambda x: x.kind if self.key.include(x.name) else 'e')

    @property
    def _by_name(self): return {i.name: i for i in self.classified if self.key.include(i.name)}

    @property
    def _cache(self):
        return self.kind[self.cache.__name__]
        # return sorted([key for key, value in self.data.__dict__.items() if Es(value).cache and self.key.include(key)])

    @property
    def _cached_property(self):
        return self.kind[self.cached_property.__name__]
        # return sorted([key for key, value in self.data.__dict__.items()
        # if Es(value).cached_property and self.key.include(key)])

    @property
    def _callable(self):
        """
        Class Callables filtered based on startswith.

        Examples:
            >>> from rc import Cls
            >>> from rc import pretty_install
            >>> from rc import TestBase
            >>>
            >>> pretty_install()
            >>> test = TestBase()
            >>> test.info.cls.callable
            ['_info', 'clsmethod', 'get', 'method_async', 'static']
            >>> test.info.cls().callable  # doctest: +ELLIPSIS
            [
                '__delattr__',
                '__dir__',
                ...,
                '__str__',
                '__subclasshook__',
                '_info',
                'clsmethod',
                'get',
                'method_async',
                'static'
            ]

        Returns:
            List of Class Callables names filtered based on startswith.
        """
        return self.kind[self.callable.__name__]
        # return sorted(self.classmethods + self.methods + self.staticmethods)

    @property
    def _classmethod(self):
        """
        Class Methods filtered based on startswith.

        Examples:
            >>> from rc import Cls
            >>> from rc import pretty_install
            >>> from rc import TestBase
            >>>
            >>> pretty_install()
            >>> test = TestBase()
            >>> test.info.cls.classmethod
            ['clsmethod']
            >>> test.clsmethod.__name__ in test.info.cls.classmethod
            True
            >>> test.info.cls().classmethod
            ['__init_subclass__', '__subclasshook__', 'clsmethod']
            >>> Mro.init_subclass.value in test.info.cls().classmethod
            True

        Returns:
            List of Class Methods names filtered based on startswith.
        """
        return self.kind[self.classmethod.__name__]
        # return list(map(Name.name_.getter, self.by_kind['class method']))

    # @property
    # def classvar(self):
    #     return self.kind[self.classvar.__name__]
    #     # return [key for key, value in self.annotations(stack=3).items() if value.classvar]

    @property
    def _coro(self):
        return self.kind[self.coro.__name__]

    @property
    def _coroutine(self):
        return self.kind[self.coroutine.__name__]

    @property
    def _coroutinefunc(self):
        return self.kind[self.coroutinefunc.__name__]

    @property
    def _data_attrs(self):
        """
        Class Data Attributes including Dataclasses Fields with default factory are not in classified,
        filtered based on startswith.

        Examples:
            >>> from rc import Cls
            >>> from rc import pretty_install
            >>> from rc import TestData
            >>>
            >>> pretty_install()
            >>> testdata = Cls(TestData)
            >>> testdata.data_attrs
            ['a', 'b', 'c', 'd', 'e', 'f', 'g']
            >>> testdata.dir
            ['a', 'b', 'c', 'd', 'e', 'f', 'g']
            >>> testdata.dir == testdata.data_attrs == testdata.datafield
            True

        Returns:
            List of attribute names.
        """
        return sorted({*map(Name.name_.getter, self.by_kind['data']), *self.datafield_dict.keys()})

    @property
    def _defaults(self):
        """
        Data Class Fields List.

        Examples:
            >>> from rc import Cls
            >>> from rc import pretty_install
            >>> from rc import TestData
            >>>
            >>> pretty_install()
            >>> testdata = Cls(TestData)
            >>> testdata.defaults
            ['a', 'b', 'c', 'd', 'e', 'f', 'g']

        Returns:
            Data Class Field Attribute Names List.
        """
        return self

    @property
    def _deleter(self):
        return self.kind[self.deleter.__name__]
        # return sorted([i.name for i in self.classified if Es(i.object).deleter])

    @property
    def _dir(self):
        # noinspection PyUnresolvedReferences
        """
        Class Data Attributes including Dataclasses Fields with default factory are not in dir,
            filtered based on startswith.

        Examples:
            >>> from rc import Cls
            >>> from rc import pretty_install
            >>> from rc import TestData
            >>>
            >>> pretty_install()
            >>> testdata = Cls(TestData)
            >>> testdata.data_attrs
            ['a', 'b', 'c', 'd', 'e', 'f', 'g']
            >>> testdata.dir
            ['a', 'b', 'c', 'd', 'e', 'f', 'g']
            >>> testdata.data_attrs == sorted([i for i in dir(TestData) \
            if testdata.key.include(i)] + testdata.datafactories)
            True

        Returns:
            List of attribute names.
        """
        return sorted([i for i in {*dir(self.data), *self.datafield_dict.keys()} if self.key.include(i)])

    @property
    def _generator(self):
        return self.kind[self.generator.__name__]

    @property
    def _getsetdescriptor(self):
        return self.kind[self.getsetdescriptor.__name__]

    @functools.cache
    def has_attr(self, name): return hasattr(self.data, name)

    @functools.cache
    def has_method(self, name): return has_method(self.data, name)

    @property
    def has_reduce(self): return has_reduce(self.data)

    @functools.cache
    def is_attr(self, name):
        return name in self.dir

    def is_callable(self, name):
        """
        Is Class Callable filtered based on startswith.

        Examples:
            >>> from rc import pretty_install
            >>> from rc import TestBase
            >>>
            >>> pretty_install()
            >>>
            >>> test = TestBase()
            >>> test.info.cls.is_callable('clsmethod')
            True
            >>> test.info.cls.is_callable(Mro.str.value)
            False
            >>> test.info.cls().is_callable(Mro.str.value)
            True
            >>> test.info.cls.is_callable('prop')
            False

        Returns:
            True if Callable Method name filtered based on startswith.
        """
        return name in self.callable

    @functools.cache
    def is_classmethod(self, name):
        """
        Is Class Method filtered based on startswith.

        Examples:
            >>> from rc import pretty_install
            >>> from rc import TestBase
            >>>
            >>> pretty_install()
            >>>
            >>> test = TestBase()
            >>> test.info.cls.is_classmethod('clsmethod')
            True
            >>> test.info.cls.is_classmethod(Mro.init_subclass.value)
            False
            >>> test.info.cls().is_classmethod(Mro.init_subclass.value)
            True
            >>> test.info.cls.is_classmethod('pprop_async')
            False

        Returns:
            True if Class Method name filtered based on startswith.
        """
        return name in self.classmethod

    @functools.cache
    def is_classvar(self, name):
        return name in self.classvar

    @functools.cache
    def is_data(self, name):
        return name in self.data_attrs

    @functools.cache
    def is_deleter(self, name):
        return name in self.deleter

    @functools.cache
    def is_datafactory(self, name):
        return name in self.datafactory

    @functools.cache
    def is_datafield(self, name):
        return name in self.datafield

    @functools.cache
    def is_initvar(self, name):
        return name in self.initvar

    @functools.cache
    def is_memberdescriptor(self, name):
        return name in self.memberdescriptor

    @functools.cache
    def is_method(self, name):
        return name in self.method

    @functools.cache
    def is_methoddescriptor(self, name):
        return name in self.methoddescriptor

    @functools.cache
    def is_pproperty(self, name):
        return name in self.pproperty

    @functools.cache
    def is_property(self, name):
        return name in self.prop

    @functools.cache
    def is_routine(self, name):
        return name in self.routine

    @functools.cache
    def is_setter(self, name):
        return name in self.setter

    @functools.cache
    def is_staticmethod(self, name):
        """
        Is Static Method filtered based on startswith.

        Examples:
            >>> from rc import pretty_install
            >>> from rc import TestBase
            >>>
            >>> pretty_install()
            >>>
            >>> test = TestBase()
            >>> test.info.cls.is_staticmethod('static')
            True
            >>> test.info.cls.is_staticmethod('pprop_async')
            False

        Returns:
            True if Static Methods name filtered based on startswith.
        """
        return name in self.staticmethod

    def kind(self, value):
        # TODO: coger el code de la clase de inspect y poner el source aqui y ver el code para el async!
        #  o ponerlo en la de Name!!!
        _fields = dict_sort(value.__dataclass_fields__) if self.es.datatype_sub else dict()

        _dict = sorted([i for i in {*dir(value), *_fields.keys()} if self.key.include(i)])
        self._kind = {kind: {name: obj for name, obj in _dict.items() if getattr(Es(obj), kind)
                             and self.key.include(name)} for kind in self.kind_attr}

    @property
    def _mappingproxytype(self):
        return self.kind[self.mappingproxytype.__name__]

    @property
    def _memberdescriptor(self):
        return self.kind[self.memberdescriptor.__name__]
        # return [i.name for i in self.classified if Es(i.object).memberdescriptor]

    @property
    def _method(self):
        return self.kind[self.method.__name__]
        # return list(map(Name.name_.getter, self.by_kind['method']))

    @property
    def _methoddescriptor(self):
        """
        Includes classmethod, staticmethod and methods but not functions defined (i.e: def info(self))

        Returns:
            Method descriptors.
        """
        return self.kind[self.methoddescriptor.__name__]
        # return [i.name for i in self.classified if Es(i.object).methoddescriptor]

    @property
    def _methodwrappertype(self):
        return self.kind[self.methodwrappertype.__name__]

    @property
    def _none(self):
        return self.kind[self.none.__name__]

    @property
    def _pproperty(self):
        return self.kind[self.pproperty.__name__]
        # return [i.name for i in self.classified if Es(i.object).pproperty]

    @property
    def _prop(self):
        return self.kind[self.prop.__name__]
        # return list(map(Name.name_.getter, self.by_kind['property']))

    @property
    def _property_any(self):
        return self.kind[self.property_any.__name__]
        # return list(map(Name.name_.getter, self.by_kind['property']))

    @staticmethod
    def propnew(name, default=None):
        return property(
            lambda self:
            self.__getattribute__(f'_{name}', default=default(self) if isinstance(default, partial) else default),
            lambda self, value: self.__setattr__(f'_{name}', value),
            lambda self: self.__delattr__(f'_{name}')
        )

    @property
    def _qualname(self): return Name._qualname0.get(self.data, default=str())

    @property
    def _routine(self):
        return self.kind[self.routine.__name__]
        # return [i.name for i in self.classified if Es(i.object).routine]

    @property
    def _setter(self):
        return self.kind[self.setter.__name__]
        # return [i.name for i in self.classified if Es(i.object).setter]

    @property
    def _staticmethod(self):
        """
        Static Methods filtered based on startswith.

        Examples:
            >>> from rc import Cls
            >>> from rc import pretty_install
            >>> from rc import TestBase
            >>>
            >>> pretty_install()
            >>> test = TestBase()
            >>> test.info.cls.staticmethod
            ['static']
            >>> test.static.__name__ in test.info.cls.staticmethod
            True

        Returns:
            List of Static Methods names filtered based on startswith.
        """
        return self.kind[self.staticmethod.__name__]
        # return list(map(Name.name_.getter, self.by_kind['static method']))
