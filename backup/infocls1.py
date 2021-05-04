class InfoCls1(InfoClsBase):
    # TODO: Examples: InfoCls1
    """
    Object Class or Class Info.

    Examples:
        >>> from rc import InfoCls1
        >>> from rc import pretty_install
        >>>
        >>> pretty_install()
        >>>
    """
    __slots__ = ()

    def __new__(cls, data):
        # TODO: Examples: __new__
        Attribute = namedtuple('Attribute', 'annotation default defining es field kind name object qual')

        es = Es(data)
        fields = data.__dataclass_fields__ if es.datatype_sub or es.datatype else {}
        data = data if es.type else data.__class__
        annotation = annotations(data, stack=2) if es.annotationstype_sub else dict()
        attribute = dict()
        for item in classify_class_attrs(data):
            e = Es(item.object)
            attribute.append(Attribute(
                    annotation=annotation.get(item.name),
                    default=item.object if item.name not in fields or fields[item.name].init else NotImplemented,
                    defining=item.defining_class, es=e, field=Es(fields.pop(item.name, None)),
                    kind=Kind[item.kind.split(' ')[0].upper()], name=item.name,
                    object=item.object, qual=Name._qualname0.get(e._func, default=item.name))
            )
        for k, value in fields.items():
            defining_cls = data
            for C in data.__mro__:
                if not Es(C).datatype_sub or (Es(C).datatype_sub and k not in C.__dataclass_fields__):
                    break
                defining_cls = C
            f = Es(value)
            obj = value.default_factory() if f.datafactory else getattr(defining_cls, k)
            attribute.append(Attribute(annotation=annotation.get(k), default=obj if value.init else NotImplemented,
                                       defining=defining_cls, es=Es(obj), field=f, kind=Kind.DATA, name=k, object=obj,
                                       qual=k))
        # TODO: super InfoCls1()
        # noinspection PySuperArguments
        return super(InfoCls1, cls).__new__(
            , attribute, es,
            )

    def __new1__(cls, data):
        # TODO: Examples: __new__
        """
        New InfoCls1 Instance for Object Class or Class.

        Examples:
            >>> from rc import InfoCls1
            >>> from rc import pretty_install
            >>> from rc import TestDataDictSlotMix
            >>>
            >>> pretty_install()
            >>>
            >>> test = InfoCls1(TestDataDictSlotMix)
            >>> attr = test.attribute['_TestData__dataclass_default_factory']
            >>> attr.default, attr.defining, attr.field.datafactory, attr.object, \
            attr.field.data.type == attr.annotation.hint, attr.object == attr.annotation.default  # doctest: +ELLIPSIS
            (NotImplemented, <class '....TestData'>, True, {}, True, True)

        Args:
            data: object

        Returns:
            New InfoCls1 Instance for Object Class or Class.
        """
        es = Es(data)
        fields = data.__dataclass_fields__ if es.datatype_sub or es.datatype else {}
        data = data if es.type else data.__class__
        annotation = annotations(data, stack=2) if es.annotationstype_sub else dict()
        attribute = dict()
        for item in classify_class_attrs(data):
            e = Es(item.object)
            attribute |= {
                item.name: Attribute(
                    annotation=annotation.get(item.name),
                    default=item.object if item.name not in fields or fields[item.name].init else NotImplemented,
                    defining=item.defining_class, es=e, field=Es(fields.pop(item.name, None)),
                    kind=Kind[item.kind.split(' ')[0].upper()], name=item.name,
                    object=item.object, qual=Name._qualname0.get(e._func, default=item.name))
            }
        for k, value in fields.items():
            defining_cls = data
            for C in data.__mro__:
                if not Es(C).datatype_sub or (Es(C).datatype_sub and k not in C.__dataclass_fields__):
                    break
                defining_cls = C
            f = Es(value)
            obj = value.default_factory() if f.datafactory else getattr(defining_cls, k)
            attribute |= {k: Attribute(annotation=annotation.get(k), default=obj if value.init else NotImplemented,
                                       defining=defining_cls, es=Es(obj), field=f, kind=Kind.DATA, name=k, object=obj,
                                       qual=k)}
        # TODO: super InfoCls1()
        # noinspection PySuperArguments
        return super(InfoCls1, cls).__new__(
            cls, annotation, dict_sort(attribute), anyin(data.__mro__, BUILTINS_CLASSES), data, es,
            importable_name(data), data.__module__, data.__mro__, data.__name__, data.__qualname__, data.__mro__[1])

    # TODO: Examples: __repr__, __str__

    def data(self, key=Attr.PRIVATE):
        # TODO: Examples: data
        """
        Class Data Attributes.

        Examples:
            >>> from rc import InfoCls1
            >>> from rc import pretty_install
            >>>
            >>> pretty_install()
            >>>

        Args:
            key: attr startswith include.

        Returns:
            Class Data Attributes filtered with attr startswith.
        """
        return self.member(key=key)

    def defaults(self, key=Attr.PRIVATE):
        # TODO: Examples: defaults
        # TODO: Examples: Aqui el __init__ si es slots!!! que le den
        # TODO: namedtuple defaults
        """
        Class Defaults.

        Examples:
            >>> from rc import InfoCls1
            >>> from rc import pretty_install
            >>>
            >>> pretty_install()
            >>>

        Args:
            key: attr startswith include.

        Returns:
            Class Defaults Dict filtered with attr startswith.
        """
        return {i.name: i.object for i in self.data(key=key).values() if i.object is not NotImplemented}

    @singledispatchmethod
    def member(self, find: Kind = Kind.DATA, key=Attr.PRIVATE):
        # TODO: Examples: member
        """
        Filter Class Members based on :class:`rc.Kind`.

        Examples:
            >>> from rc import InfoCls1
            >>> from rc import pretty_install
            >>>
            >>> pretty_install()
            >>>

        Args:
            find: :class:`rc.Kind` to filter.
            key: attr startswith include.

        Returns:
            Class Members filtered based on :class:`rc.Kind` and attr startswith.
        """
        return {i.name: i for i in self.attribute.values() if key.include(i.name) and i.kind is find}

    @member.register
    def member_es(self, find: attrgetter = Es.memberdescriptor, key=Attr.PRIVATE):
        # TODO: Examples: member_es
        """
        Filter Class Members based on return value of :class:`rc.Es` property.

        Examples:
            >>> from rc import InfoCls1
            >>> from rc import pretty_install
            >>>
            >>> pretty_install()
            >>>

        Args:
            find: :class:`operator.attrgetter` with :class:`rc.ES` property name to filter.
            key: attr startswith include.

        Returns:
            Class Members filtered based on return value :class:`rc.Es` property and attr startswith.
        """
        return {i.name: i for i in self.attribute.values() if key.include(i.name) and find(i.es)}
