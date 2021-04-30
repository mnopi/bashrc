class BaseMeta(type):
    # TODO: Examples: BaseMeta
    """
    Meta Base Helper Class.

    Examples:
        >>> from rc import BaseMeta
        >>> from rc import pretty_install
        >>>
        >>> pretty_install()
        >>>
    """
    infocls = None

    def __new__(mcs, *args, **kwargs):
        # TODO: Examples: __new__
        """
        New BaseMeta Class Instance.

        Examples:
            >>> from rc import InfoCls1
            >>> from rc import pretty_install
            >>>
            >>> pretty_install()
            >>>

        Args:
            *args:
                name: str
                bases: tuple[type, ...]
                namespace: dict[str, Any])
            **kwargs:

        Returns:
            New BaseMeta Class Instance.
        """
        c = super().__new__(mcs, *args, **kwargs)
        c.__class_getitem__ = classmethod(lambda cls, item: cls.infocls.attribute.get(item, NotImplemented))
        # TODO: arg InfoCls1()
        # c.infocls = InfoCls(c)
        return c

    def __class_getitem__(mcs, name):
        # TODO: Examples: __class_getitem__
        """
        Class Attribute Information :class:`rc.Attribute`.

        Examples:
            >>> from rc import InfoCls1
            >>> from rc import pretty_install
            >>>
            >>> pretty_install()
            >>>

        Args:
            name: attribute name.

        Returns:
            Class Attribute Information :class:`rc.Attribute`.
        """
        return mcs.infocls.attribute.get(name, NotImplemented)
