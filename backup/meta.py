
class Meta(type):
    """
    Return Value Meta Class.

    Examples:
        >>> pretty_install()
        >>> class MetaTest(metaclass=Meta): pass
        >>> Es(MetaTest).meta_sub
        False
        >>> Es(MetaTest).meta
        True
        >>> Es(MetaTest()).meta
        False

    """
    def __new__(mcs, *args, **kwargs):
        """
        RV Meta Class Instance.

        Examples:
            >>> pretty_install()
            >>> class MetaTest(metaclass=Meta): pass
            >>> Es(MetaTest).meta_sub
            False
            >>> Es(MetaTest).meta
            True
            >>> Es(MetaTest()).meta
            False

        Args:
            *args:
                name: str
                bases: tuple[type, ...]
                namespace: dict[str, Any])
            **kwargs:

        Returns:
            RV Meta Class Instance.
        """
        c = super().__new__(mcs, *args, **kwargs)
        c._rv = list()
        # c.infocls = InfoCls()
        c.infocls = None
        c.rv = property(lambda self: type(self)(self._rv) if Es(self._rv).sequence else self._rv, lambda self, value: self._rv.append(value),
                                lambda self: self.__setattr__('_rv', list()))
        c.__class_getitem__ = classmethod(GenericAlias)
        return c
