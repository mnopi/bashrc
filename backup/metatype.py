class MetaType(metaclass=ABCMeta):
    """
    Return Value Meta Class.

    Examples:
        >>> pretty_install()
        >>> class Dict: a = 1
        >>> class MetaTest(metaclass=Meta): pass
        >>> Es(MetaTest).meta_sub
        False
        >>> Es(MetaTest).meta
        True
        >>> Es(MetaTest()).meta
        False
        >>> Es(MetaTest).metatype_sub
        True
        >>> Es(MetaTest()).metatype
        True
        >>> Es(Dict).metatype_sub
        False
        >>> Es(Dict()).metatype
        False
        """
    _rv = list()
    infocls = None
    @property
    @abstractmethod
    def rv(self): return self._rv
    @rv.setter
    @abstractmethod
    def rv(self, value): self._rv = value
    @rv.deleter
    @abstractmethod
    def rv(self): self._rv = list()
    __class_getitem__ = classmethod(GenericAlias)

    @classmethod
    def __subclasshook__(cls, C):
        attrs = ('__class_getitem__', '_rv', 'infocls', 'rv', )
        callables = ('__class_getitem__', )
        if cls is MetaType:
            return all(map(partial(hasattr, C), attrs)) \
                   and all(map(partial(hasattr, C.rv), ['fdel', 'fget', 'fset'])) \
                   and all(map(callable, map(partial(getattr, C), callables)))
        return NotImplemented
