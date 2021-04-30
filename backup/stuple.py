
class stuple(tuple, metaclass=Meta):
    """
    Tuple Seq Sequence Helper Class.

    Examples:
        >>> from rc import pretty_install
        >>> from rc import stuple
        >>>
        >>> pretty_install()
        >>>
        >>> s = stuple(['bool', 'dict', 'int'])
        >>> s
        stuple('bool', 'dict', 'int')
        >>> repr(s)
        "stuple('bool', 'dict', 'int')"
    """
    def __new__(cls, iterable=()): return tuple.__new__(stuple, iterable)
    def __repr__(self): return f'{self.__class__.__name__}{super().__repr__()}'


collections.abc.Sequence.register(stuple)

