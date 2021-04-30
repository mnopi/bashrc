
class slist(list, metaclass=Meta):
    """
    List Seq Sequence Helper Class.

    Examples:
        >>> from rc import pretty_install
        >>> from rc import slist
        >>>
        >>> pretty_install()
        >>>
        >>> s = slist(['bool', 'dict', 'int'])
        >>> s
        slist['bool', 'dict', 'int']
        >>> repr(s)
        "slist['bool', 'dict', 'int']"
    """
    def __new__(cls, *args, **kwargs): return list.__new__(cls)
    def __init__(self, seq=()): list.__init__(self, seq)
    def __repr__(self): return f'{self.__class__.__name__}{super().__repr__()}'


collections.abc.Sequence.register(slist)

