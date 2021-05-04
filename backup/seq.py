
class Seq(Sequence):
    """Sequence Helper Class."""
    __builtin__ = object
    __slots__ = tuple()

    @abstractmethod
    def __new__(cls, *args, **kwargs):
        pass

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __getitem__(self, index):
        return self.__builtin__.__getitem__(self, index)

    def all_allin(self, *args, **kwargs):
        """
        ROWS: Get ALL container items/rows found if ALL args and kwargs are found IN item/row.

        Recursivily for MutableMapping.

        Examples:
            >>> from inspect import classify_class_attrs
            >>> from rc import BUILTINS_CLASSES
            >>> from rc import pretty_install
            >>> from rc import Seq
            >>> from rc import slist
            >>> from rc import stuple
            >>>
            >>> pretty_install()
            >>>
            #
            # args
            #
            >>> s = slist([dict(a=1), ['b', 'd'], None, dict(a=dict(b=dict(c=1)), b=2), ])
            >>> s.all_allin('b', 'c')
            {'a': {'b': {'c': 1}}, 'b': 2}
            >>> s.all_allin('a', 'd')
            slist[]
            >>>
            >>> s = stuple([dict(e=1), ['b', 'd'], None, dict(a=dict(b=dict(c=1)), b=2), *classify_class_attrs(stuple)])
            >>> s.all_allin(kind='method', defining_class=stuple)  # doctest: +ELLIPSIS
            stuple(Attribute(name='__init__', kind='method', defining_class=<class '....stuple'>, \
object=<function stuple.__init__ at 0x...>), Attribute(name='__repr__', kind='method', \
defining_class=<class '....stuple'>, object=<function stuple.__repr__ at 0x...>))

        Args:
            *args: to use __contains__ in container value/column.
            **kwargs: to use __getattribute__ or __getitem__ in container
            value/column.

        Returns:
            ROWS: ALL container items/rows found if ALL args and kwargs are IN item.
        """
        rv = slist()
        for item in self:
            yes = set()
            es = Es(item)
            if es.container:
                if es.mm:
                    for arg in args:
                        yes.add(True) if nested_lookup(arg, item) else yes.add(False)
                    for key, value in kwargs.items():
                        yes.add(True) if in_dict(item, {key: value}) else yes.add(False)
                    if yes == {True}:
                        rv.append(item)
                else:
                    for arg in args:
                        yes.add(True) if arg in item else yes.add(False)
                    for key, value in kwargs.items():
                        yes.add(True) if value == noexception(Exception, object.__getattribute__, item, key,
                                                              default_=NotImplemented) else yes.add(False)
                    if yes == {True}:
                        rv.append(item)
        rv = rv if Es(self.__class__).slist else self.__class__(rv)
        return rv[0] if len(rv) == 1 else rv

    def all_anyin(self, *args, **kwargs):
        """
        ROWS: Get ALL container items/rows found if ANY arg or kwarg are found IN item/row.

        Recursivily for MutableMapping.

        Examples:
            >>> from inspect import classify_class_attrs
            >>> from rc import BUILTINS_CLASSES
            >>> from rc import pretty_install
            >>> from rc import Seq
            >>> from rc import slist
            >>> from rc import stuple
            >>>
            >>> pretty_install()
            >>>
            #
            # args
            #
            >>> s = slist([dict(a=1), ['b', 'd'], None, dict(a=dict(b=dict(c=1)), b=2), ])
            >>> s.all_anyin('b', c=1)
            slist[['b', 'd'], {'a': {'b': {'c': 1}}, 'b': 2}, {'a': {'b': {'c': 1}}, 'b': 2}]
            >>> s.all_anyin('a', 'd')
            slist[{'a': 1}, ['b', 'd'], {'a': {'b': {'c': 1}}, 'b': 2}]
            >>>
            >>> s = stuple([dict(e=1), ['b', 'd'], None, dict(a=dict(b=dict(c=1)), b=2), *classify_class_attrs(slist)])
            >>> s.all_anyin('d', e=1)
            stuple({'e': 1}, ['b', 'd'])
            >>> s.all_anyin('j', e=3)
            stuple()
            >>> s.all_anyin('b', kind='data')  # doctest: +ELLIPSIS
            stuple(['b', 'd'], {'a': {'b': {'c': 1}}, 'b': 2}, Attribute(name='__abstractmethods__', kind='data', ...)

        Args:
            *args: to use __contains__ in container value/column.
            **kwargs: to use __getattribute__ or __getitem__ in container value/column.

        Returns:
            ROWS: ALL container items/rows found if ANY arg or kwarg are IN item/row.
        """
        rv = slist()
        for item in self:
            es = Es(item)
            if es.container:
                if es.mm:
                    for arg in args:
                        if nested_lookup(arg, item):
                            rv.append(item)
                    for key, value in kwargs.items():
                        if in_dict(item, {key: value}):
                            rv.append(item)
                else:
                    for arg in args:
                        if arg in item:
                            rv.append(item)
                    for key, value in kwargs.items():
                        if value == noexception(Exception, object.__getattribute__, item, key,
                                                default_=NotImplemented):
                            rv.append(item)
        rv = rv if Es(self.__class__).slist else self.__class__(rv)
        return rv[0] if len(rv) == 1 else rv

    def first_allin(self, *args, **kwargs):
        """
        ROW: Get FIRST container item/row found if ALL args and kwargs are IN item/row.

        Recursivily for MutableMapping.

        Examples:
            >>> from inspect import classify_class_attrs
            >>> from rc import BUILTINS_CLASSES
            >>> from rc import pretty_install
            >>> from rc import Seq
            >>> from rc import slist
            >>> from rc import stuple
            >>>
            >>> pretty_install()
            >>>
            #
            # args
            #
            >>> s = slist([dict(a=1), ['b', 'd'], None, dict(a=dict(b=dict(c=1)), b=2), ])
            >>> s.first_allin('b', c=1)
            {'a': {'b': {'c': 1}}, 'b': 2}
            >>> s.first_allin('b', 'd')
            ['b', 'd']
            >>> s = stuple([dict(a=1), ['b', 'd'], None, dict(a=dict(b=dict(c=1)), b=2), ])
            >>> s.first_allin('a', 'b')
            {'a': {'b': {'c': 1}}, 'b': 2}
            >>> s.first_allin('a', 'e') is None
            True
            >>>
            #
            # Kwargs - __getattribute__
            #
            >>> classified = slist(classify_class_attrs(slist))
            >>> d_getattr = classified.first_allin(kind='method', name='first_allin')
            >>> d_getattr  # doctest: +ELLIPSIS
            Attribute(name='first_allin', kind='method', defining_class=<class '....Seq'>, \
object=<function Seq.first_allin at 0x...>)
            >>>
            #
            # Kwargs - __getitem__
            #
            >>> # noinspection PyUnresolvedReferences
            >>> classified = stuple([i._asdict() for i in classify_class_attrs(slist)])
            >>> d_getitem = classified.first_allin(kind='method', name='first_allin')
            >>> d_getitem  # doctest: +ELLIPSIS
            {
                'name': 'first_allin',
                'kind': 'method',
                'defining_class': <class '....Seq'>,
                'object': <function Seq.first_allin at 0x...>
            }
            >>> classified.first_allin(name='test', kind='data') is None
            True
            >>>
            >>> d_getattr._asdict() == d_getitem
            True

        Args:
            *args: to use __contains__ in container value/column.
            **kwargs: to use __getattribute__ or __getitem__ in container value/column.

        Returns:
           ROW: FIRST container item/row found if ALL args and kwargs are IN item/row.
        """
        for item in self:
            yes = set()
            es = Es(item)
            if es.container:
                if es.mm:
                    for arg in args:
                        yes.add(True) if nested_lookup(arg, item) else yes.add(False)
                    for key, value in kwargs.items():
                        yes.add(True) if in_dict(item, {key: value}) else yes.add(False)
                    if yes == {True}:
                        return item
                else:
                    for arg in args:
                        yes.add(True) if arg in item else yes.add(False)
                    for key, value in kwargs.items():
                        yes.add(True) if value == noexception(Exception, object.__getattribute__, item, key,
                                                              default_=NotImplemented) else yes.add(False)
                    if yes == {True}:
                        return item

    def first_anyin(self, *args, **kwargs):
        """
        ROW: Get FIRST container item/row when ANY args or kwargs are found IN item/row.

        Recursivily for MutableMapping.

        Examples:
            >>> from inspect import classify_class_attrs
            >>> from rc import BUILTINS_CLASSES
            >>> from rc import Get
            >>> from rc import pretty_install
            >>> from rc import Seq
            >>> from rc import slist
            >>> from rc import stuple
            >>>
            >>> pretty_install()
            >>>
            #
            # args
            #
            >>> s = slist([dict(a=1), ['b', 'd'], None, dict(a=dict(b=dict(c=1)), b=2), ])
            >>> s.first_anyin('b', 'c')
            ['b', 'd']
            >>> s.first_anyin('d')
            ['b', 'd']
            >>> s.first_anyin('e', c=1)
            {'a': {'b': {'c': 1}}, 'b': 2}
            >>> s = stuple([dict(a=1), ['b', 'd'], None, dict(a=dict(b=dict(c=1)), b=2), ])
            >>> s.first_anyin('a', 'e')
            {'a': 1}
            >>> s.first_anyin('', 'e') is None
            True
            >>>
            #
            # Kwargs - __getitem__
            #
            >>> # noinspection PyUnresolvedReferences
            >>> classified = slist([i._asdict() for i in classify_class_attrs(slist)])
            >>> d_getitem = classified.first_anyin(test=None, kind='data')
            >>> d_getitem  # doctest: +ELLIPSIS
            {
                'name': '__abstractmethods__',
                'kind': 'data',
                'defining_class': <class '....slist'>,
                'object': frozenset()
            }
            >>>
            #
            # Kwargs - __getattribute__
            #
            >>> classified = slist(classify_class_attrs(slist))
            >>> d_getattr = classified.first_anyin(test=None, kind='data')
            >>> d_getattr  # doctest: +ELLIPSIS
            Attribute(name='__abstractmethods__', kind='data', defining_class=<class '....slist'>, object=frozenset())
            >>>
            >>> d_getitem == d_getattr._asdict()
            True

        Args:
            *args: to use __contains__ in container value/column.
            **kwargs: to use __getattribute__ or __getitem__ in container value/column.

        Returns:
            ROW: FIRST container item/row found where ANY arg or kwarg are IN item/row.
        """
        for item in self:
            es = Es(item)
            if es.container:
                if es.mm:
                    for arg in args:
                        if nested_lookup(arg, item):
                            return item
                    for key, value in kwargs.items():
                        if in_dict(item, {key: value}):
                            return item
                else:
                    for arg in args:
                        if arg in item:
                            return item
                    for key, value in kwargs.items():
                        if value == noexception(Exception, object.__getattribute__, item, key,
                                                default_=NotImplemented):
                            return item

    def get(self, *args, **kwargs):
        """
        ROWS: Get ALL container items/rows found if ALL args/columns and (kwargs(key: value) or kwargs/arg/callable
        pred)
            are found IN item/row.

        Recursivily for MutableMapping.

        Examples:
            >>> from inspect import classify_class_attrs
            >>> from rc import BUILTINS_CLASSES
            >>> from rc import pretty_install
            >>> from rc import Seq
            >>> from rc import slist
            >>> from rc import stuple
            >>>
            >>> pretty_install()
            >>>
            #
            # args
            #
            >>> s = slist([dict(a=1), ['b', 'd'], None, dict(a=dict(b=dict(c=1)), b=2), ])
            >>> s.get('a', a=1) # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            TypeError: Key/attr "a": can only have a callable as value in kwargs: 1
            >>> s.get('a', 'd')
            slist[{'a': 1}, {'a': {'b': {'c': 1}}, 'b': 2}]
            >>>
            >>> s = stuple([dict(e=1), ['b', 'd'], None, dict(a=dict(b=dict(c=1)), b=2), *classify_class_attrs(stuple)])
            >>> s.get(kind='method', defining_class=stuple)  # doctest: +ELLIPSIS
            stuple(Attribute(name='__init__', kind='method', defining_class=<class '....stuple'>, \
object=<function stuple.__init__ at 0x...>), Attribute(name='__repr__', kind='method', \
defining_class=<class '....stuple'>, object=<function stuple.__repr__ at 0x...>))

        Args:
            *args: to use __contains__ in container value/column.
            **kwargs: Callable for arg or key/value to use __getattribute__ or __getitem__ in container value/column.

        Raises:
            TypeError: Key/attr "{arg}": can only have a callable as value in kwargs: {func} and not a value

        Returns:
            ROWS: ALL container items/rows found if ALL args and kwargs are IN item.
        """

        def check(f: Any, a: str):
            if not Es(f).callable:
                raise TypeError(f'Key/attr "{a}": can only have a callable as value in kwargs: {f}')

        rv = slist()
        for item in self:
            append = set()
            es = Es(item)
            if es.mm:
                for arg in args:
                    if nested_lookup(arg, item):
                        func = kwargs.pop(arg, lambda x: True)
                        check(func, arg)
                        append.add(True) if func(item[arg]) else append.add(False)
                for key, value in kwargs.items():
                    append.add(True) if in_dict(item, {key: value}) else append.add(False)
            else:
                for arg in args:
                    if arg in dir(item):
                        func = kwargs.pop(arg, lambda x: True)
                        check(func, arg)
                        value = noexception(Exception, object.__getattribute__, item, arg, default_=NotImplemented)
                        append.add(True) if value is not NotImplemented and func(value) else append.add(False)
                for key, value in kwargs.items():
                    append.add(True) if value == noexception(Exception, object.__getattribute__, item, key,
                                                             default_=NotImplemented) else append.add(False)
            if append == {True}:
                rv.append(item)
        rv = rv if Es(self.__class__).slist else self.__class__(rv)
        return rv[0] if len(rv) == 1 else rv

    def value(self, *args, **kwargs):
        """
        COLUMNS: GET attrs/keys/args values/columns for each item/row if attr/key/arg is found (default: 'name').

        Recursivily for MutableMapping.

        Examples:
            >>> import inspect
            >>> from inspect import classify_class_attrs
            >>> from rc import Get
            >>> from rc import pretty_install
            >>> from rc import Seq
            >>> from rc import slist
            >>> from rc import stuple
            >>>
            >>> pretty_install()
            >>>
            #
            # Attribute
            #
            >>> s = slist(classify_class_attrs(slist))
            >>> s.value() == slist(dir(slist))
            True
            >>> # noinspection PyUnresolvedReferences
            >>> s.value(name=Access.PUBLIC.include) == slist([i for i in dir(slist) if not i.startswith('_')])
            True
            >>>
            #
            # Dict
            #
            >>> # noinspection PyUnresolvedReferences
            >>> s = stuple([i._asdict() for i in classify_class_attrs(slist)])
            >>> s.value() == stuple(dir(slist))
            True
            >>> # noinspection PyUnresolvedReferences
            >>> s.value(name=Access.PUBLIC.include) == stuple([i for i in dir(slist) if not i.startswith('_')])
            True
            >>>

        Args:
            *args: attribute/key names to get values (default: 'name').
            **kwargs: function to filter each arg (default: lambda x: True)

        Returns:
            COLUMNS: Values for attrs/keys/args found. nested_lookup values with no keys returned for MutaleMapping.
        """
        args = args if args else ('name', )
        rv = slist()
        for item in self:
            es = Es(item)
            one = len(args) == 1
            rv_sub = slist()
            for arg in args:
                value = NotImplemented
                if es.mm:
                    if val := nested_lookup(arg, item):
                        value = val[0] if len(val) == 1 else val
                elif arg in dir(item):
                    value = object.__getattribute__(item, arg)
                if value is not NotImplemented:
                    func = kwargs.get(arg, lambda x: True)
                    call = func(value)
                    if func(value):
                        rv.append(value) if one else rv_sub.append(value)
            if rv_sub:
                rv.append(rv_sub if Es(self.__class__).slist else self.__class__(rv_sub))
        rv = rv if Es(self.__class__).slist else self.__class__(rv)
        return rv[0] if len(rv) == 1 else rv

