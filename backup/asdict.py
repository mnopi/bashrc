class AsDict:
    """
    Dict and Attributes Class.

    Examples:

        .. code-block:: python

            json = jsonpickle.encode(col)
            obj = jsonpickle.decode(obj)
            col.to_file(name=col.col_name)
            assert (Path.cwd() / f'{col.col_name}.json').is_file()
            col.to_file(directory=tmpdir, name=col.col_name, regenerate=True)
            obj = col.from_file(directory=tmpdir, name=col.col_name)
            assert obj == col
    """
    __ignore_attr__ = ['asdict', 'attrs', 'keys', 'kwargs', 'kwargs_dict', 'public', 'values', 'values_dict', ]

    @property
    def asdict(self):
        """
        Dict including properties without routines and recursive.

        Returns:
            dict:
        """
        return info(self).asdict()

    @property
    def attrs(self):
        """
        Attrs including properties.

        Excludes:
            __ignore_attr__
            __ignore_copy__ instances.
            __ignore_kwarg__

        Returns:
            list:
        """
        return info(self).attrs

    def attrs_get(self, *args, default=None):
        """
        Get One or More Attributes.

        Examples:
            >>> from rc import AsDict
            >>> a = AsDict()
            >>> a.d1 = 1
            >>> a.d2 = 2
            >>> a.d3 = 3
            >>> assert a.attrs_get('d1') == {'d1': 1}
            >>> assert a.attrs_get('d1', 'd3') == {'d1': 1, 'd3': 3}
            >>> assert a.attrs_get('d1', 'd4', default=False) == {'d1': 1, 'd4': False}

        Raises:
            ValueError: ValueError

        Args:
            *args: attr(s) name(s).
            default: default.

        Returns:
            dict:
        """
        if not args:
            raise ValueError(f'args must be provided.')
        return {item: getattr(self, item, default) for item in args}

    def attrs_set(self, *args, **kwargs):
        """
        Sets one or more attributes.

        Examples:
            >>> from rc import AsDict
            >>> a = AsDict()
            >>> a.attrs_set(d_1=31, d_2=32)
            >>> a.attrs_set('d_3', 33)
            >>> d_4_5 = dict(d_4=4, d_5=5)
            >>> a.attrs_set(d_4_5)
            >>> a.attrs_set('c_6', 36, c_7=37)


        Raises:
            ValueError: ValueError

        Args:
            *args: attr name and value.
            **kwargs: attrs names and values.
        """
        if args:
            if len(args) > 2 or (len(args) == 1 and not isinstance(args[0], dict)):
                raise ValueError(f'args, invalid args length: {args}. One dict or two args (var name and value.')
            kwargs.update({args[0]: args[1]} if len(args) == 2 else args[0])

        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def defaults(cls, nested=True):
        """
        Return a dict with class attributes names and values.

        Returns:
            list:
        """
        return info(cls, depth=None if nested else 1).asdict(defaults=True)

    def from_file(self, directory=None, name=None, keys=True):
        name = name if name else self.__class__.__name__
        directory = PathLib(directory) if directory else PathLib.cwd()
        with (PathLib(directory) / f'{name}.json').open() as f:
            return jsonpickle.decode(json.load(f), keys=keys)

    @property
    def keys(self):
        """
        Keys from kwargs to init class (not InitVars), exclude __ignore_kwarg__ and properties.

        Returns:
            list:
        """
        return info(self).keys

    @property
    def kwargs(self):
        """
        Kwargs to init class with python objects no recursive, exclude __ignore_kwarg__ and properties.

        Example: Mongo binary.

        Returns:
            dict:
        """
        return info(self).kwargs

    @property
    def kwargs_dict(self):
        """
        Kwargs recursive to init class with python objects as dict, asdict excluding __ignore_kwarg__ and properties.

        Example: Mongo asdict.

        Returns:
            dict:
        """
        return info(self).kwargs_dict

    @property
    def public(self):
        """
        Dict including properties without routines.

        Returns:
            dict:
        """
        return info(self).public

    def to_file(self, directory=None, name=None, regenerate=False, **kwargs):
        name = name if name else self.__class__.__name__
        directory = PathLib(directory) if directory else PathLib.cwd()
        with (PathLib(directory) / f'{name}.json').open(mode='w') as f:
            json.dump(obj=info(self).to_json(regenerate=regenerate, **kwargs), fp=f, indent=4, sort_keys=True)

    def to_json(self, regenerate=True, indent=4, keys=True, max_depth=-1):
        return info(self).to_json(regenerate=regenerate, indent=indent, keys=keys, max_depth=max_depth)

    def to_obj(self, keys=True):
        return info(self).to_obj(keys=keys)

    @property
    def values(self):
        """
        Init python objects kwargs values no properties and not __ignore_kwarg__.

        Returns:
            list:
        """
        return info(self).values

    @property
    def values_dict(self):
        """
        Init python objects kwargs values no properties and not __ignore_kwarg__.

        Returns:
            list:
        """
        return info(self).values_dict
