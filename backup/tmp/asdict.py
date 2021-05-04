from threading import Lock
_block
_count
_owner

@singledispatchmethod
def asdict(data: Semaphore, convert=True):
    return dict(locked=data.locked(), value=data._value) if convert else data


@asdict.register
def asdict_chain(data: Chain, convert=True):
    data.rv = ChainRV.FIRST
    return dict(data) if convert else data


@asdict.register
def asdict_enum(data: Enum, convert=True):
    return {data.name: data.value} if convert else data


@asdict.register
def asdict_environs(data: Environs, convert=True):
    return data.dump() if convert else data


@asdict.register
def asdict_gettype(data: GetType, convert=True):
    return data if convert else data


@asdict.register
def asdict_gitsymbolic(data: GitSymbolicReference, convert=True):
    return dict(repo=data.repo, path=data.path) if convert else data


@asdict.register
def asdict_logger(data: Logger, convert=True):
    return dict(name=data.name, level=data.level) if convert else data


@asdict.register
def asdict_namedtype(data: NamedType, convert=True):
    return data._asdict() if convert else data


@asdict.register
def asdict_remote(data: Remote, convert=True):
    return dict(repo=data.repo, name=data.name) if convert else data


def asdict_ignorestr(data, convert=True):
    return str(data) if convert else data


def asdict_props(data, key=Attr.PUBLIC, pprop=False):
    """
    Properties to dict.

    Examples:
        >>> from rc import asdict_props
        >>> from rc import Es
        >>> from rc import pretty_install
        >>>
        >>> pretty_install()
        >>>

    Args:
        data: object to get properties.
        key: startswith to include.
        pprop: convert only :class:`rc.pproperty` or all properties excluding __ignore_attr__.

    Returns:
        Dict with names and values for properties.
    """
    pass


def asdict_type(data, convert=True):
    rv = data
    es = Es(data)
    if es.enum_sub:
        rv = {key: value._value_ for key, value in data.__members__.items()} if convert else data
    elif es.datatype_sub or es.dicttype_sub or es.namedtype_sub or es.slotstype_sub:
        rv = info(data).defaults if convert else data
    return rv

