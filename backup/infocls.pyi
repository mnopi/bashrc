InfoCls = NamedTuple('InfoCls', annotation=tuple[Annotation], builtin=Type, classified=tuple[inspect.Attribute],
                     cls=Type, es=Es, fields=dict[str, Field], importable=str, modname=str, mro=tuple[Type, ...],
                     name=str, qual=str, super=Type)


