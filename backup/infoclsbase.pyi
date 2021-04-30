# InfoClsBase = NamedTuple('InfoClsBase', annotation=dict[str, Any], attribute=dict[str, Attribute],
#                          builtin=Optional[Type], cls=Optional[Type], es=Es, importable=str, modname=str,
#                          mro=Optional[tuple[Type, ...]], name=Optional[str], qual=Optional[str], super=Optional[Type])

InfoClsBase = NamedTuple('InfoClsBase', annotation=stuple[Annotation], attribute=stuple[Attribute],
                         builtin=Optional[Type], cls=Optional[Type], es=Es, importable=str, modname=str,
                         mro=Optional[tuple[Type, ...]], name=Optional[str], qual=Optional[str], super=Optional[Type])
