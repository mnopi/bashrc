
def test_framedata():
    data = framedata(STACK, index=0)
    ic(data.code_context, data.file, data.function, data.git, data.imported, data.installed, data.lineno, data.modname,
       data.name, data.origin, data.package, data.path, data.spec)
    assert data.file == data.origin
    assert data.function == FUNCTION_MODULE
    assert data.git.path == data.path.parent
    assert data.installed is None
    assert data.name == f'{data.package}.{data.modname}'
    assert data.package == data.path.name


def test_importlib(pathtest):
    p = Package()

    assert p.importlib_contents is None
    assert p.importlib_files is None
    assert p.importlib_module is None
    assert p.importlib_spec is None

    assert core.file.name in package.bapy.importlib_contents
    assert pathtest.file.name in package.importlib_contents

    assert package.bapy.importlib_files == package.bapy.path
    assert package.importlib_files == package.path == pathtest.path

    assert package.bapy.importlib_module.__name__ == package.bapy.package
    assert package.bapy.importlib_module.__package__ == package.bapy.path.name

    assert package.importlib_module.__name__ == package.package
    assert package.importlib_module.__package__ == package.path.name

    assert package.bapy.importlib_spec.origin == package.bapy.file.text
    assert package.importlib_spec.origin == package.file.text

