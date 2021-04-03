from contextlib import suppress

from rc import CLI
from rc import cmd
from rc import cmdname
from rc import FUNCTION_MODULE
from rc import icc
from rc import InstallScriptPath
from rc import Path
from rc import project
from rc import TESTS
from rc.cli import tests_cmd as _command
from rc._info import __file__ as __file_info__
from tests.helpers import get_console_script
from tests.helpers import get_path


this = get_path(Path(__file__))
main = get_path(Path(__file_info__))
cli_dir = Path(InstallScriptPath.path())
cli = cli_dir / project.name


def test_main(console_script):
    icc(project.main.file, main.file, project.main.frame, project.main.index)
    assert project.main.file == main.file
    cli.rm()
    cli.write_text(console_script)
    cli.chmod('+x')
    assert project.init.file.text in cmd(f'{cli} {cmdname(_command)}', py=True).stdout
    assert project.main.file.text in cmd(f'{cli} {cmdname(_command)} --main', py=True).stdout
    cli.rm()
    assert project.main.init_py == main.path / '__init__.py'
    assert project.main.path == main.path
    assert project.main.prefix == f'{main.name.upper()}_'
    assert project.main.package == main.package
    assert project.main.project == main.path.parent


def test_framedata():
    pass
    # data = framedata(STACK, index=0)
    # ic(data.code_context, data.file, data.function, data.git, data.imported, data.installed, data.lineno, data.modname,
    #    data.name, data.origin, data.package, data.path, data.spec)
    # assert data.file == data.origin
    # assert data.function == FUNCTION_MODULE
    # assert data.git.path == data.path.parent
    # assert data.installed is None
    # assert data.name == f'{data.package}.{data.modname}'
    # assert data.package == data.path.name


def test_importlib(pathtest):
    pass
    # p = Package()
    #
    # assert p.importlib_contents is None
    # assert p.importlib_files is None
    # assert p.importlib_module is None
    # assert p.importlib_spec is None
    #
    # assert core.file.name in package.bapy.importlib_contents
    # assert pathtest.file.name in package.importlib_contents
    #
    # assert package.bapy.importlib_files == package.bapy.path
    # assert package.importlib_files == package.path == pathtest.path
    #
    # assert package.bapy.importlib_module.__name__ == package.bapy.package
    # assert package.bapy.importlib_module.__package__ == package.bapy.path.name
    #
    # assert package.importlib_module.__name__ == package.package
    # assert package.importlib_module.__package__ == package.path.name
    #
    # assert package.bapy.importlib_spec.origin == package.bapy.file.text
    # assert package.importlib_spec.origin == package.file.text


def run():
    console_script = get_console_script()
    test_main(console_script)
    # module = f'{project.name}.{CLI}'
    # variable = project.instance_name
    # app = f'{variable}.{CLI}'
    # cs = f"# -*- coding: utf-8 -*-\nimport re\nimport sys\n" \
    #      f"from {module} import {variable}\n\nif __name__ == '__main__':\n" \
    #      f"    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])\n" \
    #      f"    sys.exit({app}.{CLI}())"
    # with Path.tmp() as p:
    #     test_importlib(p)


with suppress(ModuleNotFoundError):
    from conftest import ic
    if ic.enabled:
        run()
