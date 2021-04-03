import inspect
from collections import namedtuple

from rc import CLI
from rc import Path
from rc import project

PathTest = namedtuple('PathTest', 'file git modname name package path prefix relative')


def get_console_script():
    module = f'{project.name}.{CLI}'
    variable = project.instance_name
    app = f'{variable}.{CLI}'

    return f"# -*- coding: utf-8 -*-\nimport re\nimport sys\n" \
           f"from {module} import {variable}\n\nif __name__ == '__main__':\n" \
           f"    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])\n" \
           f"    sys.exit({app}.{CLI}())"


def get_path(f: Path) -> PathTest:
    # Pytest first import conftest so package is tests.conftest since it is the first import to bapy
    # However, run configuration first import bapy and latter imports debug from conftest.
    file = Path(f)
    top = Path.git(file)
    modname = inspect.getmodulename(file.text)
    path = file.parent
    name = file.parent.name
    p = f'{name}.{modname}'
    relative = file.relative_to(top.path)
    return PathTest(file, top, modname, name, p, path, f'{name.upper()}_', relative)
