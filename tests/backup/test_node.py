import ast

from rc import Ast
from rc import icc
from rc import Path
from rc._project import TESTS
from tests.data import frame

a = Ast(frame.__file__)

# for line in [80, 81, 83, 85, 86, 87, 88, 90, 91, 92, 95, 96, 97, 98, 100, 102, 103, 106, 107, 110, 115, 116, 121, 121, ]:
#     icc(a(lineno=line))
#
imports = set()

project = Path.gitpath()
packages = project.find_packages
packages.remove(TESTS)
for package in packages:
    files = (project / package).glob('**/*.py')
    for file in files:
        if '__init__.py' not in file:
            code = file.read_text()
            nodes = ast.parse(code, filename=frame.__file__)
            for n in nodes.body:
                # icc(ast.get_source_segment(code, n).splitlines())
                text = ast.get_source_segment(code, n)
                if (text.startswith('import ') or text.startswith('from ')) and ' .' not in text:
                    imports.add(text.rstrip(' '))
# icc('\n'.join(sorted(imports)))

# icc(imports)
for i in sorted(imports):
    icc(i)
    exec(i)

icc(Union)

# packages.remove(TESTS)
# icc(packages)
# icc(Path.cwd().find_packages)
# for file in Path.cwd().glob('')
# code = Path(node.__file__).read_text()
# nodes = ast.parse(code, filename=node.__file__)
# imports = set()
# for n in nodes.body:
#     icc(ast.get_source_segment(code, n).splitlines())
#     text = ast.get_source_segment(code, n)
#     if (text.startswith('import ') or text.startswith('from ')) and ' .' not in text:
#         imports.add(text.rstrip(' '))

# icc('\n'.join(sorted(imports)))

