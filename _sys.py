import importlib
import sys
from sysconfig import get_paths
from types import ModuleType

sys.get_asyncgen_hooks()
sys.meta_path
sys.platlibdir
ModuleType
SYS_BUILTIN = sys.builtin_module_names
SYS_MODULES = sys.modules
ModuleSpec = importlib._bootstrap.ModuleSpec
origins = ['frozen', 'built-in']
SYS_PATHS = get_paths()
SYS_PATHS_EXCLUDE = (SYS_PATHS['stdlib'], SYS_PATHS['purelib'], SYS_PATHS['include'], SYS_PATHS['platinclude'], SYS_PATHS['scripts'])
file = '/Users/jose'
exclude = True
for f in SYS_PATHS_EXCLUDE:
    if file in f:
        break


SYS_PATHS_PATH = Box({key: Path(value) for key, value in get_paths().items()})
