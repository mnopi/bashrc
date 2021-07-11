from enum import Enum
from enum import auto
from os import environ
from urllib.parse import urlparse
from environs import Env
from furl import furl

env = Env()


class Test(Enum):
    A = auto()
    B = auto()


@env.parser_for("furl")
def furl_parser(value):
    return furl(value)


environ['VAR'] = 'https://github.com'
env.read_env(override=True)
assert env.furl('VAR') == furl('https://github.com')

environ['VAR'] = 'https://github.com'
env.read_env(override=True)
assert env.url('VAR') == urlparse('https://github.com')

environ['VAR'] = '1, 2, 3,4'
env.read_env(override=True)
assert env.list('VAR', subcast=int) == [1, 2, 3, 4]

environ['VAR'] = '1 2 3 4'
env.read_env(override=True)
assert env.list('VAR', delimiter=' ', subcast=int) == [1, 2, 3, 4]

environ['VAR'] = '2=2, 3= 3, 4 = 4'
env.read_env(override=True)
assert env.dict('VAR', subcast_keys=int, subcast_values=int) == {2: 2, 3: 3, 4: 4}

environ['VAR'] = '{"name":"John", "age":30, "car":null}'
env.read_env(override=True)
assert env.json('VAR') == {'name': 'John', 'age': 30, 'car': None}

environ['VAR'] = 'a'
env.read_env(override=True)
assert env.enum('VAR', ignore_case=True, type=Test) == Test.A
