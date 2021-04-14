import inspect
import pathlib
import re

import typer


def all_var(file: str = None):
    file = pathlib.Path(file if file else inspect.stack()[1].filename)
    if file.exists():
        add = list()
        exclude = ['os.environ', '__all__', '@', 'import ', 'from ']
        lines = file.read_text().splitlines()
        for line in lines:
            if 'spec' in line:
                if not any([re.search('^ ', line), *[v in line for v in exclude], line == str()]):
                    found = False
                    for word in ['async def ', 'def ', 'class ']:
                        if line.startswith(word):
                            add.append("    '" + line.replace(':', str()).split(word)[1].split('(')[0] + "'")
                            found = True
                            break
                    if not found and ' = ' in line:
                        add.append("    '" + line.split(' = ')[0] + "'")

        print(f'__all__ = ({NEWLINE}' + f',{NEWLINE}'.join(add) + f'{NEWLINE})')


def main(file: str):
    """__all__ var"""
    all_var(file)


if __name__ == '__main__':
    typer.run(main)
