import enum
import os


class _EntryPointType(enum.Enum):
    PYTHON_PACKAGE = 'PYTHON_PACKAGE'
    PYTHON_PROGRAM = 'PYTHON_PROGRAM'
    COMMAND = 'COMMAND'


PYTHON_PACKAGE = _EntryPointType.PYTHON_PACKAGE
PYTHON_PROGRAM = _EntryPointType.PYTHON_PROGRAM
COMMAND = _EntryPointType.COMMAND


def get(path, name):  # type: (str, str) -> _EntryPointType
    """
    Args:
        path (string): Directory where the entry point is located
        name (string): Name of the entry point file

    Returns:
        (_EntryPointType): The type of the entry point
    """
    if 'setup.py' in os.listdir(path):
        return _EntryPointType.PYTHON_PACKAGE
    elif name.endswith('.py'):
        return _EntryPointType.PYTHON_PROGRAM
    else:
        return _EntryPointType.COMMAND
