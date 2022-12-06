import sys


def get_generic_type_arg(cls):
    t = cls.__orig_bases__[0]
    if sys.version_info >= (3, 8):
        from typing import get_args

        return get_args(t)[0]

    return t.__args__[0]
