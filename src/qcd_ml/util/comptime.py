import functools
from typing import Iterable, Callable


"""
Brings comptime evaluation to functions.
"""

class ComptimeFunc:
    def __init__(self, func: Callable, comptime_args: Iterable):
        self._values = {
                arg: func(*arg) for arg in comptime_args
                }
        functools.update_wrapper(self, func)

    def __call__(self, *args):
        try:
            return self._values[args]
        except:
            pass
        raise ValueError(f"{args} were not known at compile time")

def comptime(comptime_args: Iterable):
    def _comptime_eval(func: Callable):
        return ComptimeFunc(func, comptime_args)
    return _comptime_eval
