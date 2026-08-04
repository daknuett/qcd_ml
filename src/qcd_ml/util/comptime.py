import functools
from typing import Iterable, Callable, Any, Tuple

"""
Brings comptime evaluation to functions.

This module provides the ``comptime`` decorator which allows functions to be
evaluated at compile-time for known arguments, improving performance.
"""


class ComptimeFunc:
    def __init__(self, func: Callable, comptime_args: Iterable) -> None:
        self._values = {
                arg: func(*arg) for arg in comptime_args
                }
        functools.update_wrapper(self, func)

    def __call__(self, *args: Any) -> Any:
        try:
            return self._values[args]
        except:
            raise ValueError(f"{args} were not known at compile time")


def comptime(comptime_args: Iterable) -> Callable[[Callable], ComptimeFunc]:
    """Decorator to enable compile-time evaluation for specific arguments.

    This decorator wraps a function and pre-computes its results for the provided
    compile-time arguments. When the wrapped function is called with these
    arguments, it returns the cached result.

    Example::

        @comptime([(1, 2), (3, 4)])
        def add(a, b):
            return a + b

        result = add(1, 2)  # Returns pre-computed value 3
        result = add(3, 4)  # Returns pre-computed value 7
        result = add(5, 6)  # Raises ValueError: (5, 6) were not known at compile time
    """
    def _comptime_eval(func: Callable) -> ComptimeFunc:
        return ComptimeFunc(func, comptime_args)

    return _comptime_eval
