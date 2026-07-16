import functools
from typing import Iterable, Callable, Any, Tuple

"""
Brings comptime evaluation to functions.

This module provides the ``comptime`` decorator which allows functions to be
evaluated at compile-time for known arguments, improving performance.
"""


class ComptimeFunc:
    """Callable wrapper that caches function results for compile-time known arguments.

    This class wraps a function and caches its results for specific arguments that
    are known at compile time. When called with these arguments, it returns the
    cached result instead of re-computing.

    Attributes:
        _values: Dictionary mapping argument tuples to their pre-computed values.
    """

    def __init__(self, func: Callable, comptime_args: Iterable) -> None:
        """Initialize the ComptimeFunc wrapper.

        Args:
            func: The function to wrap.
            comptime_args: Iterable of argument tuples to pre-compute at compile time.
        """
        self._values = {
                arg: func(*arg) for arg in comptime_args
                }
        functools.update_wrapper(self, func)

    def __call__(self, *args: Any) -> Any:
        """Call the wrapped function with cached results for compile-time known arguments.

        Args:
            *args: Positional arguments to pass to the function.

        Returns:
            Any: The pre-computed value if args match a compile-time known argument.

        Raises:
            ValueError: If the provided arguments were not known at compile time.
        """
        try:
            return self._values[args]
        except:
            raise ValueError(f"{args} were not known at compile time")


def comptime(comptime_args: Iterable) -> Callable[[Callable], ComptimeFunc]:
    """Decorator to enable compile-time evaluation for specific arguments.

    This decorator wraps a function and pre-computes its results for the provided
    compile-time arguments. When the wrapped function is called with these
    arguments, it returns the cached result.

    Args:
        comptime_args: Iterable of argument tuples to pre-compute.

    Returns:
        A decorator function that wraps the target function.

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
