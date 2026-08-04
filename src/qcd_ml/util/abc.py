"""
qcd_ml.util.abc
=============

Abstract Base Classes and ABCMeta classes.

Currently only ABCMetaWithProperty is implemented.
"""

from abc import ABCMeta

class ABCMetaWithProperty(ABCMeta):
    """
    This metaclass brings abstract class properties: Properties of the class that must be 
    implemented by subclasses. 

    Use as such::
    
        class A(metaclass=ABCMetaWithProperty, abstractclassproperties={"foobar"}):
            @abstractmethod
            def do_something(self, other):
                pass

        class C(A, abstractclassproperties={"baz"}):
            pass

        class B(C):
            foobar = 3
            baz = 4
            def do_something(self, other):
                return other

        class D(A):
            foobar = 9
            def do_something(self, other):
                return other


        b = B()
        d = D()

    
    """
    def __new__(mcls, clsname, bases, namespace, /, **kwargs):
        key = "abstractclassproperties"
        abstractclassproperties = set()
        if key in kwargs:
            abstractclassproperties = kwargs[key]
            del(kwargs[key])

        cls = super().__new__(mcls, clsname, bases, namespace, **kwargs)
        if not hasattr(cls, "_abstract_properties"):
            cls._abstract_properties = set()
        cls._abstract_properties = cls._abstract_properties | abstractclassproperties
        return cls

    def __call__(cls, *args, **kwargs):
        for required_attr in sorted(cls._abstract_properties):
            if not hasattr(cls, required_attr):
                raise TypeError(f"Can't instantiate abstract class {cls.__name__} without implementation for abstract class attribute '{required_attr}'")
        return super(ABCMetaWithProperty, cls).__call__(*args, **kwargs)

