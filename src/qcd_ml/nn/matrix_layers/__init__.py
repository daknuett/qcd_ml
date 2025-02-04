r"""
Layers for matrix valued fields, i.e., fields that transform as
.. math::
    M(x) \rightarrow \Omega(x) M(x) \Omega(x)
"""

from .convolution import LGE_Convolution
from .bilinear import LGE_Bilinear
from .loop_generator import PolyakovLoopGenerator, PositiveOrientationPlaquetteGenerator
