r"""
qcd_ml.nn.matrix_layers
=======================

Layers for matrix valued fields, i.e., fields that transform as
.. math::
    M(x) \rightarrow \Omega(x) M(x) \Omega(x).

Provides the following layers:

- ``LGE_Convolution``
- ``LGE_Bilinear``
- ``LGE_ReTrAct``
- ``LGE_Exp``
- ``PolyakovLoopGenerator`` and ``PositiveOrientationPlaquetteGenerator``

See [10.1103/PhysRevLett.128.032003].
"""

from .convolution import LGE_Convolution
from .bilinear import LGE_Bilinear
from .loop_generator import PolyakovLoopGenerator, PositiveOrientationPlaquetteGenerator
from .activation import LGE_ReTrAct
from .exponentiation import LGE_Exp
