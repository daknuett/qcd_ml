qcd_ml -- some machine learning layers for QCD 
**********************************************

written using torch.

.. image:: https://zenodo.org/badge/821360626.svg
  :target: https://zenodo.org/doi/10.5281/zenodo.13254662

.. image:: https://github.com/daknuett/qcd_ml/actions/workflows/python-package.yml/badge.svg

.. image:: https://www.nfdi.de/wp-content/uploads/2021/12/PUNCH4NFDI-Logo_RGB.png 
   :target: https://www.nfdi.de/punch4nfdi/
   :width: 80px

`DOCUMENTATION <https://daknuett.github.io/qcd_ml/>`_

.. contents::

Getting Started
===============

Installation
------------

The easiest way to install ``qcd_ml`` is by cloning the repository
and installing the package using ``pip``::

    git clone https://github.com/daknuett/qcd_ml
    cd qcd_ml 
    pip install .

This will install all dependencies (mostly ``numpy`` and ``pytorch``)
automatically.

If you need a specific version of ``numpy`` or ``pytorch``, install them manually
before installing ``qcd_ml``.


Functionality
=============

Base
----

- Various group operations of gauge and spin group.
- Gauge-equivariant vector hop.
- Gauge-equivariant paths for vector-like and matrix-like objects.
- Gauge-equivariant path-buffers for vector-like and matrix-like objects.

Neural Network
--------------

For vector-like objects the following layers are provided:

- ``v_PTC``, ``v_LPTC``, and ``v_LPTC_NG`` that implement `2302.05419 <http://arxiv.org/abs/2302.05419>`_.
- ``v_ProjectLayer`` that implements `2304.10438 <https://arxiv.org/abs/2304.10438>`_ parallel transport pooling. 
- ``v_PT`` and ``v_Dense``

For matrix-like objects the following layers are provided:

- ``LGE_Convolution``
- ``LGE_Bilinear``
- ``LGE_ReTrAct``
- ``LGE_Exp``
- ``PolyakovLoopGenerator`` and ``PositiveOrientationPlaquetteGenerator``

See `10.1103/PhysRevLett.128.032003 <https://doi.org/10.1103/PhysRevLett.128.032003>`_.



QCD
---

- Euclidean gamma matrices.
- Wilson Dirac operator and Wilson-Clover Dirac operator.
- Stout link smearing.
- Plaquette and topological charge field.

Utilities
---------

- GMRES iterative solver.
- ``ZPP_Multigrid``: Zero-Point-Projected Multigrid.
- Coarsened 9-point operators for Multigrid.

Compatibility
-------------

- Compatibility layer for `lehner/gpt <https://github.com/lehner/gpt>`_

