qcd_ml -- some machine learning layers for QCD 
**********************************************

written using torch.

.. image:: https://www.nfdi.de/wp-content/uploads/2021/12/PUNCH4NFDI-Logo_RGB.png 
   :target: https://www.nfdi.de/punch4nfdi/
   :width: 80px

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
- Gauge-equivariant paths for vector-like objects.

Neural Network
--------------

Currently, only vector-like objects an be handled by neural networks.

- ``v_PTC`` and ``v_LPTC``


QCD
---

- Euclidean gamma matrices.
- Wilson Dirac operator and Wilson-Clover Dirac operator.

Utilities
---------

- GMRES iterative solver.
- ``ZPP_Multigrid``: Zero-Point-Projected Multigrid.

Compatibility
-------------

- Compatibility layer for `lehner/gpt<https://github.com/lehner/gpt>`_

