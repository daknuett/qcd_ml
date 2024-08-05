"""
qcd_ml.compat
=============

Compatibility layers for other frameworks.

The documentation of the sub-modules is included here, because 
the soft requirements may not be met.

qcd_ml.compat.gpt
_________________

Compatibility to `lehner/gpt <https://github.com/lehner/gpt>`_. 

This module provides 

``lattice2ndarray(lattice)``
    Converts a gpt  lattice to a numpy ndarray 
    keeping the ordering of axes as one would expect.

    Example::

        q_top = g.qcd.gauge.topological_charge_5LI(U_smeared, field=True)
        plot_scalar_field(lattice2ndarray(q_top))

``ndarray2lattice(ndarray, grid, lat_constructor)``
    Converts an ndarray to a gpt lattice, it is the inverse 
    of lattice2ndarray.

    Example::

        lat = ndarray2lattice(arr, g.grid([4,4,4,8], g.double), g.vspincolor)

Using these two functions it is possible to use gpt operators
in torch or numpy::

    w_gpt = g.qcd.fermion.wilson_clover(U_gpt, {"mass": -0.58,
        "csw_r": 1.0,
        "csw_t": 1.0,
        "xi_0": 1.0,
        "nu": 1.0,
        "isAnisotropic": False,
        "boundary_phases": [1,1,1,1]})

    w = lambda x: torch.tensor(lattice2ndarray(w_gpt(ndarray2lattice(x.numpy(), U_gpt[0].grid, g.vspincolor))))

"""
