"""
qcd_ml.compat.gpt
=================

Compatibility to `lehner/gpt <https://github.com/lehner/gpt>`_. 

Using the two functions ``lattice2ndarray`` and ``ndarray2lattice`` it is possible to 
use gpt operators in torch or numpy::

    w_gpt = g.qcd.fermion.wilson_clover(U_gpt, {"mass": -0.58,
        "csw_r": 1.0,
        "csw_t": 1.0,
        "xi_0": 1.0,
        "nu": 1.0,
        "isAnisotropic": False,
        "boundary_phases": [1,1,1,1]})

    w = lambda x: torch.tensor(lattice2ndarray(w_gpt(ndarray2lattice(x.numpy(), U_gpt[0].grid, g.vspincolor))))

"""

import gpt as g
import numpy as np


def lattice2ndarray(lattice):
    """ 
    Converts a `lehner/gpt <https://github.com/lehner/gpt>`_ lattice to a numpy ndarray 
    keeping the ordering of axes as one would expect.
    Example::

        q_top = g.qcd.gauge.topological_charge_5LI(U_smeared, field=True)
        plot_scalar_field(lattice2ndarray(q_top))
    """
    fdims = lattice.grid.fdimensions
    shape = lattice.grid.fdimensions
    shape = list(reversed(shape))
    if lattice[:].shape[1:] != (1,):
        shape.extend(lattice[:].shape[1:])

    coordinates = g.coordinates(lattice)
   
    result = lattice[coordinates].reshape(shape)
    for i in range(len(fdims) // 2):
        result = np.swapaxes(result, i, len(fdims) - i - 1)
    return result

def ndarray2lattice(ndarray, grid, lat_constructor):
    """
    Converts an ndarray to a gpt lattice, it is the inverse 
    of lattice2ndarray.
    Example::

        lat = ndarray2lattice(arr, g.grid([4,4,4,8], g.double), g.vspincolor)
    """
    data = ndarray
    fdims = grid.fdimensions
    
    lat = lat_constructor(grid)
    
    for i in range(len(fdims) // 2):
        data = np.swapaxes(data, i, len(fdims) - i - 1)
        
    coordinates = g.coordinates(lat)
    lat[coordinates] = data.reshape([np.prod(np.array(fdims))] + list(data.shape[len(fdims):]))
    return lat
