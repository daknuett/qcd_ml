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
from typing import Any, Callable

import gpt as g
import numpy as np


def lattice2ndarray(lattice: Any) -> np.ndarray:
    """
    Converts a `lehner/gpt <https://github.com/lehner/gpt>`_ lattice to a numpy ndarray
    keeping the ordering of axes as one would expect.

    Args:
        lattice: A gpt lattice object to convert.

    Returns:
        A numpy ndarray with axes reordered to match the standard convention.
        The axes are swapped such that the time dimension is first, followed by
        the spatial dimensions.

    Example::

        q_top = g.qcd.gauge.topological_charge_5LI(U_smeared, field=True)
        plot_scalar_field(lattice2ndarray(q_top))
    """
    shape = lattice.grid.fdimensions
    shape = list(reversed(shape))
    if lattice[:].shape[1:] != (1,):
        shape.extend(lattice[:].shape[1:])

    coordinates = g.coordinates(lattice)

    result = lattice[coordinates].reshape(shape)
    result = np.swapaxes(result, 0, 3)
    result = np.swapaxes(result, 1, 2)
    return result


def ndarray2lattice(
    ndarray: np.ndarray,
    grid: Any,
    lat_constructor: Callable[..., Any]
) -> Any:
    """
    Converts an ndarray to a gpt lattice, it is the inverse
    of lattice2ndarray.

    Args:
        ndarray: The numpy array to convert to a gpt lattice.
        grid: The gpt grid object defining the lattice dimensions.
        lat_constructor: The gpt lattice constructor (e.g., g.vspincolor, g.vcolor, etc.)
            that defines the type of the lattice.

    Returns:
        A gpt lattice object with data from the ndarray.

    Example::

        lat = ndarray2lattice(arr, g.grid([4,4,4,8], g.double), g.vspincolor)
    """
    lat = lat_constructor(grid)
    data = np.swapaxes(ndarray, 0, 3)
    data = np.swapaxes(data, 1, 2)
    coordinates = g.coordinates(lat)
    lat[coordinates] = data.reshape([data.shape[0] * data.shape[1] * data.shape[2] * data.shape[3]] + list(data.shape[4:]))
    return lat
