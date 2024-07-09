import numpy as np

try:
    import gpt as g
    from qcd_ml.compat.gpt import lattice2ndarray, ndarray2lattice


    def test_lattice2array2lattice():
        grid = g.grid([4, 4, 4, 8], g.double)

        rng = g.random("foo")
        psi = g.vspincolor(grid)

        rng.cnormal(psi)

        ndarray1 = lattice2ndarray(psi)
        ndarray2 = lattice2ndarray(ndarray2lattice(ndarray1, grid, g.vspincolor))

        assert np.allclose(ndarray1, ndarray2)
except ImportError:
    pass
