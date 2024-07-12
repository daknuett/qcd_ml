import numpy as np
import pytest

try:
    import gpt as g
    from qcd_ml.compat.gpt import lattice2ndarray, ndarray2lattice


    def test_lattice2array2lattice():
        grid = g.grid([4, 4, 4, 8], g.double)

        rng = g.random("foo")
        psi = g.vspincolor(grid)

        rng.cnormal(psi)

        psi2 = ndarray2lattice(lattice2ndarray(psi), grid, g.vspincolor)

        assert g.norm2(psi2 - psi) < 1e-12

    def test_array2lattice2array():
        grid = g.grid([4, 4, 4, 8], g.double)

        rng = g.random("foo")
        psi = g.vspincolor(grid)

        rng.cnormal(psi)

        ndarray1 = lattice2ndarray(psi)
        ndarray2 = lattice2ndarray(ndarray2lattice(ndarray1, grid, g.vspincolor))

        assert np.allclose(ndarray1, ndarray2)
except ImportError:
    @pytest.mark.skip("missing gpt")
    def test_lattice2array2lattice():
        pass

    @pytest.mark.skip("missing gpt")
    def test_array2lattice2array():
        pass
