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

        assert g.norm2(psi2 - psi) / g.norm2(psi) < 1e-14


    def test_array2lattice2array():
        grid = g.grid([4, 4, 4, 8], g.double)

        rng = g.random("foo")
        psi = g.vspincolor(grid)

        rng.cnormal(psi)

        ndarray1 = lattice2ndarray(psi)
        ndarray2 = lattice2ndarray(ndarray2lattice(ndarray1, grid, g.vspincolor))

        assert np.allclose(ndarray1, ndarray2)
        assert np.linalg.norm(ndarray1 - ndarray2) / np.linalg.norm(ndarray1) < 1e-14


    def test_lattice2array():
        grid = g.grid([4, 4, 4, 8], g.double)
        value = np.array([[1, 2, 0], [3, 4 + 8j, 5 + 8j], [6, 7+ 3j, 8], [9, 10, 11 + 4j]])

        psi = g.vspincolor(grid)
        psi[:] = 0
        psi[1, 2, 3, 4] = g.vspincolor(value)

        ndarray = lattice2ndarray(psi)

        expect = np.zeros((4, 4, 4, 8, 4, 3), dtype=np.complex128)
        expect[1, 2, 3, 4] = value


        assert np.allclose(ndarray, expect)
        assert np.linalg.norm(ndarray - expect) / np.linalg.norm(expect) < 1e-14

except ImportError:
    @pytest.mark.skip("missing gpt")
    def test_lattice2array2lattice():
        pass

    @pytest.mark.skip("missing gpt")
    def test_array2lattice2array():
        pass


    @pytest.mark.skip("missing gpt")
    def test_lattice2array():
        pass
