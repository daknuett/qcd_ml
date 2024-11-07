import math
from qcd_ml.util.tensor import get_permutation_sign, levi_civita_index_and_sign_iterator

def test_levi_civita_iterator():
    for nd in range(2, 8):
        epsilon = list(levi_civita_index_and_sign_iterator(nd))
        assert len(epsilon) == math.factorial(nd)

        for idx, sgn in epsilon:
            assert get_permutation_sign(idx) == sgn


