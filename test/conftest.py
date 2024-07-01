import pytest
import torch
import numpy as np

import os

def pytest_addoption(parser):
    parser.addoption(
            "--runslow"
            , action="store_true"
            , default=False
            , help="run slow tests"
            )
    parser.addoption(
            "--onlyselected"
            , action="store_true"
            , default=False
            , help="run only selected tests"
            )


def pytest_collection_modifyitems(config, items):
    if(config.getoption("--runslow")
            and config.getoption("--onlyselected")):
        return

    if(not config.getoption("--onlyselected")):
        skip_slow = pytest.mark.skip(reason="slow test")
        run_slow = config.getoption("--runslow")
        for item in items:
            if not run_slow:
                if "slow" in item.keywords:
                    item.add_marker(skip_slow)
    else:
        skip_not_selected = pytest.mark.skip(reason="not selected")
        for item in items:
            if not "selected" in item.keywords:
                item.add_marker(skip_not_selected)


@pytest.fixture 
def config_1500():
    return torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "assets","1500.config.npy")))


@pytest.fixture 
def config_1500_adj():
    return torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "assets","1500_adj.npy")))


@pytest.fixture 
def config_1200():
    return torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "assets","1200.config.npy")))


@pytest.fixture 
def psi_test():
    return torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "assets","psi_test.npy")))


@pytest.fixture 
def psi_1500mu0_psitest():
    return torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "assets","psi_1500mu0_psitest.npy")))


@pytest.fixture 
def config_1500_gtrans_1200mu0():
    return torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "assets","1500_gtrans_1200mu0.npy")))


@pytest.fixture 
def config_1500_gtrans_1200mu0():
    return torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "assets","1500_gtrans_1200mu0.npy")))


@pytest.fixture 
def V_1500mu0_1500mu2():
    return torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "assets","V_1500mu0_1500mu2.npy")))


@pytest.fixture 
def psi_Dw1500_m0p5_psitest():
    return torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "assets","psi_Dw1500_m0p5_psitest.npy")))
@pytest.fixture 
def psi_Dwc1500_m0p5_psitest():
    return torch.tensor(np.load(os.path.join(os.path.dirname(__file__), "assets","psi_Dwc1500_m0p5_psitest.npy")))
