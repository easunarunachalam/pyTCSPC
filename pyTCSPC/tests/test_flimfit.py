import numpy as np
import pytest


import sys
sys.path.append("../")

import pyTCSPC as pc


@pytest.fixture
def load_IRF():

    irf = pc.load_sdt(r"tests/data_flimfit/test_IRF.sdt", dims="CXM", dtype=np.uint32)

    dc_kwargs_M1 = {
        "trunc": True,
        "peak_start": 2.6,
        "peak_end": 3.1,
        "bgsub": True,
        "bg_start": 8,
        "bg_end": 10,
    }

    dc_kwargs_M2 = {
        "trunc": True,
        "peak_start": 2.25,
        "peak_end": 3.5,
        "bgsub": True,
        "bg_start": 8,
        "bg_end": 10,
    }

    return irf, dc_kwargs_M1, dc_kwargs_M2

def test_load_IRF(load_IRF):
    irf, dc_kwargs_M1, dc_kwargs_M2 = load_IRF
    dc_M1 = pc.decay_curve(irf.sel(channel="M1"), plot=False, **dc_kwargs_M1).compute()
    dc_M2 = pc.decay_curve(irf.sel(channel="M2"), plot=False, **dc_kwargs_M2).compute()

    dc_M1_ref = np.loadtxt("tests/data_flimfit/test_IRF_dc_M1.csv")
    dc_M2_ref = np.loadtxt("tests/data_flimfit/test_IRF_dc_M2.csv")

    assert np.allclose(dc_M1, dc_M1_ref, 1e-6)
    assert np.allclose(dc_M2, dc_M2_ref, 1e-6)

    pass

@pytest.fixture
def load_data():

    dc = np.loadtxt("tests/data_flimfit/test_data_decay_curve.csv")

    return dc


def test_1exp_model(load_IRF, load_data):
    irf, dc_kwargs_M1, dc_kwargs_M2 = load_IRF
    dc_data = load_data

    dg = pc.decay_group(dc_data, irf.sel(channel="M2"), irf_kws=dc_kwargs_M2, fit_start_bin=10, fit_end_bin=235)
    print(dg)
    fps = dg.fit(model="1exp", plot=True)
