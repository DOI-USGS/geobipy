#!/usr/bin/env python

from geobipy import Model
from geobipy import FdemData, TdemData, TempestData
from geobipy import get_prng


import numpy as np
import matplotlib.pyplot as plt

from geobipy import StatArray

data_path = '..//documentation_source//source//supplementary//data//'


def test_resolve(model_type):

    model = Model.create_synthetic_model(model_type)

    title = 'resolve_'+ model_type
    prng = get_prng(seed=0)

    model.mesh.y_edges = model.mesh.y_edges / 10.0

    ds = FdemData(system=data_path+'resolve.stm')
    ds, _ = ds.create_synthetic_data(model, prng)

    ds_check = FdemData.read_csv("data_checks/{}_clean.csv".format(title), system=ds.system)

    assert np.allclose(ds.data, ds_check.data), ValueError("{} doesn't match".format(title))

def test_skytem(model_type):
    from geobipy import TdemData

    model = Model.create_synthetic_model(model_type)

    title = 'skytem_' + model_type

    prng = get_prng(seed=0)

    ds = TdemData(system=[data_path+'SkytemHM.stm', data_path+'SkytemLM.stm'])
    ds, _ = ds.create_synthetic_data(model, prng)

    clean_data = 'data_checks//{}_clean.csv'.format(title)
    ds_check = TdemData.read_csv(clean_data,
                                 system=ds.system)

    assert np.allclose(ds.data, ds_check.data), ValueError("{} doesn't match".format(title))

#%%
def test_tempest(model_type):
    from geobipy import TempestData

    model = Model.create_synthetic_model(model_type)

    title = 'tempest_'+ model_type
    prng = get_prng(seed=0)

    ds = TempestData(system=[data_path+'tempest.stm'])
    ds, _ = ds.create_synthetic_data(model, prng)

    ds_check = TempestData.read_csv("data_checks//{}_clean.csv".format(title),
                                    ds.system)

    assert np.allclose(ds.data, ds_check.data), ValueError("{} doesn't match".format(title))

if __name__ == '__main__':
    models = ['glacial', 'saline_clay', 'resistive_dolomites', 'resistive_basement', 'coastal_salt_water', 'ice_over_salt_water']

    for model in models:
        test_resolve(model)
        test_skytem(model)
        test_tempest(model)
