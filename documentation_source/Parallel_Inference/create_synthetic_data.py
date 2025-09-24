#!/usr/bin/env python

from geobipy import Model
from geobipy import get_prng

import numpy as np
import matplotlib.pyplot as plt

from geobipy import StatArray

data_path = '..//source//supplementary//data'

def make_figure(ds, model, title):
    from pathlib import Path
    fig = plt.figure();
    plt.suptitle(title)
    splt = fig.add_gridspec(2, 3, width_ratios=[1, 3, 1], wspace=0.3);
    d = ds.datapoint(0); plt.subplot(splt[0, 0]); d.plot();

    ax = plt.subplot(splt[0, 1]);
    ds.plot_data();
    ax.get_legend().remove();
    ax1 = plt.subplot(splt[1, 1], sharex=ax);
    model.pcolor(log=10);
    ax1.sharex(ax)

    d = ds.datapoint(69); plt.subplot(splt[0, 2]); d.plot();

    Path(data_path+'//figures').mkdir(parents=True, exist_ok=True)
    plt.savefig(data_path+'//figures//'+title+'.png');

def create_resolve(model):
    from geobipy import FdemData

    title = 'resolve_'+ model
    model = Model.create_synthetic_model(model)

    prng = get_prng(seed=0)

    model.mesh.y_edges[:, 1] = -np.linspace(25.0, 1.0, model.mesh.x.nCells)
    model.mesh.y_edges[:, 2] = -np.linspace(30.0, 200.0, model.mesh.x.nCells)

    ds = FdemData(system=data_path+'//resolve.stm')
    ds, ds_noisy = ds.create_synthetic_data(model, prng)
    ds.write_csv(data_path+'//{}_clean.csv'.format(title))
    ds_noisy.write_csv(data_path+'//{}.csv'.format(title))

    make_figure(ds, model, title)

def create_skytem(model):
    from geobipy import TdemData

    title = 'skytem_' + model

    model = Model.create_synthetic_model(model)

    prng = get_prng(seed=0)

    ds = TdemData(system=[data_path+'//SkytemHM.stm', data_path+'//SkytemLM.stm'])
    ds, ds_noisy = ds.create_synthetic_data(model, prng)

    ds.write_csv(data_path+'//{}_clean.csv'.format(title))
    ds_noisy.write_csv(data_path+'//{}.csv'.format(title))

    make_figure(ds, model, title)

#%%
def create_tempest(model):
    from geobipy import TempestData

    title = 'tempest_'+ model

    model = Model.create_synthetic_model(model)

    prng = get_prng(seed=0)

    ds = TempestData(system=[data_path+'//Tempest.stm'])

    ds, ds_noisy = ds.create_synthetic_data(model, prng)

    ds.write_csv(data_path+'//{}_clean.csv'.format(title))
    ds_noisy.write_csv(data_path+'//{}.csv'.format(title))

    make_figure(ds, model, title)

if __name__ == '__main__':
    models = ['glacial', 'saline_clay', 'resistive_dolomites', 'resistive_basement', 'coastal_salt_water', 'ice_over_salt_water', 'water_into_basalt']

    for model in models:
        create_resolve(model)
        # create_skytem(model)
        # create_tempest(model)
