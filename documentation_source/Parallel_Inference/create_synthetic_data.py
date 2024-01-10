#!/usr/bin/env python

from geobipy import RectilinearMesh2D_stitched
from geobipy import RectilinearMesh1D
from geobipy import Model
from geobipy import CircularLoops
from geobipy import Distribution
from geobipy import get_prng

import numpy as np
import matplotlib.pyplot as plt

from geobipy import StatArray

data_path = '..//source//supplementary//data//'

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
    model.pcolor(flipY=True, log=10);
    ax1.sharex(ax)

    d = ds.datapoint(69); plt.subplot(splt[0, 2]); d.plot();

    Path(data_path+'//figures').mkdir(parents=True, exist_ok=True)
    plt.savefig(data_path+'//figures//'+title+'.png');

def create_resolve(model, output_suffix):
    from geobipy import FdemData

    title = 'resolve_'+output_suffix
    prng = get_prng(seed=0)

    ds = FdemData(system=data_path+'//resolve.stm')
    ds_clean, ds_noisy = ds.create_synthetic_data(model, prng)
    ds_clean.write_csv(data_path+'//{}_clean.csv'.format(title))
    ds_noisy.write_csv(data_path+'//{}.csv'.format(title))

    make_figure(ds_clean, model, title)

def create_skytem(model, output_suffix, system):
    from geobipy import TdemData, CircularLoops

    title = 'skytem_{}_'.format(system) + output_suffix
    prng = get_prng(seed=0)

    ds = TdemData(system=[data_path+'//SkytemHM_{}.stm'.format(system), data_path+'//SkytemLM_{}.stm'.format(system)])
    ds_clean, ds_noisy = ds.create_synthetic_data(model, prng)

    ds.write_csv(data_path+'//{}_clean.csv'.format(title))
    ds.write_csv(data_path+'//{}.csv'.format(title))

    make_figure(ds_clean, model, title)

#%%
def create_tempest(model, output_suffix):
    from geobipy import TempestData

    title = 'tempest_'+output_suffix
    prng = get_prng(seed=0)

    ds = TempestData(system=[data_path+'//Tempest.stm'])

    ds_clean, ds_noisy = ds.create_synthetic_data(model, prng)

    ds.write_csv(data_path+'//{}_clean.csv'.format(title))
    ds.write_csv(data_path+'//{}.csv'.format(title))

    make_figure(ds_clean, model, title)

if __name__ == '__main__':
    keys = ['glacial', 'saline_clay', 'resistive_dolomites', 'resistive_basement', 'coastal_salt_water', 'ice_over_salt_water']

    for k in keys:
        model = Model.create_synthetic_model(k)

        create_resolve(model, k)
        # create_skytem(model, k, 512)
        # create_tempest(model, k)
