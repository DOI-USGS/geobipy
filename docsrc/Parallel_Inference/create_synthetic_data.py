#!/usr/bin/env python

from geobipy import Model
from geobipy import get_prng

import numpy as np
import matplotlib.pyplot as plt

from geobipy import StatArray

data_path = '..//..//examples//data//data'

def make_figure(ds, ds_noisy, model, title):
    from pathlib import Path
    fig = plt.figure(figsize=(20, 12));
    plt.suptitle(title)

    splt = fig.add_gridspec(2, 3, width_ratios=[1, 3, 1], wspace=0.3);
    ax = plt.subplot(splt[0, 0])
    dn = ds_noisy.datapoint(0); dn.plot()
    d = ds.datapoint(0); d.plot(with_error_bars=False, linestyle='solid', marker=None);

    ax1 = plt.subplot(splt[0, 1], sharey=ax);
    ds_noisy.plot_data();
    ax1.get_legend().remove();
    ax2 = plt.subplot(splt[1, 1], sharex=ax1);
    model.pcolor(log=10);
    ax2.sharex(ax1)

    plt.subplot(splt[0, 2], sharey=ax);
    dn = ds_noisy.datapoint(69); dn.plot()
    d = ds.datapoint(69); d.plot(with_error_bars=False, linestyle='solid', marker=None);

    Path(data_path+'//figures').mkdir(parents=True, exist_ok=True)
    plt.savefig(data_path+'//figures//'+title+'.png');

def create_resolve(model):
    from geobipy import FdemData

    title = 'resolve_'+ model
    model = Model.create_synthetic_model(model, left_thickness=np.r_[25.0, 5.0], right_thickness=np.r_[1.0, 69])

    prng = get_prng(seed=0)

    ds = FdemData(system=data_path+'//resolve.stm')
    ds, ds_noisy = ds.create_synthetic_data(model, prng)
    ds.write_csv(data_path+'//{}_clean.csv'.format(title))
    ds_noisy.write_csv(data_path+'//{}.csv'.format(title))

    make_figure(ds, ds_noisy, model, title)

def create_skytem(model):
    from geobipy import TdemData

    title = 'skytem_' + model

    model = Model.create_synthetic_model(model)

    prng = get_prng(seed=0)

    ds = TdemData(system=[data_path+'//SkytemHM.stm', data_path+'//SkytemLM.stm'])
    ds, ds_noisy = ds.create_synthetic_data(model, prng)

    ds.write_csv(data_path+'//{}_clean.csv'.format(title))
    ds_noisy.write_csv(data_path+'//{}.csv'.format(title))

    make_figure(ds, ds_noisy, model, title)

#%%
def create_tempest(model):
    from geobipy import TempestData

    title = 'tempest_'+ model

    model = Model.create_synthetic_model(model)

    prng = get_prng(seed=0)

    ds = TempestData(system=[data_path+'//tempest.stm'])

    ds, ds_noisy = ds.create_synthetic_data(model, prng)

    ds.write_csv(data_path+'//{}_clean.csv'.format(title))
    ds_noisy.write_csv(data_path+'//{}.csv'.format(title))

    make_figure(ds, ds_noisy, model, title)

if __name__ == '__main__':
    models = ['glacial',
              'offshore_fresh_discharge',
              'saline_clay',
              'resistive_dolomites',
              'resistive_basement',
              'coastal_salt_water',
              'ice_over_salt_water',
              'water_into_basalt']

    for model in models:
        # create_resolve(model)
        create_skytem(model)
        # create_tempest(model)
