#!/usr/bin/env python

from geobipy import RectilinearMesh2D_stitched
from geobipy import RectilinearMesh1D
from geobipy import Model
from geobipy import CircularLoops
from geobipy import Distribution
import numpy as np
import matplotlib.pyplot as plt

from geobipy import StatArray

np.random.seed(0)

data_path = '..//supplementary//data'

def create_model(model_type):

    n_points = 79
    zwedge = np.linspace(50.0, 1.0, n_points)
    zdeep = np.linspace(75.0, 500.0, n_points)

    resistivities = {'glacial' : np.r_[100, 10, 30],   # Glacial sediments, sands and tills
                     'saline_clay' : np.r_[100, 10, 1],    # Easier bottom target, uncommon until high salinity clay is 5-10 ish
                     'resistive_dolomites' : np.r_[50, 500, 50],   # Glacial sediments, resistive dolomites, marine shale.
                     'resistive_basement' : np.r_[100, 10, 10000],# Resistive Basement
                     'coastal_salt_water' : np.r_[1, 100, 20],    # Coastal salt water upper layer
                     'ice_over_salt_water' : np.r_[10000, 100, 1] # Antarctica glacier ice over salt water
    }
    conductivities = {'glacial' : np.r_[0.01, 0.1, 0.03333333],   # Glacial sediments, sands and tills
                      'saline_clay' : np.r_[0.01, 0.1, 1.  ],    # Easier bottom target, uncommon until high salinity clay is 5-10 ish
                      'resistive_dolomites' : np.r_[0.02, 0.002, 0.02 ],   # Glacial sediments, resistive dolomites, marine shale.
                      'resistive_basement' : np.r_[0.01, 0.1, 0.0001],# Resistive Basement
                      'coastal_salt_water' : np.r_[1., 0.01, 0.05],    # Coastal salt water upper layer
                      'ice_over_salt_water' : np.r_[1.e-04, 1.e-02, 1.e+00] # Antarctica glacier ice over salt water
    }

    conductivity = StatArray(conductivities[model_type], name="Conductivity", units='$\\frac{S}{m}$')
    print(conductivity)

    # # Create distributions for three lithology classes
    # lithology_distribution = Distribution('MvLogNormal',
    #                                    mean=conductivity,
    #                                    variance=[0.5,0.5,0.5],
    #                                    linearSpace=True)

    x = RectilinearMesh1D(centres=StatArray(np.arange(n_points, dtype=np.float64), name='x'))
    mesh = RectilinearMesh2D_stitched(3, x=x)
    mesh.nCells[:] = 3
    mesh.y_edges[:, 1] = zwedge
    mesh.y_edges[:, 2] = zdeep
    mesh.y_edges[:, 3] = np.inf
    mesh.y_edges.name, mesh.y_edges.units = 'Depth', 'm'

    return Model(mesh=mesh, values=np.repeat(conductivity[None, :], n_points, 0))

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

    d = ds.datapoint(99); plt.subplot(splt[0, 2]); d.plot();

    Path(data_path+'//figures').mkdir(parents=True, exist_ok=True)
    plt.savefig(data_path+'//figures//'+title+'.png');

def create_resolve(model, output_suffix):
    from geobipy import FdemData

    title = 'resolve_'+output_suffix

    ds = FdemData(system=data_path+'//FdemSystem2.stm')

    ds.x = np.arange(model.x.nCells, dtype=np.float64)
    ds.y = np.zeros(model.x.nCells)
    ds.z = np.full(model.x.nCells, fill_value=30.0)
    ds.elevation = np.zeros(model.x.nCells)
    ds.relative_error = np.full((model.x.nCells, 1), fill_value = 0.05)
    ds.additive_error = np.full((model.x.nCells, 1), fill_value = 5)

    dp = ds.datapoint(0)

    model.mesh.y_edges = model.mesh.y_edges / 4.1

    for k in range(model.x.nCells):
        mod = model[k]
        dp.forward(mod)
        ds.data[k, :] = dp.predictedData

    ds.write_csv(data_path+'//{}_clean.csv'.format(title))

    ds.data += (np.random.randn(model.x.nCells, ds.nChannels) * ds.std)

    make_figure(ds, model, title)

    ds.write_csv(data_path+'//{}.csv'.format(title))

    # ds = FdemData.read_csv('data/{}.csv'.format(title), '../../documentation_source/source/examples/supplementary/data/FdemSystem2.stm')

def create_skytem(model, output_suffix, system):
    from geobipy import TdemData, CircularLoops

    title = 'skytem_{}_'.format(system) + output_suffix

    ds = TdemData(system=[data_path+'//SkytemHM_{}.stm'.format(system), data_path+'//SkytemLM_{}.stm'.format(system)])

    ds.x = np.arange(model.x.nCells, dtype=np.float64)
    ds.y = np.zeros(model.x.nCells)
    ds.z = np.full(model.x.nCells, fill_value=30.0)
    ds.elevation = np.zeros(model.x.nCells)

    ds.loop_pair.transmitter = CircularLoops(x=ds.x, y=ds.y, z=ds.z,
                    #  pitch=0.0, roll=0.0, yaw=0.0,
                    radius=np.full(model.x.nCells, ds.system[0].loopRadius()))

    ds.loop_pair.receiver = CircularLoops(x=ds.transmitter.x -13.0,
                    y=ds.transmitter.y + 0.0,
                    z=ds.transmitter.z + 2.0,
                    #  pitch=0.0, roll=0.0, yaw=0.0,
                    radius=np.full(model.x.nCells, ds.system[0].loopRadius()))

    ds.relative_error = np.full((model.x.nCells, 2), fill_value = 0.03)
    ds.additive_error = np.full((model.x.nCells, 2), fill_value = 1e-15)
    ds.additive_error[:, 1] = 1e-14

    dp = ds.datapoint(0)

    for k in range(model.x.nCells):
        mod = model[k]

        dp.forward(mod)
        ds.secondary_field[k, :] = dp.predictedData

    ds.write_csv(data_path+'//{}_clean.csv'.format(title))

    ds.secondary_field += (np.random.randn(model.x.nCells, ds.nChannels) * ds.std)

    make_figure(ds, model, title)

    ds.write_csv(data_path+'//{}.csv'.format(title))

    # from geobipy import TdemData
    # ds = TdemData.read_csv('data/{}.csv'.format(title),
    #                     ['data/SkytemHM-SLV.stm',
    #                     'data/SkytemLM-SLV.stm'])

#%%
def create_aerotem(model, output_suffix):
    from geobipy import TdemData

    title = 'aerotem_'+output_suffix

    ds = TdemData(system=[data_path+'//aerotem.stm'])
    ds.x = np.arange(model.x.nCells, dtype=np.float64)
    ds.y = np.zeros(model.x.nCells)
    ds.z = np.full(model.x.nCells, fill_value=30.0)
    ds.elevation = np.zeros(model.x.nCells)

    ds.transmitter = CircularLoops(x=ds.x, y=ds.y, z=ds.z,
                    #  pitch=0.0, roll=0.0, yaw=0.0,
                    radius=np.full(model.x.nCells, fill_value=ds.system[0].loopRadius()))

    ds.receiver = CircularLoops(x=ds.transmitter.x + 2.0,
                    y=ds.transmitter.y + 2.0,
                    z=ds.transmitter.z - 12.0,
                    #  pitch=0.0, roll=0.0, yaw=0.0,
                    radius=np.full(model.x.nCells, fill_value=ds.system[0].loopRadius()))

    ds.relative_error = np.full((model.x.nCells, 1), fill_value = 0.03)
    ds.additive_error = np.full((model.x.nCells, 1), fill_value = 1e-9)

    dp = ds.datapoint(0)

    for k in range(model.x.nCells):
        mod = model[k]

        dp.forward(mod)
        ds.secondary_field[k, :] = dp.predicted_secondary_field

    ds.write_csv(data_path+'//{}_clean.csv'.format(title))

    ds.secondary_field += (np.random.randn(model.x.nCells, ds.nChannels) * ds.std)

    make_figure(ds, model, title)

    ds.write_csv(data_path+'//{}.csv'.format(title))

    # TdemData.read_csv('data/aerotem.csv', 'data/aerotem.stm')
#%%
def create_tempest(model, output_suffix):
    from geobipy import TempestData

    title = 'tempest_'+output_suffix

    ds = TempestData(system=[data_path+'//Tempest.stm'])

    ds.x = np.arange(model.x.nCells, dtype=np.float64)
    ds.y = np.zeros(model.x.nCells)
    ds.z = np.full(model.x.nCells, fill_value = 120.0)
    ds.elevation = np.zeros(model.x.nCells)
    ds.fiducial = np.arange(model.x.nCells)

    ds.loop_pair.transmitter = CircularLoops(
                    x = ds.x, y = ds.y, z = ds.z,
                    pitch = np.zeros(model.x.nCells), #np.random.uniform(low=-1.0, high=1.0, size=model.x.nCells),
                    roll  = np.zeros(model.x.nCells), #np.random.uniform(low=-1.0, high=1.0, size=model.x.nCells),
                    yaw   = np.zeros(model.x.nCells), #np.random.uniform(low=-1.0, high=1.0, size=model.x.nCells),
                    radius = np.full(model.x.nCells, fill_value=ds.system[0].loopRadius()))

    ds.loop_pair.receiver = CircularLoops(
                    x = ds.transmitter.x - 107.0,
                    y = ds.transmitter.y + 0.0,
                    z = ds.transmitter.z - 45.0,
                    pitch = np.zeros(model.x.nCells), #np.random.uniform(low=-0.5, high=0.5, size=model.x.nCells),
                    roll  = np.zeros(model.x.nCells), #np.random.uniform(low=-0.5, high=0.5, size=model.x.nCells),
                    yaw   = np.zeros(model.x.nCells), #np.random.uniform(low=-0.5, high=0.5, size=model.x.nCells),
                    radius = np.full(model.x.nCells, fill_value=ds.system[0].loopRadius()))

    ds.relative_error = np.repeat(np.r_[0.001, 0.001][None, :], model.x.nCells, 0)
    add_error = np.r_[0.011474, 0.012810, 0.008507, 0.005154, 0.004742, 0.004477, 0.004168, 0.003539, 0.003352, 0.003213, 0.003161, 0.003122, 0.002587, 0.002038, 0.002201,
                    0.007383, 0.005693, 0.005178, 0.003659, 0.003426, 0.003046, 0.003095, 0.003247, 0.002775, 0.002627, 0.002460, 0.002178, 0.001754, 0.001405, 0.001283]
    ds.additive_error = np.repeat(add_error[None, :], model.x.nCells, 0)


    dp = ds.datapoint(0)

    for k in range(model.x.nCells):
        mod = model[k]

        print(mod.summary)
        dp.forward(mod)
        dp.secondary_field[:] = dp.predicted_secondary_field
        dp.primary_field[:] = dp.predicted_primary_field

        ds.primary_field[k, :] = dp.primary_field
        ds.secondary_field[k, :] = dp.secondary_field

    ds.write_csv(data_path+'//{}_clean.csv'.format(title))

    # Add noise to various solvable parameters

    # ds.z += np.random.uniform(low=-5.0, high=5.0, size=model.x.nCells)
    # ds.receiver.x += np.random.normal(loc=0.0, scale=0.25**2.0, size=model.x.nCells)
    # ds.receiver.z += np.random.normal(loc = 0.0, scale = 0.25**2.0, size=model.x.nCells)
    # ds.receiver.pitch += np.random.normal(loc = 0.0, scale = 0.25**2.0, size=model.x.nCells)
    # ds.receiver.roll += np.random.normal(loc = 0.0, scale = 0.5**2.0, size=model.x.nCells)
    # ds.receiver.yaw += np.random.normal(loc = 0.0, scale = 0.5**2.0, size=model.x.nCells)

    ds.secondary_field += (np.random.randn(model.x.nCells, ds.nChannels) * ds.std)
    ds.write_csv(data_path+'//{}.csv'.format(title))

    make_figure(ds, model, title)

    # from geobipy import TempestData
    # ds = TempestData.read_csv('../../documentation_source/source/examples/supplementary/data/Tempest.csv', '../../documentation_source/source/examples/supplementary/data/Tempest.stm')


if __name__ == '__main__':

    np.random.seed(0)

    keys = ['glacial', 'saline_clay', 'resistive_dolomites', 'resistive_basement', 'coastal_salt_water', 'ice_over_salt_water']

    for k in keys:
        wedge_model = create_model(k)

        # create_resolve(wedge_model, k)
        # create_skytem(wedge_model, k, 512)
        # create_skytem(wedge_model, k, 304)
        # create_aerotem(wedge_model, k)
        create_tempest(wedge_model, k)