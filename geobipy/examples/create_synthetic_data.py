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

# import warnings
# warnings.filterwarnings('error')

n_points = 100

zwedge = np.linspace(10.0, 1.0, n_points)
zdeep = np.linspace(50.0, 200.0, n_points)

conductivity = StatArray([0.005, 0.05, 1.0], name="Conductivity", units='$\\frac{S}{m}$')

# Create distributions for three lithology classes
lithology_distribution = Distribution('MvLogNormal',
                                       mean=conductivity,
                                       variance=[0.5,0.5,0.5],
                                       linearSpace=True)

x = RectilinearMesh1D(centres=StatArray(np.arange(n_points, dtype=np.float64), name='x'))
mesh = RectilinearMesh2D_stitched(3, x=x)
mesh.nCells[:] = 3
mesh.y_edges[:, 1] = zwedge
mesh.y_edges[:, 2] = zdeep
mesh.y_edges[:, 3] = np.inf
mesh.y_edges.name, mesh.y_edges.units = 'Depth', 'm'

wedge_model = Model(mesh=mesh, values=np.repeat(lithology_distribution.mean[None, :], n_points, 0))

plt.figure()
wedge_model.pcolor(flipY=True)
plt.savefig('Wedge.png')

def create_resolve():
    from geobipy import FdemData, FdemSystem

    ds = FdemData(system='../../documentation_source/source/examples/supplementary/data/FdemSystem2.stm')

    ds.x = np.arange(n_points, dtype=np.float64)
    ds.y = np.zeros(n_points)
    ds.z = np.full(n_points, fill_value=30.0)
    ds.elevation = np.zeros(n_points)
    ds.relative_error = np.full((n_points, 1), fill_value = 0.05)
    ds.additive_error = np.full((n_points, 1), fill_value = 5)

    dp = ds.datapoint(0)

    dp.relative_error = 0.05
    dp.additive_error = 5.0

    for k in range(n_points):
        mod = wedge_model[k]

        dp.forward(mod)
        dp.data = dp.predictedData

        dp.data += dp.std * np.random.randn(dp.nChannels)

        ds.data[k, :] = dp.data

    plt.figure(constrained_layout=True)
    plt.suptitle('Resolve')
    ax = plt.subplot(211)
    ds.data[:, :6].plot(x=ds.x)
    ds.data[:, 6:].plot(x=ds.x, linestyle='-.')
    # ax.get_legend().remove()
    plt.subplot(212, sharex=ax)
    wedge_model.pcolor(flipY=True, log=10)

    ds.write_csv('../../documentation_source/source/examples/supplementary/data/resolve.csv')

    ds = FdemData.read_csv('../../documentation_source/source/examples/supplementary/data/resolve.csv', '../../documentation_source/source/examples/supplementary/data/FdemSystem2.stm')

def create_skytem():
    from geobipy import TdemData, TdemSystem, CircularLoops

    n_points = 100

    ds = TdemData(system=['../../documentation_source/source/examples/supplementary/data/SkytemHM-SLV.stm',
                        '../../documentation_source/source/examples/supplementary/data/SkytemLM-SLV.stm'])

    ds.x = np.arange(n_points, dtype=np.float64)
    ds.y = np.zeros(n_points)
    ds.z = np.full(n_points, fill_value=30.0)
    ds.elevation = np.zeros(n_points)

    ds.transmitter = CircularLoops(x=ds.x, y=ds.y, z=ds.z,
                    #  pitch=0.0, roll=0.0, yaw=0.0,
                    radius=np.full(n_points, ds.system[0].loopRadius()))

    ds.receiver = CircularLoops(x=ds.transmitter.x -13.0,
                    y=ds.transmitter.y + 0.0,
                    z=ds.transmitter.z + 2.0,
                    #  pitch=0.0, roll=0.0, yaw=0.0,
                    radius=np.full(n_points, ds.system[0].loopRadius()))

    ds.relative_error = np.full((n_points, 2), fill_value = 0.05)
    ds.additive_error = np.full((n_points, 2), fill_value = 1e-12)
    ds.additive_error[:, 1] = 1e-11

    dp = ds.datapoint(0)

    for k in range(n_points):
        mod = wedge_model[k]

        dp.forward(mod)
        dp.secondary_field = dp.predictedData

        dp.secondary_field += dp.std * np.random.randn(dp.nChannels)

        ds.secondary_field[k, :] = dp.data

    plt.figure(constrained_layout=True);
    plt.suptitle('Skytem')
    ax = plt.subplot(211);
    ds.plot();
    ax.get_legend().remove();
    plt.subplot(212, sharex=ax);
    wedge_model.pcolor(flipY=True, log=10)

    d0 = ds.datapoint(0)
    d1 = ds.datapoint(99)
    plt.figure()
    plt.suptitle('Skytem')
    plt.subplot(121)
    d0.plot()
    plt.subplot(122)
    d1.plot()

    plt.savefig('Skytem.png');

    ds.write_csv('../../documentation_source/source/examples/supplementary/data/skytem.csv')

    from geobipy import TdemData
    ds = TdemData.read_csv('../../documentation_source/source/examples/supplementary/data/skytem.csv',
                        ['../../documentation_source/source/examples/supplementary/data/SkytemHM-SLV.stm',
                        '../../documentation_source/source/examples/supplementary/data/SkytemLM-SLV.stm'])

def create_aerotem():
    from geobipy import TdemData, TdemSystem, CircularLoop

    n_points = 100

    ds = TdemData(system=['../../documentation_source/source/examples/supplementary/data/aerotem.stm'])
    ds.x = np.arange(n_points, dtype=np.float64)
    ds.y = np.zeros(n_points)
    ds.z = np.full(n_points, fill_value=30.0)
    ds.elevation = np.zeros(n_points)

    ds.transmitter = CircularLoops(x=ds.x, y=ds.y, z=ds.z,
                    #  pitch=0.0, roll=0.0, yaw=0.0,
                    radius=np.full(n_points, fill_value=ds.system[0].loopRadius()))

    ds.receiver = CircularLoops(x=ds.transmitter.x + 2.0,
                    y=ds.transmitter.y + 2.0,
                    z=ds.transmitter.z - 12.0,
                    #  pitch=0.0, roll=0.0, yaw=0.0,
                    radius=np.full(n_points, fill_value=ds.system[0].loopRadius()))

    ds.relative_error = np.full((n_points, 1), fill_value = 0.05)
    ds.additive_error = np.full((n_points, 1), fill_value = 1e-8)

    dp = ds.datapoint(0)

    for k in range(n_points):
        mod = wedge_model[k]

        dp.forward(mod)
        dp.secondary_field = dp.predicted_secondary_field

        dp.secondary_field += dp.std * np.random.randn(dp.nChannels)

        ds.secondary_field[k, :] = dp.secondary_field

    plt.figure(constrained_layout=True)
    plt.suptitle('Aerotem')
    ax = plt.subplot(211)
    ds.plot()
    ax.get_legend().remove()
    plt.subplot(212, sharex=ax)
    wedge_model.pcolor(flipY=True, log=10)

    d0 = ds.datapoint(0)
    d1 = ds.datapoint(99)
    plt.figure()
    plt.suptitle('Aerotem')
    plt.subplot(121)
    d0.plot()
    plt.subplot(122)
    d1.plot()

    ds.write_csv('../../documentation_source/source/examples/supplementary/data/aerotem.csv')

    TdemData.read_csv('../../documentation_source/source/examples/supplementary/data/aerotem.csv', '../../documentation_source/source/examples/supplementary/data/aerotem.stm')

def create_tempest():
    from geobipy import TempestData, TdemSystem, CircularLoop

    n_points = 100

    ds = TempestData(system=['../../documentation_source/source/examples/supplementary/data/Tempest.stm'])
    ds.x = np.arange(n_points, dtype=np.float64)
    ds.y = np.zeros(n_points)
    ds.z = np.full(n_points, fill_value = 120.0)
    ds.elevation = np.zeros(n_points)
    ds.fiducial = np.arange(n_points)

    ds.transmitter = CircularLoops(x=ds.x, y=ds.y, z=ds.z,
                    pitch = np.random.uniform(low=-10.0, high=10.0, size=n_points),
                    roll = np.random.uniform(low=-25.0, high=25.0, size=n_points),
                    yaw = np.random.uniform(low=-15.0, high=15.0, size=n_points),
                    radius=np.full(n_points, fill_value=ds.system[0].loopRadius()))

    ds.receiver = CircularLoops(x=ds.transmitter.x - 107.0,
                    y=ds.transmitter.y + 0.0,
                    z=ds.transmitter.z - 45.0,
                    pitch = np.random.uniform(low=-5.0, high=5.0, size=n_points),
                    roll = np.random.uniform(low=-10.0, high=10.0, size=n_points),
                    yaw = np.random.uniform(low=-5.0, high=5.0, size=n_points),
                    radius=np.full(n_points, fill_value=ds.system[0].loopRadius()))

    ds.relative_error = np.repeat(np.r_[0.05, 0.05][None, :], n_points, 0)
    add_error = np.r_[0.011474, 0.012810, 0.008507, 0.005154, 0.004742, 0.004477, 0.004168, 0.003539, 0.003352, 0.003213, 0.003161, 0.003122, 0.002587, 0.002038, 0.002201,
                    0.007383, 0.005693, 0.005178, 0.003659, 0.003426, 0.003046, 0.003095, 0.003247, 0.002775, 0.002627, 0.002460, 0.002178, 0.001754, 0.001405, 0.001283]
    ds.additive_error = np.repeat(add_error[None, :], n_points, 0)


    dp = ds.datapoint(0)

    for k in range(n_points):
        mod = wedge_model[k]

        dp.forward(mod)
        dp.secondary_field[:] = dp.predicted_secondary_field
        dp.primary_field[:] = dp.predicted_primary_field

        ds.primary_field[k, :] = dp.primary_field
        ds.secondary_field[k, :] = dp.secondary_field

    ds.write_csv('../../documentation_source/source/examples/supplementary/data/Tempest_no_noise.csv')

    # Add noise to various solvable parameters

    # ds.z += np.random.uniform(low=-5.0, high=5.0, size=n_points)
    ds.receiver.x += np.random.normal(loc=0.0, scale=1.0**2.0, size=n_points)
    ds.receiver.z += np.random.normal(loc = 0.0, scale = 1.0**2.0, size=n_points)
    ds.receiver.pitch += np.random.normal(loc = 0.0, scale = 1.5**2.0, size=n_points)
    # ds.receiver.roll += np.random.normal(loc = 0.0, scale = 0.5**2.0, size=n_points)
    # ds.receiver.yaw += np.random.normal(loc = 0.0, scale = 0.5**2.0, size=n_points)

    ds.secondary_field += (np.random.randn(n_points, dp.nChannels) * ds.std)
    ds.write_csv('../../documentation_source/source/examples/supplementary/data/Tempest.csv')


    plt.figure(constrained_layout=True)
    plt.suptitle('Tempest')
    ax = plt.subplot(211)
    ds.plot(legend=False)
    plt.subplot(212, sharex=ax)
    wedge_model.pcolor(flipY=True, log=10)

    d0 = ds.datapoint(0)
    d1 = ds.datapoint(99)
    plt.figure()
    plt.suptitle('Tempest')
    plt.subplot(121)
    d0.plot()
    plt.subplot(122)
    d1.plot()



    from geobipy import TempestData
    ds = TempestData.read_csv('../../documentation_source/source/examples/supplementary/data/Tempest.csv', '../../documentation_source/source/examples/supplementary/data/Tempest.stm')


# create_resolve()
# create_skytem()
# create_aerotem()
create_tempest()

# plt.show()