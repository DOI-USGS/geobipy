#!/usr/bin/env python

from geobipy import Model1D
import numpy as np
import matplotlib.pyplot as plt

n_points = 100

zwedge = np.linspace(10.0, 1.0, n_points)
zdeep = np.linspace(50.0, 200.0, n_points)

sigma = np.r_[0.005, 0.05, 0.2]

plt.figure()
plt.plot(np.arange(n_points), -zwedge, '.')
plt.plot(np.arange(n_points), -zdeep, '.')


#%%
from geobipy import FdemData, FdemSystem

ds = FdemData(system='../../documentation_source/source/examples/supplementary/Data/FdemSystem2.stm')

ds.x = np.arange(n_points, dtype=np.float64)
ds.y = np.zeros(n_points)
ds.z = np.full(n_points, fill_value=30.0)
ds.elevation = np.zeros(n_points)
ds.relative_error = np.full((n_points, 1), fill_value = 0.05)
ds.additive_error = np.full((n_points, 1), fill_value = 5)

dp = ds.datapoint(0)

dp.relErr = 0.05
dp.addErr = 5.0

for k in range(n_points):    
    thk = np.hstack([np.diff([0, zwedge[k], zdeep[k]]), np.inf])
    mod = Model1D(widths=thk, parameters=sigma)

    dp.forward(mod)
    dp.data = dp.predictedData
    
    dp.data += dp.std * np.random.randn(dp.nChannels)
    
    ds.data[k, :] = dp.data
    
plt.figure()
ax = plt.subplot(211)
ds.data[:, :6].plot(x=ds.x)
ds.data[:, 6:].plot(x=ds.x, linestyle='-.')
# ax.get_legend().remove()
plt.subplot(212, sharex=ax)
plt.hlines(0.0, ds.x[0], ds.x[-1], 'k')
plt.plot(ds.x, -zwedge, 'k')
plt.plot(ds.x, -zdeep, 'k')

#%%
ds.write_csv('../../documentation_source/source/examples/supplementary/Data//resolve.csv')

#%%
ds = FdemData.read_csv('../../documentation_source/source/examples/supplementary/Data/resolve.csv', '../../documentation_source/source/examples/supplementary/Data/FdemSystem2.stm')

#%%
from geobipy import TdemData, TdemSystem, CircularLoop

n_points = 100

ds = TdemData(system=['../../documentation_source/source/examples/supplementary/Data/SkytemHM-SLV.stm',
                      '../../documentation_source/source/examples/supplementary/Data/SkytemLM-SLV.stm'])
ds.x = np.arange(n_points, dtype=np.float64)
ds.y = np.zeros(n_points)
ds.z = np.full(n_points, fill_value=30.0)
ds.elevation = np.zeros(n_points)

T = CircularLoop(x=0.0, y=0.0, z=30.0,
                 pitch=0.0, roll=0.0, yaw=0.0,
                 radius=ds.system[0].loopRadius())
ds.transmitter = [T] * ds.nPoints

R = CircularLoop(x=T.x -13.0,
                 y=T.y + 0.0,
                 z=T.z + 2.0,
                 pitch=0.0, roll=0.0, yaw=0.0,
                 radius=ds.system[0].loopRadius())
ds.receiver = [R] * ds.nPoints

ds.relative_error = np.full((n_points, 2), fill_value = 0.05)
ds.additive_error = np.full((n_points, 2), fill_value = 1e-12)

dp = ds.datapoint(0)

dp.relErr = [0.05, 0.05]
dp.addErr = [1e-12, 1e-12]


for k in range(n_points):    
    thk = np.hstack([np.diff([0, zwedge[k], zdeep[k]]), np.inf])
    mod = Model1D(widths=thk, parameters=sigma)

    dp.forward(mod)
    dp.data = dp.predictedData
    
    dp.data += dp.std * np.random.randn(dp.nChannels)
    
    ds.secondary_field[k, :] = dp.data

plt.figure();
ax = plt.subplot(211);
ds.plot(log=10);
ax.get_legend().remove();
plt.subplot(212, sharex=ax);
plt.hlines(0.0, ds.x[0], ds.x[-1], 'k');
plt.plot(ds.x, -zwedge, 'k');
plt.plot(ds.x, -zdeep, 'k');

plt.savefig('Skytem.png');

#%%
ds.write_csv('../../documentation_source/source/examples/supplementary/Data/skytem.csv')

#%%
from geobipy import TdemData
ds = TdemData.read_csv('../../documentation_source/source/examples/supplementary/Data/skytem.csv', ['../../documentation_source/source/examples/supplementary/Data/SkytemHM-SLV.stm',
                      '../../documentation_source/source/examples/supplementary/Data/SkytemLM-SLV.stm'])

#%%
from geobipy import TdemData, TdemSystem, CircularLoop

n_points = 100

ds = TdemData(system=['../../documentation_source/source/examples/supplementary/Data/aerotem.stm'])
ds.x = np.arange(n_points, dtype=np.float64)
ds.y = np.zeros(n_points)
ds.z = np.full(n_points, fill_value=30.0)
ds.elevation = np.zeros(n_points)

T = CircularLoop(x=0.0, y=0.0, z=30.0,
                 pitch=0.0, roll=0.0, yaw=0.0,
                 radius=ds.system[0].loopRadius())
ds.transmitter = [T] * ds.nPoints

R = CircularLoop(x=T.x + 2.0,
                 y=T.y + 2.0,
                 z=T.z - 12.0,
                 pitch=0.0, roll=0.0, yaw=0.0,
                 radius=ds.system[0].loopRadius())
ds.receiver = [R] * ds.nPoints

ds.relative_error = np.full((n_points, 1), fill_value = 0.05)
ds.additive_error = np.full((n_points, 1), fill_value = 1e-7)

dp = ds.datapoint(0)

dp.relErr = 0.05
dp.addErr = 1e-7


for k in range(n_points):    
    thk = np.hstack([np.diff([0, zwedge[k], zdeep[k]]), np.inf])
    mod = Model1D(widths=thk, parameters=sigma)

    dp.forward(mod)
    dp.data = dp.predictedData
    
    dp.data += dp.std * np.random.randn(dp.nChannels)
    
    ds.data[k, :] = dp.data

plt.figure()
ax = plt.subplot(211)
ds.plot(log=10)
ax.get_legend().remove()
plt.subplot(212, sharex=ax)
plt.hlines(0.0, ds.x[0], ds.x[-1], 'k')
plt.plot(ds.x, -zwedge, 'k')
plt.plot(ds.x, -zdeep, 'k')

#%%
ds.write_csv('../../documentation_source/source/examples/supplementary/Data/aerotem.csv')

TdemData.read_csv('../../documentation_source/source/examples/supplementary/Data/aerotem.csv', '../../documentation_source/source/examples/supplementary/Data/aerotem.stm')

#%%
from geobipy import TempestData, TdemSystem, CircularLoop

n_points = 100

ds = TempestData(system=['../../documentation_source/source/examples/supplementary/Data/Tempest.stm'])
ds.x = np.arange(n_points, dtype=np.float64)
ds.y = np.zeros(n_points)
ds.z = np.full(n_points, fill_value=121.0)
ds.elevation = np.zeros(n_points)

T = CircularLoop(x=0.0, y=0.0, z=121.0,
                 pitch=0.0, roll=0.0, yaw=0.0,
                 radius=ds.system[0].loopRadius())
ds.transmitter = [T] * ds.nPoints


R = CircularLoop(x=T.x - 90.0,
                 y=T.y + 0.0,
                 z=T.z - 45.0,
                 pitch=0.0, roll=0.0, yaw=0.0,
                 radius=ds.system[0].loopRadius())
ds.receiver = [R] * ds.nPoints

ds.relative_error = np.full((n_points, 1), fill_value = 0.05)
ds.additive_error = np.full((n_points, 1), fill_value = 1e-1)

dp = ds.datapoint(0)
dp.relErr = [0.05, 0.05]
dp.addErr = [0.011474, 0.012810, 0.008507, 0.005154, 0.004742, 0.004477, 0.004168, 0.003539, 0.003352, 0.003213, 0.003161, 0.003122, 0.002587, 0.002038, 0.002201,
             0.007383, 0.005693, 0.005178, 0.003659, 0.003426, 0.003046, 0.003095, 0.003247, 0.002775, 0.002627, 0.002460, 0.002178, 0.001754, 0.001405, 0.001283]

for k in range(n_points):    
    edges = np.hstack([0, zwedge[k], zdeep[k], np.inf])
    mod = Model1D(edges=edges, parameters=sigma)

    dp.forward(mod)
    dp.primary_field = dp.predicted_primary_field
    dp.secondary_field = dp.predicted_secondary_field
    std = np.sqrt((0.05 * dp.secondary_field)**2 + dp.addErr**2)
    noise = (np.random.randn(dp.nChannels) * std)
    data = dp.data + noise

                      
    ds.primary_field[k, :] = dp.primary_field
    ds.secondary_field[k, :] = np.hstack([data[dp._component_indices(0, 0)] - dp.primary_field[0], data[dp._component_indices(1, 0)] - dp.primary_field[1]])
      
plt.figure()
ax = plt.subplot(311)
ds.secondary_field[:, [0, 7, 14]].plot()
ax = plt.subplot(312)
ds.secondary_field[:, [15, 22, 29]].plot()
plt.subplot(313, sharex=ax)
plt.hlines(0.0, ds.x[0], ds.x[-1], 'k')
plt.plot(ds.x, -zwedge, 'k')
plt.plot(ds.x, -zdeep, 'k')

#%%
ds.write_csv('../../documentation_source/source/examples/supplementary/Data/Tempest.csv')

#%%
from geobipy import TempestData
ds = TempestData.read_csv('../../documentation_source/source/examples/supplementary/Data/Tempest.csv', '../../documentation_source/source/examples/supplementary/Data/Tempest.stm')

