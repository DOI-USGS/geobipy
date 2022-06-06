#!/usr/bin/env python

from geobipy import RectilinearMesh1D
from geobipy import Model
from geobipy import CircularLoops
import numpy as np
import matplotlib.pyplot as plt

# import warnings
# warnings.filterwarnings('error')

n_points = 100

zwedge = np.linspace(10.0, 1.0, n_points)
zdeep = np.linspace(50.0, 200.0, n_points)

sigma = np.r_[0.005, 0.05, 0.2]

# plt.figure()
# plt.plot(np.arange(n_points), -zwedge, '.')
# plt.plot(np.arange(n_points), -zdeep, '.')


# #%%
# from geobipy import FdemData, FdemSystem

# ds = FdemData(system='../../documentation_source/source/examples/supplementary/data/FdemSystem2.stm')

# ds.x = np.arange(n_points, dtype=np.float64)
# ds.y = np.zeros(n_points)
# ds.z = np.full(n_points, fill_value=30.0)
# ds.elevation = np.zeros(n_points)
# ds.relative_error = np.full((n_points, 1), fill_value = 0.05)
# ds.additive_error = np.full((n_points, 1), fill_value = 5)

# dp = ds.datapoint(0)

# dp.relErr = 0.05
# dp.addErr = 5.0

# for k in range(n_points):    
#     thk = np.hstack([np.diff([0, zwedge[k], zdeep[k]]), np.inf])
#     mod = Model(RectilinearMesh1D(widths=thk), values=sigma)

#     dp.forward(mod)
#     dp.data = dp.predictedData
    
#     dp.data += dp.std * np.random.randn(dp.nChannels)
    
#     ds.data[k, :] = dp.data
    
# plt.figure()
# ax = plt.subplot(211)
# ds.data[:, :6].plot(x=ds.x)
# ds.data[:, 6:].plot(x=ds.x, linestyle='-.')
# # ax.get_legend().remove()
# plt.subplot(212, sharex=ax)
# plt.hlines(0.0, ds.x[0], ds.x[-1], 'k')
# plt.plot(ds.x, -zwedge, 'k')
# plt.plot(ds.x, -zdeep, 'k')

# #%%
# ds.write_csv('../../documentation_source/source/examples/supplementary/data/resolve.csv')

# #%%
# ds = FdemData.read_csv('../../documentation_source/source/examples/supplementary/data/resolve.csv', '../../documentation_source/source/examples/supplementary/data/FdemSystem2.stm')

# #%%
# from geobipy import TdemData, TdemSystem, CircularLoops

# n_points = 100

# ds = TdemData(system=['../../documentation_source/source/examples/supplementary/data/SkytemHM-SLV.stm',
#                       '../../documentation_source/source/examples/supplementary/data/SkytemLM-SLV.stm'])

# ds.x = np.arange(n_points, dtype=np.float64)
# ds.y = np.zeros(n_points)
# ds.z = np.full(n_points, fill_value=30.0)
# ds.elevation = np.zeros(n_points)

# ds.transmitter = CircularLoops(x=ds.x, y=ds.y, z=ds.z,
#                 #  pitch=0.0, roll=0.0, yaw=0.0,
#                  radius=np.full(n_points, ds.system[0].loopRadius()))

# ds.receiver = CircularLoops(x=ds.transmitter.x -13.0,
#                  y=ds.transmitter.y + 0.0,
#                  z=ds.transmitter.z + 2.0,
#                 #  pitch=0.0, roll=0.0, yaw=0.0,
#                  radius=np.full(n_points, ds.system[0].loopRadius()))

# ds.relative_error = np.full((n_points, 2), fill_value = 0.05)
# ds.additive_error = np.full((n_points, 2), fill_value = 1e-13)

# dp = ds.datapoint(0)

# for k in range(n_points):    
#     thk = np.hstack([np.diff([0, zwedge[k], zdeep[k]]), np.inf])
#     mod = Model(RectilinearMesh1D(widths=thk), values=sigma)

#     dp.forward(mod)
#     dp.secondary_field = dp.predictedData
    
#     dp.secondary_field += dp.std * np.random.randn(dp.nChannels)
    
#     ds.secondary_field[k, :] = dp.data

# plt.figure();
# ax = plt.subplot(211);
# ds.plot();
# ax.get_legend().remove();
# plt.subplot(212, sharex=ax);
# plt.hlines(0.0, ds.x[0], ds.x[-1], 'k');
# plt.plot(ds.x, -zwedge, 'k');
# plt.plot(ds.x, -zdeep, 'k');

# d0 = ds.datapoint(0)
# d1 = ds.datapoint(99)
# plt.figure()
# plt.subplot(121)
# d0.plot()
# plt.subplot(122)
# d1.plot()

# plt.savefig('Skytem.png');

# #%%
# ds.write_csv('../../documentation_source/source/examples/supplementary/data/skytem.csv')

# #%%
# from geobipy import TdemData
# ds = TdemData.read_csv('../../documentation_source/source/examples/supplementary/data/skytem.csv', 
#                       ['../../documentation_source/source/examples/supplementary/data/SkytemHM-SLV.stm',
#                        '../../documentation_source/source/examples/supplementary/data/SkytemLM-SLV.stm'])

# #%%
# from geobipy import TdemData, TdemSystem, CircularLoop

# n_points = 100

# ds = TdemData(system=['../../documentation_source/source/examples/supplementary/data/aerotem.stm'])
# ds.x = np.arange(n_points, dtype=np.float64)
# ds.y = np.zeros(n_points)
# ds.z = np.full(n_points, fill_value=30.0)
# ds.elevation = np.zeros(n_points)

# ds.transmitter = CircularLoops(x=ds.x, y=ds.y, z=ds.z,
#                 #  pitch=0.0, roll=0.0, yaw=0.0,
#                  radius=np.full(n_points, fill_value=ds.system[0].loopRadius()))
# # ds.transmitter = [T] * ds.nPoints

# ds.receiver = CircularLoops(x=ds.transmitter.x + 2.0,
#                  y=ds.transmitter.y + 2.0,
#                  z=ds.transmitter.z - 12.0,
#                 #  pitch=0.0, roll=0.0, yaw=0.0,
#                  radius=np.full(n_points, fill_value=ds.system[0].loopRadius()))
# # ds.receiver = [R] * ds.nPoints

# ds.relative_error = np.full((n_points, 1), fill_value = 0.05)
# ds.additive_error = np.full((n_points, 1), fill_value = 1e-8)

# dp = ds.datapoint(0)

# for k in range(n_points):    
#     thk = np.hstack([np.diff([0, zwedge[k], zdeep[k]]), np.inf])
#     mod = Model(RectilinearMesh1D(widths=thk), values=sigma)

#     dp.forward(mod)
#     dp.secondary_field = dp.predicted_secondary_field

#     dp.secondary_field += dp.std * np.random.randn(dp.nChannels)

#     ds.secondary_field[k, :] = dp.data

# plt.figure()
# ax = plt.subplot(211)
# ds.plot()
# ax.get_legend().remove()
# plt.subplot(212, sharex=ax)
# plt.hlines(0.0, ds.x[0], ds.x[-1], 'k')
# plt.plot(ds.x, -zwedge, 'k')
# plt.plot(ds.x, -zdeep, 'k')

# d0 = ds.datapoint(0)
# d1 = ds.datapoint(99)
# plt.figure()
# plt.subplot(121)
# d0.plot()
# plt.subplot(122)
# d1.plot()

# #%%
# ds.write_csv('../../documentation_source/source/examples/supplementary/data/aerotem.csv')

# TdemData.read_csv('../../documentation_source/source/examples/supplementary/data/aerotem.csv', '../../documentation_source/source/examples/supplementary/data/aerotem.stm')

#%%
from geobipy import TempestData, TdemSystem, CircularLoop

n_points = 100

ds = TempestData(system=['../../documentation_source/source/examples/supplementary/data/Tempest.stm'])
ds.x = np.arange(n_points, dtype=np.float64)
ds.y = np.zeros(n_points)
ds.z = np.full(n_points, fill_value=121.0)
ds.elevation = np.zeros(n_points)

ds.transmitter = CircularLoops(x=ds.x, y=ds.y, z=ds.z,
                #  pitch=0.0, roll=0.0, yaw=0.0,
                 radius=np.full(n_points, fill_value=ds.system[0].loopRadius()))

ds.receiver = CircularLoops(x=ds.transmitter.x - 90.0,
                 y=ds.transmitter.y + 0.0,
                 z=ds.transmitter.z - 45.0,
                #  pitch=0.0, roll=0.0, yaw=0.0,
                 radius=np.full(n_points, fill_value=ds.system[0].loopRadius()))

ds.relative_error = np.full((n_points, ds.n_components), fill_value = 0.05)
add_error = np.r_[0.011474, 0.012810, 0.008507, 0.005154, 0.004742, 0.004477, 0.004168, 0.003539, 0.003352, 0.003213, 0.003161, 0.003122, 0.002587, 0.002038, 0.002201,
            0.007383, 0.005693, 0.005178, 0.003659, 0.003426, 0.003046, 0.003095, 0.003247, 0.002775, 0.002627, 0.002460, 0.002178, 0.001754, 0.001405, 0.001283]
ds.additive_error = np.repeat(add_error[None, :], n_points, 0)


dp = ds.datapoint(0)

for k in range(n_points):    
    edges = np.hstack([0, zwedge[k], zdeep[k], np.inf])
    mod = Model(RectilinearMesh1D(edges=edges), values=sigma)

    dp.forward(mod)
    noise = (np.random.randn(dp.nChannels) * np.sqrt((0.05 * dp.predicted_secondary_field)**2 + dp.addErr**2))

    ds.primary_field[k, :] = dp.predicted_primary_field
    ds.secondary_field[k, :] = dp.predicted_secondary_field + noise
      
plt.figure()
ax = plt.subplot(211)
ds.secondary_field.plot()
plt.subplot(212, sharex=ax)
plt.hlines(0.0, ds.x[0], ds.x[-1], 'k')
plt.plot(ds.x, -zwedge, 'k')
plt.plot(ds.x, -zdeep, 'k')

#%%
ds.write_csv('../../documentation_source/source/examples/supplementary/data/Tempest.csv')

#%%
from geobipy import TempestData
ds = TempestData.read_csv('../../documentation_source/source/examples/supplementary/data/Tempest.csv', '../../documentation_source/source/examples/supplementary/data/Tempest.stm')

# plt.show()