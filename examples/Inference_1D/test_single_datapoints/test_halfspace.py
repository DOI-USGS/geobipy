import os
import sys

from os.path import join
import pathlib
import numpy as np
import h5py
from datetime import timedelta
import time
import matplotlib.pyplot as plt
from geobipy import Waveform
from geobipy import SquareLoop, CircularLoop
from geobipy import butterworth
from geobipy import TdemSystem
from geobipy import TdemData
from geobipy import TdemDataPoint
from geobipy import RectilinearMesh1D
from geobipy import RectilinearMesh2D
from geobipy import RectilinearMesh3D
from geobipy import Inference3D
from geobipy import Inference1D
from geobipy import Model
from geobipy import StatArray
from geobipy import Distribution
from geobipy import user_parameters
from copy import deepcopy
from geobipy import get_prng

np.random.seed(0)
seed = 146100583096709124601953385843316024947
prng = get_prng(seed=seed)

base_files = '../../../'
dataFolder = join(base_files, 'supplementary', 'data')
supplementary = os.path.join(base_files, "supplementary")

file_path = os.path.join(base_files, 'output', 'skytem_test')
pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
output_directory = file_path

data_type = 'skytem'
systemFile=[join(dataFolder, 'SkytemHM.stm'), join(dataFolder, 'SkytemLM.stm')]


par = StatArray(np.r_[0.3 - 0.2], "Conductivity", "$\frac{S}{m}$")
mod = Model(RectilinearMesh1D(edges=np.r_[0, np.inf]), values=par)

tdp = TdemData(system=systemFile)
tdp.x = 0.0
tdp.y = 0.0
tdp.z = 30.0
tdp.elevation = 0.0

tdp.loop_pair.transmitter = CircularLoop(x=tdp.x, y=tdp.y, z=tdp.z,
                #  pitch=0.0, roll=0.0, yaw=0.0,
                radius=[tdp.system[0].loopRadius()])

tdp.loop_pair.receiver = CircularLoop(x=tdp.transmitter.x - 13.0,
                y=tdp.transmitter.y + 0.0,
                z=tdp.transmitter.z + 2.0,
                radius=[tdp.system[0].loopRadius()])

tdp.relative_error = [[0.03, 0.03]]
tdp.additive_error = [[1e-14, 1e-12]]

dp = tdp.datapoint(0)

dp.forward(mod)
tdp.secondary_field[0, :] = dp.predicted_secondary_field
tdp_noisy = deepcopy(tdp)
tdp_noisy.secondary_field += prng.normal(scale=tdp.std, size=(1, tdp.n_channels))

#sing = tdp.datapoint(index=0)
sing = tdp_noisy.datapoint(index=0)

plt.figure()
plt.subplot(121)
_ = mod.plot(flipY=True, xscale='log')
plt.subplot(122)
_ = sing.plot()
_ = sing.plot_predicted()
plt.tight_layout()

kwargs = {}# user_parameters.read(inputFile)
kwargs['n_markov_chains'] = 100000
kwargs['update_plot_every'] = 5000
kwargs['live_plot'] = True
kwargs['system_filename'] = systemFile
kwargs['save_hdf5'] = False
kwargs['save_png'] = False
kwargs['solve_parameter'] = False
kwargs['solve_gradient'] = True
kwargs['solve_relative_error'] = True
kwargs['solve_additive_error'] = True
kwargs['solve_height'] = False
kwargs['solve_calibration'] = False
kwargs['maximum_number_of_layers'] = 30
kwargs['minimum_depth'] = 1.0
kwargs['maximum_depth'] = 1000.0
kwargs['minimum_thickness'] = None
kwargs['number_of_depth_bins'] = 133
kwargs['initial_relative_error'] = np.array([0.03, 0.03])
kwargs['minimum_relative_error'] = np.array([0.005, 0.005])
kwargs['maximum_relative_error'] = np.array([0.1, 0.1])
kwargs['initial_additive_error'] = np.array([1e-14, 1e-12])
kwargs['minimum_additive_error'] = np.array([1e-16, 1e-16])
kwargs['maximum_additive_error'] = np.array([1e-10, 1e-10])
kwargs['maximum_height_change'] = 1.0
kwargs['relative_error_proposal_variance'] = np.array([1e-6, 1e-6])
kwargs['additive_error_proposal_variance'] = np.array([1e-4, 1e-4])
kwargs['height_proposal_variance'] = 0.01
kwargs['probability_of_birth'] = 1.0/6.0
kwargs['probability_of_death'] = 1.0/6.0
kwargs['probability_of_perturb'] = 1.0/6.0
kwargs['probability_of_no_change'] = 0.5
kwargs['stochastic_newton'] = True

kwargs['parameter_mean'] = None

kwargs['factor'] = None
kwargs['step_length'] = None
kwargs['multiplier'] = None
kwargs['clip_ratio'] = None
kwargs['ignore_likelihood'] = False
kwargs['parameter_limits'] = None

kwargs['verbose'] = True

kwargs['seed'] = seed

inference = Inference1D(prng=prng, **kwargs)
inference.initialize(sing)
inference.infer(hdf_file_handle=None)

r'''
from matplotlib import pyplot as plt
fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(2, 2)

p = np.unravel_index(0, (2, 2))
fig, ax = inference._init_posterior_plots(gs=gs[p])
inference.plot_posteriors(axes=gs[p])
'''

plt.show(block=True)
#plt.savefig('{}_{}_points.png'.format(data_type, model_type), dpi=300)