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

base_files = '..//..//data'
dataFolder = join(base_files, 'data')

file_path = os.path.join(base_files, 'output', 'skytem_test')
pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
output_directory = file_path

data_type = 'skytem'
systemFile=[join(dataFolder, 'SkytemHM.stm'), join(dataFolder, 'SkytemLM.stm')]


par = StatArray(np.r_[0.01, 0.1, 5], "Conductivity", "$\frac{S}{m}$")
mod = Model(RectilinearMesh1D(edges=np.r_[0, 10.0, 75.0, np.inf]), values=par)

tdp = TdemData(system=systemFile)
tdp.x = 0.0
tdp.y = 0.0
tdp.z = 30.0
tdp.elevation = 0.0

tdp.loop_pair.transmitter = CircularLoop(x=tdp.x, y=tdp.y, z=tdp.z,
                #  pitch=0.0, roll=0.0, yaw=0.0,
                # radius=[tdp.system[0].loopRadius()])
)

tdp.loop_pair.receiver = CircularLoop(x=tdp.transmitter.x - 13.0,
                y=tdp.transmitter.y + 0.0,
                z=tdp.transmitter.z + 2.0,
                # radius=[tdp.system[0].loopRadius()])
)


dp = tdp.datapoint(0)

dp.forward(mod)
dp.secondary_field[:] = dp.predicted_secondary_field
dp.relative_error = np.r_[0.03, 0.03]

# dp.additive_error = np.minimum(np.r_[1e-14, 1e-13], dp.secondary_field[np.r_[dp.channels_per_system[0]-2, -2]])
dp.additive_error = dp.secondary_field[np.r_[dp.channels_per_system[0]-1, -1]]

dp_noisy = deepcopy(dp)
dp_noisy.secondary_field += prng.normal(scale=dp.std, size=dp.n_channels)

#sing = tdp.datapoint(index=0)
# sing = tdp_noisy.datapoint(index=0)
sing = dp_noisy

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
kwargs['save_hdf5'] = False
kwargs['save_png'] = False

kwargs['solve_value'] = False
# kwargs['value_weight'] = None
kwargs['maximum_number_of_layers'] = 30
kwargs['minimum_depth'] = 1.0
kwargs['maximum_depth'] = 300.0
kwargs['minimum_thickness'] = None
kwargs['number_of_depth_bins'] = 133
kwargs['probability_of_birth'] = 1.0/6.0
kwargs['probability_of_death'] = 1.0/6.0
kwargs['probability_of_perturb'] = 1.0/6.0
kwargs['probability_of_no_change'] = 0.5

# kwargs['value_mean'] = 0.001
# kwargs['value_standard_deviation'] = 2.39
kwargs['value_limits'] = None
kwargs['penalize_thin_layers'] = True
kwargs['stochastic_newton'] = True

kwargs['solve_gradient'] = True
# kwargs['gradient_weight'] = None
kwargs['gradient_standard_deviation'] = None


kwargs['step_length'] = None
kwargs['multiplier'] = None


kwargs['solve_relative_error'] = True
kwargs['initial_relative_error'] = np.array([0.03, 0.03])
kwargs['minimum_relative_error'] = np.array([0.005, 0.005])
kwargs['maximum_relative_error'] = np.array([0.5, 0.5])
kwargs['relative_error_proposal_variance'] = np.array([1e-6, 1e-6])

kwargs['solve_additive_error'] = True
kwargs['initial_additive_error'] = sing.additive_error
kwargs['minimum_additive_error'] = np.array([1e-20, 1e-20])
kwargs['maximum_additive_error'] = np.array([1e-10, 1e-10])
kwargs['additive_error_proposal_variance'] = sing.additive_error * 1e10
kwargs['additive_error_proposal_variance'] = np.array([1e-3, 1e-3])

kwargs['solve_height'] = False
kwargs['maximum_height_change'] = 1.0
kwargs['height_proposal_variance'] = 0.01


kwargs['solve_calibration'] = False
kwargs['clip_ratio'] = None


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