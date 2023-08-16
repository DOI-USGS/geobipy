"""
Do Stuff
--------

All plotting in GeoBIPy can be carried out using the 3D inference class

"""

import matplotlib.pyplot as plt
import numpy as np
from geobipy import Inference2D
from create_model import create_model

def create_plots(model_type, system):
    #%%
    # Inference for a line of inferences
    # ++++++++++++++++++++++++++++++++++
    #
    # We can instantiate the inference handler by providing a path to the directory containing
    # HDF5 files generated by GeoBIPy.
    #
    # The InfereceXD classes are low memory.  They only read information from the HDF5 files
    # as and when it is needed.
    #
    # The first time you use these classes to create plots, expect long initial processing times.
    # I precompute expensive properties and store them in the HDF5 files for later use.

    ################################################################################
    results_2d = Inference2D.fromHdf('../parallel/skytem_{}/{}/0.0.h5'.format(system, model_type))

    ################################################################################
    # Plot a location map of the data point locations along the line

    ################################################################################
    # Before we start plotting cross sections, lets set some common keywords
    xAxis = 'x'
    kwargs = {
                #   "reciprocateParameter" : True, # Plot resistivity instead?
                # "vmin" : -2.0, # Set the minimum colour bar range in log space
                # "vmax" : 0.0, # Set the maximum colour bar range in log space
                "log" : 10, # I want to plot the log conductivity
            #    "equalize" : True
                "cmap" : 'jet'
                }

    results_2d.compute_mean_parameter()
    ################################################################################
    # We can show a basic cross-section of the parameter inverted for
    plt.figure(figsize=(16, 4))
    plt.suptitle(model_type)

    true_model = create_model(model_type)
    kwargs['vmin'] = np.log10(np.min(true_model.values))
    kwargs['vmax'] = np.log10(np.max(true_model.values))

    plt.subplot(4, 1, 1)
    true_model.pcolor(**kwargs); results_2d.plot_data_elevation(linewidth=0.3); results_2d.plot_elevation(linewidth=0.3);
    plt.ylim([-550, 60])

    plt.subplot(4, 1, 2)
    results_2d.plot_mean_model(use_variance=True, **kwargs); results_2d.plot_data_elevation(linewidth=0.3); results_2d.plot_elevation(linewidth=0.3);
    plt.subplot(4, 1, 3)
    results_2d.plot_median_model(use_variance=True, **kwargs); results_2d.plot_data_elevation(linewidth=0.3); results_2d.plot_elevation(linewidth=0.3);
    plt.subplot(4, 1, 4)
    results_2d.plot_mode_model(use_variance=True, **kwargs); results_2d.plot_data_elevation(linewidth=0.3); results_2d.plot_elevation(linewidth=0.3);

    # plt.show(block=True)
    plt.savefig('skytem_{}_{}_mmm.png'.format(system, model_type), dpi=600)


if __name__ == '__main__':
   types = ['glacial']#, 'saline_clay', 'resistive_dolomites', 'resistive_basement', 'coastal_salt_water', 'ice_over_salt_water']

   for model in types:
      print(model)
      create_plots(model, 304)
      create_plots(model, 512)