.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_examples_Models_plot_model_1d.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_Models_plot_model_1d.py:


1D Model with an infinite halfspace
-----------------------------------


.. code-block:: default

    from geobipy import StatArray
    from geobipy import Model1D
    import matplotlib.pyplot as plt
    import numpy as np
    import h5py
    from geobipy import hdfRead








Instantiate the 1D Model with a Half Space
++++++++++++++++++++++++++++++++++++++++++


.. code-block:: default


    # Make a test model with 10 layers, and increasing parameter values
    par = StatArray(np.linspace(0.01, 0.1, 10), "Conductivity", "$\\frac{S}{m}$")
    thk = StatArray(np.ones(9) * 10.0)
    mod = Model1D(parameters=par, thickness=thk)








Randomness and Model Perturbations
We can set the priors on the 1D model by assigning minimum and maximum layer
depths and a maximum number of layers.  These are used to create priors on
the number of cells in the model, a new depth interface, new parameter values
and the vertical gradient of those parameters.


.. code-block:: default

    prng = np.random.RandomState()
    # Set the priors
    mod.setPriors(halfSpaceValue=1.0, minDepth=1.0, maxDepth=150.0, maxLayers=30, parameterPrior=True, gradientPrior=True, prng=prng)








To propose new models, we specify the probabilities of creating, removing, perturbing, and not changing
a layer interface
mod.setProposals(probabilities = [0.25, 0.25, 0.1, 0.1], prng=prng)


.. code-block:: default


    # ################################################################################
    # # We can then perturb the layers of the model
    # perturbed = mod.perturbStructure()

    # ################################################################################
    # plt.figure(figsize=(8,6))
    # plt.subplot(121)
    # mod.pcolor(grid=True)
    # plt.subplot(122)
    # _ = perturbed.pcolor(grid=True)

    # ################################################################################
    # plt.figure()
    # _ = mod.plot()

    # ################################################################################
    # # We can evaluate the prior of the model using depths only
    # print('Probability of the Model given its priors: ', mod.priorProbability(False, False))
    # # Or with priors on its parameters, and parameter gradient with depth.
    # print('Probability of the Model given its priors: ', mod.priorProbability(True, True))








Perturbing a model multiple times
+++++++++++++++++++++++++++++++++
We have already 


.. code-block:: default


    # ################################################################################
    # # If we perturb a model multiple times, we can add each model to the hitmap
    # perturbed.addToHitMap(Hitmap=Hit)
    # for i in range(1000):
    #     perturbed = perturbed.perturbStructure()
    #     perturbed.addToHitMap(Hitmap=Hit)

    # ################################################################################
    # plt.figure()
    # _ = Hit.pcolor(flipY=True, xscale='log', cmap='gray_r')







    # ################################################################################
    # # Write to a HDF5 file

    # with h5py.File('Model1D.h5','w') as hf:
    #     mod.toHdf(hf,'Model1D')

    # ################################################################################
    # # Read from the file
    # ModNew = hdfRead.readKeyFromFiles('Model1D.h5','/','Model1D')


    # ################################################################################
    # plt.figure()
    # ax = plt.subplot(131)
    # ModNew.pcolor(grid=True)
    # plt.subplot(133, sharey = ax)
    # _ = ModNew.plot(flipY=False)


    # ################################################################################
    # # Creating memory in HDF5 to accomodate multiple models

    # # Create an initial Model class with enough layers to hold as many layers as we expect. (an upper bound)
    # tmp = Model1D(nCells=20)

    # # Open the file
    # f = h5py.File('Model1D.h5','w')

    # # Create the memory using the temporary model with space for 2 models.
    # tmp.createHdf(f, myName='test', nRepeats=2)

    # # Write mod and perturbed to different entries in the HDF5 file
    # mod.writeHdf(f, 'test', index=0)
    # perturbed.writeHdf(f, 'test', index=1)

    # # Close the file
    # f.close()

    # ################################################################################
    # # Reading from a HDF5 file with multiple models

    # # Special read functions
    # from geobipy import hdfRead
    # # Open the file
    # f = h5py.File('Model1D.h5', 'r')
    # # Read the Model1D from the file
    # tmp = hdfRead.readKeyFromFile(f, fName='Model1D.h5', groupName='/', key='test', index=1)
    # f.close()



    # ################################################################################
    # # We can access and plot the elements of model. The parameters are an [StatArray](../../Base/StatArray_Class.ipynb)
    # plt.figure()
    # _ = mod.par.plot()

    # ################################################################################
    # # Or we can plot the 1D model as coloured blocks
    # plt.figure()
    # _ = perturbed.pcolor(grid=True)








.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.004 seconds)


.. _sphx_glr_download_examples_Models_plot_model_1d.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_model_1d.py <plot_model_1d.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_model_1d.ipynb <plot_model_1d.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
