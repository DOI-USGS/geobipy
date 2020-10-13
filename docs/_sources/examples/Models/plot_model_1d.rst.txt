.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_examples_Models_plot_model_1d.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_examples_Models_plot_model_1d.py:


1D Model with an infinite halfspace
-----------------------------------


.. code-block:: default

    from geobipy import StatArray
    from geobipy import Model1D
    from geobipy import Distribution
    from geobipy import FdemData
    import matplotlib.pyplot as plt
    import numpy as np
    import h5py
    from geobipy import hdfRead








Instantiate the 1D Model with a Half Space
++++++++++++++++++++++++++++++++++++++++++


.. code-block:: default


    # Make a test model with 10 layers, and increasing parameter values
    nLayers = 2
    par = StatArray(np.linspace(0.001, 0.02, nLayers), "Conductivity", "$\\frac{S}{m}$")
    thk = StatArray(np.full(nLayers-1, fill_value=10.0))
    mod = Model1D(parameters=par, thickness=thk)








Randomness and Model Perturbations
++++++++++++++++++++++++++++++++++
We can set the priors on the 1D model by assigning minimum and maximum layer
depths and a maximum number of layers.  These are used to create priors on
the number of cells in the model, a new depth interface, new parameter values
and the vertical gradient of those parameters.
The halfSpaceValue is used as a reference value for the parameter prior.


.. code-block:: default

    prng = np.random.RandomState()
    # Set the priors
    mod.setPriors(halfSpaceValue = 0.01,
                  minDepth = 1.0, 
                  maxDepth = 150.0, 
                  maxLayers = 30, 
                  parameterPrior = True, 
                  gradientPrior = True, 
                  prng = prng)








To propose new models, we specify the probabilities of creating, removing, perturbing, and not changing
a layer interface


.. code-block:: default

    pProposal = Distribution('LogNormal', 0.01, np.log(2.0)**2.0, linearSpace=True, prng=prng)
    mod.setProposals(probabilities = [0.25, 0.25, 0.25, 0.25], parameterProposal=pProposal, prng=prng)








We can then perturb the layers of the model
perturbed = mod.perturbStructure()


.. code-block:: default

    remapped, perturbed = mod.perturb()









.. code-block:: default

    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(121)
    mod.pcolor(grid=True)
    ax = plt.subplot(122)
    perturbed.pcolor(grid=True)




.. image:: /examples/Models/images/sphx_glr_plot_model_1d_001.png
    :alt: plot model 1d
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customPlots.py:873: MatplotlibDeprecationWarning: shading='flat' when X and Y have the same dimensions as C is deprecated since 3.3.  Either specify the corners of the quadrilaterals with X and Y, or pass shading='auto', 'nearest' or 'gouraud', or set rcParams['pcolor.shading'].  This will become an error two minor releases later.
      pm = ax.pcolormesh(X, Y, v, color=c, **kwargs)

    <AxesSubplot:ylabel='Depth (m)'>



We can evaluate the prior of the model using depths only


.. code-block:: default

    print('Log probability of the Model given its priors: ', mod.priorProbability(False, False, log=True))
    # Or with priors on its parameters, and parameter gradient with depth.
    print('Log probability of the Model given its priors: ', mod.priorProbability(True, True, log=True))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Log probability of the Model given its priors:  -11.358381323561012
    Log probability of the Model given its priors:  -20.111609194656108




Perturbing a model multiple times
+++++++++++++++++++++++++++++++++
In the stochasitic inference process, we perturb the model structure, 
and parameter values, multiple times. 
Each time the model is perturbed, we can record its state
in a posterior distribution.

For a 1D model, the parameter posterior is a 2D hitmap with depth in one dimension
and the parameter value in the other.
We also attach a 1D histogram for the number of layers,
and a 1D histogram for the locations of interfaces.

Since we have already set the priors on the Model, we can set the posteriors
based on bins from from the priors.


.. code-block:: default

    mod.setPosteriors()

    mod0 = mod.deepcopy()








Now we randomly perturb the model, and update its posteriors.


.. code-block:: default

    mod.updatePosteriors()
    for i in range(1000):
        remapped, perturbed = mod.perturb()

        # And update the model posteriors
        perturbed.updatePosteriors()

        mod = perturbed








We can now plot the posteriors of the model.

Remember in this case, we are simply perturbing the model structure and parameter values
The proposal for the parameter values is fixed and centred around a single value.


.. code-block:: default

    fig = plt.figure(figsize=(8, 6))
    plt.subplot(131)
    mod.nCells.posterior.plot()
    ax = plt.subplot(132)
    mod.par.posterior.pcolor(cmap='gray_r', xscale='log', noColorbar=True, flipY=True)
    plt.subplot(133, sharey=ax)
    mod.depth.posterior.plot(rotate=True, flipY=True);



.. image:: /examples/Models/images/sphx_glr_plot_model_1d_002.png
    :alt: plot model 1d
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customPlots.py:649: MatplotlibDeprecationWarning: You are modifying the state of a globally registered colormap. In future versions, you will not be able to modify a registered colormap in-place. To remove this warning, you can make a copy of the colormap first. cmap = copy.copy(mpl.cm.get_cmap("gray_r"))
      kwargs['cmap'].set_bad(color='white')

    <AxesSubplot:xlabel='Frequency', ylabel='Depth (m)'>




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  2.776 seconds)


.. _sphx_glr_download_examples_Models_plot_model_1d.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_model_1d.py <plot_model_1d.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_model_1d.ipynb <plot_model_1d.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
