.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_examples_Models_model_1d.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_Models_model_1d.py:


1D Model with an infinite halfspace
-----------------------------------


.. code-block:: default


    from geobipy import StatArray
    from geobipy import Model1D
    import matplotlib.pyplot as plt
    import numpy as np
    import h5py
    from geobipy import hdfRead







Make a test model with 10 layers, and increasing parameter values


.. code-block:: default


    par = StatArray(np.linspace(0.01, 0.1, 10), "Conductivity", "$\\frac{S}{m}$")
    thk = StatArray(np.ones(9) * 10.0)
    mod = Model1D(parameters=par, thickness=thk)









.. code-block:: default



    mod.summary()






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    1D Model: 
    Name: # of Cells
         Units: 
         Shape: (1,)
         Values: [10]
    Top of the model: [0.]
    Name: Thickness
         Units: m
         Shape: (10,)
         Values: [10. 10. 10. ... 10. 10. inf]
    Name: Conductivity
         Units: $\frac{S}{m}$
         Shape: (10,)
         Values: [0.01 0.02 0.03 ... 0.08 0.09 0.1 ]
    Name: Depth
         Units: m
         Shape: (10,)
         Values: [10. 20. 30. ... 80. 90. inf]




Randomness and Model Perturbations
We can make the 1D model perturbable by assigning minimum and maximum layer
depths, a maximum number of layers, and a probability wheel describing the
relative probabilities of either creating a layer, deleting a layer, moving
an interface, or doing nothing.


.. code-block:: default


    prng = np.random.RandomState()
    # Assign probabilities to the model layers
    # They are the cumulative probability of life-death-perturb-doNothing
    mod.setPriors(pWheel=[0.5, 0.05, 0.15, 0.1], halfSpaceValue=1.0, minDepth=1.0, maxDepth=150.0, maxLayers=30, prng=prng)
    # We can then perturb the layers of the model
    perturbed = mod.perturbStructure()









.. code-block:: default



    plt.figure(figsize=(8,6))
    plt.subplot(121)
    mod.pcolor(grid=True)
    plt.subplot(122)
    perturbed.pcolor(grid=True)
    plt.savefig('Perturbed.png', dpi=200, figsize=(8,6))





.. image:: /examples/Models/images/sphx_glr_model_1d_001.png
    :class: sphx-glr-single-img





.. code-block:: default



    plt.figure()
    mod.plot()





.. image:: /examples/Models/images/sphx_glr_model_1d_002.png
    :class: sphx-glr-single-img




We can evaluate the prior of the model


.. code-block:: default


    try:
      tmp.priorProbability(True,True) # This is meant to fail here!
    except:
      print('This will not work because no prior has been assigned')





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    This will not work because no prior has been assigned



This last command failed because we did not assign a prior distribution to the model parameters


.. code-block:: default


    # Set priors on the depth interfaces, given a number of layers
    mod.depth.setPrior('Order',mod.minDepth,mod.maxDepth,mod.minThickness,30)
    # To include priors on the parameter and change in the parameter, we need to assign their priors
    # Assign a multivariate normal distribution that is logged to the conductivities
    mod.par.setPrior('MvNormalLog',np.log(0.004),np.log(11.0), prng=prng)
    # Assign a prior to the derivative of the model
    mod.dpar.setPrior('MvNormalLog',0.0,np.float64(1.5), prng=prng)
    # We can evaluate the prior of the model using depths only
    print('Probability of the Model given its priors: ', mod.priorProbability(False,False))
    # Or with priors on its parameters, and parameter gradient with depth.
    print('Probability of the Model given its priors: ', mod.priorProbability(True,True))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Probability of the Model given its priors:  -6.321532975483965
    Probability of the Model given its priors:  -43.41755252558525



Evaluating the prior uses the probability of the parameter distributions


.. code-block:: default


    # Evaluate the probability for these depths
    print(mod.depth.probability(mod.nCells))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    0.113338300042515



Write to a HDF5 file


.. code-block:: default


    with h5py.File('Model1D.h5','w') as hf:
        mod.toHdf(hf,'Model1D')








Read from the file


.. code-block:: default



    ModNew=hdfRead.readKeyFromFiles('Model1D.h5','/','Model1D')









.. code-block:: default



    plt.figure()
    ax = plt.subplot(131)
    ModNew.pcolor(grid=True)
    plt.subplot(133, sharey = ax)
    ModNew.plot(flipY=False)





.. image:: /examples/Models/images/sphx_glr_model_1d_003.png
    :class: sphx-glr-single-img




Creating memory in HDF5 to accomodate multiple models


.. code-block:: default


    # Create an initial Model class with enough layers to hold as many layers as we expect. (an upper bound)
    tmp = Model1D(nCells=20)

    # Open the file
    f = h5py.File('Model1D.h5','w')

    # Create the memory using the temporary model with space for 2 models.
    tmp.createHdf(f, myName='test', nRepeats=2)

    # Write mod and perturbed to different entries in the HDF5 file
    mod.writeHdf(f, 'test', index=0)
    perturbed.writeHdf(f, 'test', index=1)

    # Close the file
    f.close()







Reading from a HDF5 file with multiple models


.. code-block:: default


    # Special read functions
    from geobipy import hdfRead
    # Open the file
    f = h5py.File('Model1D.h5', 'r')
    # Read the Model1D from the file
    tmp = hdfRead.readKeyFromFile(f, fName='Model1D.h5', groupName='/', key='test', index=1)
    f.close()







Creating a hitmap and adding a 1D model to it


.. code-block:: default


    from geobipy import Hitmap2D
    x = StatArray(np.logspace(-3, -0, 100), name='Parameter')
    y = StatArray(np.linspace(0.0, 200.0, 100), name='Depth', units='m')
    Hit = Hitmap2D(xBins=x, yBins=y)








If we perturb a model multiple times, we can add each model to the hitmap


.. code-block:: default


    perturbed.addToHitMap(Hitmap=Hit)
    for i in range(100):
        perturbed = perturbed.perturbStructure()
        perturbed.addToHitMap(Hitmap=Hit)









.. code-block:: default



    plt.figure()
    Hit.pcolor(flipY=True, xscale='log', cmap='gray_r')




.. image:: /examples/Models/images/sphx_glr_model_1d_004.png
    :class: sphx-glr-single-img




We can access and plot the elements of model. The parameters are an [StatArray](../../Base/StatArray_Class.ipynb)


.. code-block:: default


    plt.figure()
    mod.par.plot()




.. image:: /examples/Models/images/sphx_glr_model_1d_005.png
    :class: sphx-glr-single-img




Or we can plot the 1D model as coloured blocks


.. code-block:: default


    plt.figure()
    perturbed.pcolor(grid=True)



.. image:: /examples/Models/images/sphx_glr_model_1d_006.png
    :class: sphx-glr-single-img





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  1.796 seconds)


.. _sphx_glr_download_examples_Models_model_1d.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: model_1d.py <model_1d.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: model_1d.ipynb <model_1d.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
