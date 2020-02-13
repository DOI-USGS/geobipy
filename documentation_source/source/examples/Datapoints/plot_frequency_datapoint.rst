.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_examples_Datapoints_plot_frequency_datapoint.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_Datapoints_plot_frequency_datapoint.py:


Frequency domain datapoint
--------------------------

There are two ways in which to create a frequency domain datapoint,

1) :ref:`Instantiating a frequency domain data point`

2) :ref:`Obtaining a datapoint from a dataset`

Once instantiated, see :ref:`Using a frequency domain datapoint`


.. code-block:: default

    from os.path import join
    import numpy as np
    import h5py
    import matplotlib.pyplot as plt
    from geobipy import hdfRead
    from geobipy import CircularLoop
    from geobipy import FdemSystem
    from geobipy import FdemData
    from geobipy import FdemDataPoint
    from geobipy import Model1D
    from geobipy import StatArray
    from geobipy import Distribution








Instantiating a frequency domain data point
+++++++++++++++++++++++++++++++++++++++++++

To instantiate a frequency domain datapoint we need to define some
characteristics of the acquisition system.

We need to define the frequencies in Hz of the transmitter,
and the geometery of the loops used for each frequency.


.. code-block:: default


    frequencies = np.asarray([380.0, 1776.0, 3345.0, 8171.0, 41020.0, 129550.0])

    transmitterLoops = [CircularLoop(orient='z'),     CircularLoop(orient='z'),
                        CircularLoop('x', moment=-1), CircularLoop(orient='z'),
                        CircularLoop(orient='z'),     CircularLoop(orient='z')]

    receiverLoops    = [CircularLoop(orient='z', x=7.93),    CircularLoop(orient='z', x=7.91),
                        CircularLoop('x', moment=1, x=9.03), CircularLoop(orient='z', x=7.91),
                        CircularLoop(orient='z', x=7.91),    CircularLoop(orient='z', x=7.89)]








Now we can instantiate the system.


.. code-block:: default

    fds = FdemSystem(frequencies, transmitterLoops, receiverLoops)








And use the system to instantiate a datapoint

Note the extra arguments that can be used to create the data point.
data is for any observed data one might have, while std are the estimated standard
deviations of those observed data.

Define some in-phase then quadrature data for each frequency.


.. code-block:: default

    data = np.r_[145.3, 435.8, 260.6, 875.1, 1502.7, 1516.9,
                 217.9, 412.5, 178.7, 516.5, 405.7, 255.7]

    fdp = FdemDataPoint(x=0.0, y=0.0, z=30.0, elevation=0.0,
                        data=data, std=None, predictedData=None,
                        system=fds, lineNumber=0.0, fiducial=0.0)









.. code-block:: default

    plt.figure()
    _ = fdp.plot()




.. image:: /examples/Datapoints/images/sphx_glr_plot_frequency_datapoint_001.png
    :class: sphx-glr-single-img





Obtaining a datapoint from a dataset
++++++++++++++++++++++++++++++++++++

More often than not, our observed data is stored in a file on disk.
We can read in a dataset and pull datapoints from it.

For more information about the frequency domain data set see :ref:`Frequency domain dataset`

Set some paths and file names


.. code-block:: default

    dataFolder = "..//supplementary//Data//"
    # The data file name
    dataFile = dataFolder + 'Resolve2.txt'
    # The EM system file name
    systemFile = dataFolder + 'FdemSystem2.stm'








Initialize and read an EM data set


.. code-block:: default

    D = FdemData()
    D.read(dataFile,systemFile)








Get a data point from the dataset


.. code-block:: default

    fdp = D.datapoint(0)
    plt.figure()
    _ = fdp.plot()




.. image:: /examples/Datapoints/images/sphx_glr_plot_frequency_datapoint_002.png
    :class: sphx-glr-single-img





Using a datapoint
+++++++++++++++++

We can define a 1D layered earth model, and use it to predict some data


.. code-block:: default

    nCells = 19
    par = StatArray(np.linspace(0.01, 0.1, nCells), "Conductivity", "$\frac{S}{m}$")
    thk = StatArray(np.ones(nCells-1) * 10.0)
    mod = Model1D(nCells = nCells, parameters=par, thickness=thk)








Forward model the data


.. code-block:: default

    fdp.forward(mod)









.. code-block:: default

    plt.figure()
    plt.subplot(121)
    _ = mod.pcolor()
    plt.subplot(122)
    _ = fdp.plotPredicted()
    plt.tight_layout()




.. image:: /examples/Datapoints/images/sphx_glr_plot_frequency_datapoint_003.png
    :class: sphx-glr-single-img





Compute the sensitivity matrix for a given model


.. code-block:: default

    J = fdp.sensitivity(mod)
    plt.figure()
    _ = np.abs(J).pcolor(equalize=True, log=10, flipY=True)




.. image:: /examples/Datapoints/images/sphx_glr_plot_frequency_datapoint_004.png
    :class: sphx-glr-single-img





Attaching statistical descriptors to the datapoint
++++++++++++++++++++++++++++++++++++++++++++++++++

Define a multivariate log normal distribution as the prior on the predicted data.


.. code-block:: default

    fdp.predictedData.setPrior('MvLogNormal', fdp.data[fdp.active], fdp.std[fdp.active]**2.0)








This allows us to evaluate the likelihood of the predicted data


.. code-block:: default

    print(fdp.likelihood(log=True))
    # Or the misfit
    print(fdp.dataMisfit())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    -316.19118049547296
    22.819882232476




We can perform a quick search for the best fitting half space


.. code-block:: default

    halfspace = fdp.FindBestHalfSpace()
    print('Best half space conductivity is {} $S/m$'.format(halfspace.par))
    plt.figure()
    _ = fdp.plot()
    _ = fdp.plotPredicted()




.. image:: /examples/Datapoints/images/sphx_glr_plot_frequency_datapoint_005.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Best half space conductivity is [0.00982172] $S/m$




Compute the misfit between observed and predicted data


.. code-block:: default

    print(fdp.dataMisfit())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    26.18321050483556




Plot the misfits for a range of half space conductivities


.. code-block:: default

    plt.figure()
    _ = fdp.plotHalfSpaceResponses(-6.0, 4.0, 200)
    plt.title("Halfspace responses");




.. image:: /examples/Datapoints/images/sphx_glr_plot_frequency_datapoint_006.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    Text(0.5, 1.0, 'Halfspace responses')



We can attach priors to the height of the datapoint,
the relative error multiplier, and the additive error noise floor


.. code-block:: default


    # Set values of relative and additive error for both systems.
    fdp.relErr = 0.05
    fdp.addErr = 10

    # Define the distributions used as priors.
    heightPrior = Distribution('Uniform', min=np.float64(fdp.z) - 2.0, max=np.float64(fdp.z) + 2.0)
    relativePrior = Distribution('Uniform', min=0.01, max=0.5)
    additivePrior = Distribution('Uniform', min=5, max=15)
    fdp.setPriors(heightPrior, relativePrior, additivePrior)








In order to perturb our solvable parameters, we need to attach proposal distributions


.. code-block:: default

    heightProposal = Distribution('Normal', mean=fdp.z, variance = 0.01)
    relativeProposal = Distribution('MvNormal', mean=fdp.relErr, variance=2.5e-7)
    additiveProposal = Distribution('MvLogNormal', mean=fdp.addErr, variance=1e-4)
    fdp.setProposals(heightProposal, relativeProposal, additiveProposal)








With priorss set we can auto generate the posteriors


.. code-block:: default

    fdp.setPosteriors()








Perturb the datapoint and record the perturbations


.. code-block:: default

    for i in range(1000):
        fdp.perturb(True, True, True, False)
        fdp.updatePosteriors()








Plot the posterior distributions


.. code-block:: default

    plt.figure()
    _ = fdp.z.plotPosteriors()




.. image:: /examples/Datapoints/images/sphx_glr_plot_frequency_datapoint_007.png
    :class: sphx-glr-single-img






.. code-block:: default

    plt.figure()
    _ = fdp.relErr.plotPosteriors()




.. image:: /examples/Datapoints/images/sphx_glr_plot_frequency_datapoint_008.png
    :class: sphx-glr-single-img






.. code-block:: default

    plt.figure()
    _ = fdp.addErr.plotPosteriors()


.. image:: /examples/Datapoints/images/sphx_glr_plot_frequency_datapoint_009.png
    :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  10.177 seconds)


.. _sphx_glr_download_examples_Datapoints_plot_frequency_datapoint.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_frequency_datapoint.py <plot_frequency_datapoint.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_frequency_datapoint.ipynb <plot_frequency_datapoint.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
