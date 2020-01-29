.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_examples_Datapoints_plot_frequency_datapoint.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_Datapoints_plot_frequency_datapoint.py:


Frequency domain datapoint
--------------------------

There are two ways in which to create a datapoint,

1) :ref:`Instantiating a frequency domain data point`

2) :ref:`Obtaining a datapoint from a dataset`


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


.. code-block:: default

    fdp = FdemDataPoint(x=0.0, y=0.0, z=30, elevation=0.0, 
                        data=None, std=None, predictedData=None, 
                        system=fds, lineNumber=0.0, fiducial=0.0)








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

    P = D.datapoint(0)
    plt.figure()
    _ = P.plot()




.. image:: /examples/Datapoints/images/sphx_glr_plot_frequency_datapoint_002.png
    :class: sphx-glr-single-img





Predict data using the same model as before


.. code-block:: default

    P.forward(mod)
    plt.figure()
    _ = P.plot()
    _ = P.plotPredicted()
    plt.tight_layout();




.. image:: /examples/Datapoints/images/sphx_glr_plot_frequency_datapoint_003.png
    :class: sphx-glr-single-img





Attaching statistical descriptors to the datapoint
++++++++++++++++++++++++++++++++++++++++++++++++++

Define a multivariate log normal distribution as the prior on the predicted data.


.. code-block:: default

    P.predictedData.setPrior('MvLogNormal', P.data[P.active], P.std[P.active]**2.0)








This allows us to evaluate the likelihood of the predicted data


.. code-block:: default

    print(P.likelihood(log=True))
    # Or the misfit
    print(P.dataMisfit())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    -316.19118049547296
    22.819882232476




We can perform a quick search for the best fitting half space


.. code-block:: default

    halfspace = P.FindBestHalfSpace()
    print('Best half space conductivity is {} $S/m$')
    plt.figure()
    P.plot()
    P.plotPredicted()




.. image:: /examples/Datapoints/images/sphx_glr_plot_frequency_datapoint_004.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Best half space conductivity is {} $S/m$

    <matplotlib.axes._subplots.AxesSubplot object at 0x127016dd0>



Compute the misfit between observed and predicted data


.. code-block:: default

    print(P.dataMisfit())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    26.18321050483556




Plot the misfits for a range of half space conductivities


.. code-block:: default

    plt.figure()
    _ = P.plotHalfSpaceResponses(-6.0, 4.0, 200)
    plt.title("Halfspace responses")




.. image:: /examples/Datapoints/images/sphx_glr_plot_frequency_datapoint_005.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    Text(0.5, 1.0, 'Halfspace responses')



Compute the sensitivity matrix for a given model


.. code-block:: default


    J = P.sensitivity(mod)
    plt.figure()
    np.abs(J).pcolor(equalize=True, log=10);

    # ################################################################################
    # # We can save the FdemDataPoint to a HDF file

    # with h5py.File('FdemDataPoint.h5','w') as hf:
    #     P.createHdf(hf, 'fdp')
    #     P.writeHdf(hf, 'fdp')

    # ################################################################################
    # # And then read it in

    # P1=hdfRead.readKeyFromFiles('FdemDataPoint.h5','/','fdp')



.. image:: /examples/Datapoints/images/sphx_glr_plot_frequency_datapoint_006.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    (<matplotlib.axes._subplots.AxesSubplot object at 0x1270b2750>, <matplotlib.collections.QuadMesh object at 0x1288d7ad0>, <matplotlib.colorbar.Colorbar object at 0x1242b9150>)




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  8.078 seconds)


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
