.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_examples_Datapoints_plot_time_datapoint.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_Datapoints_plot_time_datapoint.py:


Time Domain Datapoint Class
---------------------------

There are three ways in which to create a time domain datapoint

1) :ref:`Instantiating a time domain datapoint`

2) :ref:`Reading a datapoint from a file`

3) :ref:`Obtaining a datapoint from a dataset`

Once instantiated, see :ref:`Using a time domain datapoint`


.. code-block:: default

    from os.path import join
    import numpy as np
    import h5py
    import matplotlib.pyplot as plt
    from geobipy import hdfRead
    from geobipy import Waveform
    from geobipy import SquareLoop, CircularLoop
    from geobipy import butterworth
    from geobipy import TdemSystem
    from geobipy import TdemData
    from geobipy import TdemDataPoint
    from geobipy import Model1D
    from geobipy import StatArray
    from geobipy import Distribution

    dataFolder = "..//supplementary//Data//"
    # dataFolder = "source//examples//supplementary//Data//"








Instantiating a time domain datapoint
+++++++++++++++++++++++++++++++++++++

In this first example, we define a ground based WalkTEM data point.

Ground time domain data are forward modelled using the `empymod package <https://empymod.readthedocs.io/en/stable/index.html>`_

Define some time gates


.. code-block:: default


    # Low moment
    lm_off_time = np.array([
        1.149E-05, 1.350E-05, 1.549E-05, 1.750E-05, 2.000E-05, 2.299E-05,
        2.649E-05, 3.099E-05, 3.700E-05, 4.450E-05, 5.350E-05, 6.499E-05,
        7.949E-05, 9.799E-05, 1.215E-04, 1.505E-04, 1.875E-04, 2.340E-04,
        2.920E-04, 3.655E-04, 4.580E-04, 5.745E-04, 7.210E-04
    ])

    # High moment
    hm_off_time = np.array([
        9.810e-05, 1.216e-04, 1.506e-04, 1.876e-04, 2.341e-04, 2.921e-04,
        3.656e-04, 4.581e-04, 5.746e-04, 7.211e-04, 9.056e-04, 1.138e-03,
        1.431e-03, 1.799e-03, 2.262e-03, 2.846e-03, 3.580e-03, 4.505e-03,
        5.670e-03, 7.135e-03
    ])








Define some observed data values for each time gate.


.. code-block:: default

    lm_data = np.array([
        7.980836E-06, 4.459270E-06, 2.909954E-06, 2.116353E-06, 1.571503E-06,
        1.205928E-06, 9.537814E-07, 7.538660E-07, 5.879494E-07, 4.572059E-07,
        3.561824E-07, 2.727531E-07, 2.058368E-07, 1.524225E-07, 1.107586E-07,
        7.963634E-08, 5.598970E-08, 3.867087E-08, 2.628711E-08, 1.746382E-08,
        1.136561E-08, 7.234771E-09, 4.503902E-09
    ])

    # High moment
    hm_data = np.array([
        1.563517e-07, 1.139461e-07, 8.231679e-08, 5.829438e-08, 4.068236e-08,
        2.804896e-08, 1.899818e-08, 1.268473e-08, 8.347439e-09, 5.420791e-09,
        3.473876e-09, 2.196246e-09, 1.372012e-09, 8.465165e-10, 5.155328e-10,
        3.099162e-10, 1.836829e-10, 1.072522e-10, 6.161256e-11, 3.478720e-11
    ])








Create a Waveform

The Waveform class defines a half waveform


.. code-block:: default

    lm_waveform = Waveform(time=np.r_[-1.041E-03, -9.850E-04, 0.000E+00, 4.000E-06],
                           amplitude=np.r_[0.0, 1.0, 1.0, 0.0],
                           current=1.0)
    hm_waveform = Waveform(time=np.r_[-8.333E-03, -8.033E-03, 0.000E+00, 5.600E-06],
                           amplitude=np.r_[0.0, 1.0, 1.0, 0.0],
                           current=1.0)

    plt.figure()
    lm_waveform.plot(label='Low Moment')
    hm_waveform.plot(label='High Moment', linestyle='-.')
    plt.legend()




.. image:: /examples/Datapoints/images/sphx_glr_plot_time_datapoint_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    <matplotlib.legend.Legend object at 0x1290982d0>



Define the transmitter and reciever loops


.. code-block:: default

    transmitter = SquareLoop(sideLength=40.0)
    receiver = CircularLoop()








Define two butterworth filters to be applied to the off-time data.


.. code-block:: default

    filters = [butterworth(1, 4.5e5, btype='low'), butterworth(1, 3.e5, btype='low')]








Create the time domain systems for both moments


.. code-block:: default

    lm_system = TdemSystem(offTimes=lm_off_time,
                           transmitterLoop=transmitter,
                           receiverLoop=receiver,
                           loopOffset=np.r_[0.0, 0.0, 0.0], # Centre loop sounding
                           waveform=lm_waveform,
                           offTimeFilters=filters)

    hm_system = TdemSystem(offTimes=hm_off_time,
                           transmitterLoop=transmitter,
                           receiverLoop=receiver,
                           loopOffset=np.r_[0.0, 0.0, 0.0], # Centre loop sounding
                           waveform=hm_waveform,
                           offTimeFilters=filters)

    systems = [lm_system, hm_system]








Instantiate the time domain datapoint


.. code-block:: default

    tdp = TdemDataPoint(x=0.0, y=0.0, z=0.0, elevation=0.0,
                        data=[lm_data, hm_data], std=None, predictedData=None,
                        system=systems, lineNumber=0.0, fiducial=0.0)









.. code-block:: default

    plt.figure()
    tdp.plot(with_error_bars=False)





.. image:: /examples/Datapoints/images/sphx_glr_plot_time_datapoint_002.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    <matplotlib.axes._subplots.AxesSubplot object at 0x129013590>



Reading a datapoint from a file
+++++++++++++++++++++++++++++++
We can read in time domain datapoints from individual datapoint files using the
AarhusInv data format.


.. code-block:: default

    tdp = TdemDataPoint()
    tdp.read([dataFolder+"//WalkTEM_LM.txt", dataFolder+"//WalkTEM_HM.txt"])









.. code-block:: default

    plt.figure()
    tdp.plot()




.. image:: /examples/Datapoints/images/sphx_glr_plot_time_datapoint_003.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    <matplotlib.axes._subplots.AxesSubplot object at 0x1296d1110>



Obtaining a datapoint from a dataset
++++++++++++++++++++++++++++++++++++
More often than not, our observed data is stored in a file on disk.
We can read in a dataset and pull datapoints from it.

For more information about the time domain data set, see :ref:`Time domain dataset`


.. code-block:: default


    # The data file name
    dataFile=[dataFolder + 'Skytem_High.txt', dataFolder + 'Skytem_Low.txt']
    # The EM system file name
    systemFile=[dataFolder + 'SkytemHM-SLV.stm', dataFolder + 'SkytemLM-SLV.stm']








Initialize and read an EM data set


.. code-block:: default

    D = TdemData()
    D.read(dataFile, systemFile)








Get a datapoint from the dataset


.. code-block:: default

    tdp = D.datapoint(0)
    plt.figure()
    tdp.plot()




.. image:: /examples/Datapoints/images/sphx_glr_plot_time_datapoint_004.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    <matplotlib.axes._subplots.AxesSubplot object at 0x130565d10>



Using a time domain datapoint
+++++++++++++++++++++++++++++

We can define a 1D layered earth model, and use it to predict some data


.. code-block:: default

    par = StatArray(np.r_[500.0, 20.0], "Conductivity", "$\frac{S}{m}$")
    mod = Model1D(depth=np.r_[75.0], parameters=par)








Forward model the data


.. code-block:: default

    tdp.forward(mod)









.. code-block:: default

    plt.figure()
    plt.subplot(121)
    _ = mod.pcolor()
    plt.subplot(122)
    _ = tdp.plot()
    _ = tdp.plotPredicted()
    plt.tight_layout()




.. image:: /examples/Datapoints/images/sphx_glr_plot_time_datapoint_005.png
    :class: sphx-glr-single-img






.. code-block:: default

    plt.figure()
    tdp.plotDataResidual(xscale='log', log=10)




.. image:: /examples/Datapoints/images/sphx_glr_plot_time_datapoint_006.png
    :class: sphx-glr-single-img





Compute the sensitivity matrix for a given model


.. code-block:: default

    J = tdp.sensitivity(mod)
    plt.figure()
    _ = np.abs(J).pcolor(equalize=True, log=10, flipY=True)




.. image:: /examples/Datapoints/images/sphx_glr_plot_time_datapoint_007.png
    :class: sphx-glr-single-img





Attaching statistical descriptors to the datapoint
++++++++++++++++++++++++++++++++++++++++++++++++++

Define a multivariate log normal distribution as the prior on the predicted data.


.. code-block:: default

    tdp.predictedData.setPrior('MvLogNormal', tdp.data[tdp.active], tdp.std[tdp.active]**2.0)








This allows us to evaluate the likelihood of the predicted data


.. code-block:: default

    print(tdp.likelihood(log=True))
    # Or the misfit
    print(tdp.dataMisfit())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    -19707.246512951275
    201.7577916092427




We can perform a quick search for the best fitting half space


.. code-block:: default

    halfspace = tdp.FindBestHalfSpace()
    print('Best half space conductivity is {} $S/m$'.format(halfspace.par))
    plt.figure()
    _ = tdp.plot()
    _ = tdp.plotPredicted()




.. image:: /examples/Datapoints/images/sphx_glr_plot_time_datapoint_008.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Best half space conductivity is [0.02009233] $S/m$




Compute the misfit between observed and predicted data


.. code-block:: default

    print(tdp.dataMisfit())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    452.4915367942427




Plot the misfits for a range of half space conductivities


.. code-block:: default

    plt.figure()
    _ = tdp.plotHalfSpaceResponses(-6.0, 4.0, 200)
    plt.title("Halfspace responses")




.. image:: /examples/Datapoints/images/sphx_glr_plot_time_datapoint_009.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    Text(0.5, 1.0, 'Halfspace responses')



We can attach priors to the height of the datapoint,
the relative error multiplier, and the additive error noise floor


.. code-block:: default


    # Set values of relative and additive error for both systems.
    tdp.relErr = [0.05, 0.05]
    tdp.addErr = [1e-11, 1e-12]

    # Define the distributions used as priors.
    heightPrior = Distribution('Uniform', min=np.float64(tdp.z) - 2.0, max=np.float64(tdp.z) + 2.0)
    relativePrior = Distribution('Uniform', min=np.r_[0.01, 0.01], max=np.r_[0.5, 0.5])
    additivePrior = Distribution('Uniform', min=np.r_[1e-12, 1e-13], max=np.r_[1e-10, 1e-11], log=True)
    tdp.setPriors(heightPrior, relativePrior, additivePrior)








In order to perturb our solvable parameters, we need to attach proposal distributions


.. code-block:: default

    heightProposal = Distribution('Normal', mean=tdp.z, variance = 0.01)
    relativeProposal = Distribution('MvNormal', mean=tdp.relErr, variance=2.5e-7)
    additiveProposal = Distribution('MvLogNormal', mean=tdp.addErr, variance=2.5e-3, linearSpace=True)
    tdp.setProposals(heightProposal, relativeProposal, additiveProposal)








With priorss set we can auto generate the posteriors


.. code-block:: default

    tdp.setPosteriors()








Perturb the datapoint and record the perturbations


.. code-block:: default

    for i in range(1000):
        tdp.perturb(True, True, True, False)
        tdp.updatePosteriors()








Plot the posterior distributions


.. code-block:: default

    plt.figure()
    _ = tdp.z.plotPosteriors()




.. image:: /examples/Datapoints/images/sphx_glr_plot_time_datapoint_010.png
    :class: sphx-glr-single-img






.. code-block:: default

    plt.figure()
    _ = tdp.relErr.plotPosteriors()




.. image:: /examples/Datapoints/images/sphx_glr_plot_time_datapoint_011.png
    :class: sphx-glr-single-img






.. code-block:: default

    plt.figure()
    _ = tdp.addErr.plotPosteriors()









.. image:: /examples/Datapoints/images/sphx_glr_plot_time_datapoint_012.png
    :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  8.762 seconds)


.. _sphx_glr_download_examples_Datapoints_plot_time_datapoint.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_time_datapoint.py <plot_time_datapoint.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_time_datapoint.ipynb <plot_time_datapoint.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
