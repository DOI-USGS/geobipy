.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_examples_Data_time_domain_datapoint.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_Data_time_domain_datapoint.py:


Time Domain Datapoint Class
---------------------------

Tdem Data contains entire data sets

Tdem Data Points can forward model and evaluate themselves


.. code-block:: default


    from os.path import join
    import numpy as np
    import h5py
    import matplotlib.pyplot as plt
    from geobipy import hdfRead
    from geobipy import TdemData
    from geobipy import TdemDataPoint
    from geobipy import Model1D
    from geobipy import StatArray









.. code-block:: default


    dataFolder = "..//supplementary//Data//"

    # The data file name
    dataFile=[dataFolder + 'Skytem_High.txt', dataFolder + 'Skytem_Low.txt']
    # The EM system file name
    systemFile=[dataFolder + 'SkytemHM-SLV.stm', dataFolder + 'SkytemLM-SLV.stm']







Initialize and read an EM data set


.. code-block:: default


    D = TdemData()
    D.read(dataFile, systemFile)








Summarize the Data

Grab a measurement from the data set


.. code-block:: default



    P = D.getDataPoint(0)
    P._std[:] = 1e-12
    P.summary()
    plt.figure()
    P.plot()




.. image:: /examples/Data/images/sphx_glr_time_domain_datapoint_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Data Point: 
    Channel Names ['Time 2.450e-06 s', 'Time 6.095e-06 s', 'Time 8.095e-06 s', 'Time 1.010e-05 s', 'Time 1.210e-05 s', 'Time 1.409e-05 s', 'Time 1.610e-05 s', 'Time 1.909e-05 s', 'Time 2.360e-05 s', 'Time 2.960e-05 s', 'Time 3.709e-05 s', 'Time 4.609e-05 s', 'Time 5.759e-05 s', 'Time 7.209e-05 s', 'Time 9.060e-05 s', 'Time 1.141e-04 s', 'Time 1.431e-04 s', 'Time 1.801e-04 s', 'Time 2.266e-04 s', 'Time 2.846e-04 s', 'Time 3.581e-04 s', 'Time 4.506e-04 s', 'Time 5.671e-04 s', 'Time 7.136e-04 s', 'Time 8.981e-04 s', 'Time 1.131e-03 s', 'Time 1.423e-03 s', 'Time 1.791e-03 s', 'Time 2.255e-03 s', 'Time 2.838e-03 s', 'Time 3.573e-03 s', 'Time 4.498e-03 s', 'Time 5.662e-03 s', 'Time 7.128e-03 s', 'Time 2.450e-06 s', 'Time 6.095e-06 s', 'Time 8.095e-06 s', 'Time 1.010e-05 s', 'Time 1.210e-05 s', 'Time 1.409e-05 s', 'Time 1.610e-05 s', 'Time 1.909e-05 s', 'Time 2.360e-05 s', 'Time 2.960e-05 s', 'Time 3.709e-05 s', 'Time 4.609e-05 s', 'Time 5.759e-05 s', 'Time 7.209e-05 s', 'Time 9.060e-05 s', 'Time 1.141e-04 s', 'Time 1.431e-04 s', 'Time 1.801e-04 s', 'Time 2.266e-04 s', 'Time 2.846e-04 s', 'Time 3.581e-04 s', 'Time 4.506e-04 s', 'Time 5.671e-04 s', 'Time 7.136e-04 s', 'Time 8.981e-04 s', 'Time 1.131e-03 s'] 
    x: [4194346.5] 
    y: [431112.2] 
    z: [36.7] 
    elevation: [2304.] 
    Number of active channels: 26 
    Name: Time domain data
         Units: $\frac{V}{Am^{4}}$
         Shape: (26,)
         Values: [5.5959e-11 3.5732e-11 2.3111e-11 ... 2.6904e-11 1.8231e-11 1.3699e-11]
     Name: Predicted Data
         Units: $\frac{V}{Am^{4}}$
         Shape: (26,)
         Values: [0. 0. 0. ... 0. 0. 0.]
     Name: Standard Deviation
         Units: $\frac{V}{Am^{4}}$
         Shape: (26,)
         Values: [1.e-12 1.e-12 1.e-12 ... 1.e-12 1.e-12 1.e-12]
 
    Line number: 100101.0 
    Fiducial: 154.0
    Relative Error Name: $\epsilon_{Relative}x10^{2}$
         Units: %
         Shape: (2,)
         Values: [0. 0.]

    Additive Error Name: $\epsilon_{Additive}$
         Units: $\frac{V}{Am^{4}}$
         Shape: (2,)
         Values: [0. 0.]

    TdemSystem: 
    ..//supplementary//Data//SkytemHM-SLV.stm
    Name: Time
         Units: s
         Shape: (34,)
         Values: [2.4500e-06 6.0950e-06 8.0950e-06 ... 4.4976e-03 5.6621e-03 7.1276e-03]

    TdemSystem: 
    ..//supplementary//Data//SkytemLM-SLV.stm
    Name: Time
         Units: s
         Shape: (26,)
         Values: [2.45000e-06 6.09500e-06 8.09500e-06 ... 7.13595e-04 8.98095e-04
     1.13060e-03]





We can forward model the EM response of a 1D layered earth <a href="../Model/Model1D.ipynb">Model1D</a>


.. code-block:: default


    par = StatArray(np.linspace(0.01, 0.1, 19), "Conductivity", "$\\frac{S}{m}$")
    thk = StatArray(np.ones(18) * 10.0)
    mod = Model1D(nCells = 19, parameters=par, thickness=thk)
    plt.figure()
    mod.pcolor(grid=True)





.. image:: /examples/Data/images/sphx_glr_time_domain_datapoint_002.png
    :class: sphx-glr-single-img




Compute and plot the data from the model


.. code-block:: default


    mod = Model1D(depth=np.asarray([125]), parameters=np.asarray([0.00327455, 0.00327455]))
    mod.summary()






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    1D Model: 
    Name: # of Cells
         Units: 
         Shape: (1,)
         Values: [2]
    Top of the model: [0.]
    Name: Thickness
         Units: m
         Shape: (2,)
         Values: [125.  inf]
    Name: 
         Units: 
         Shape: (2,)
         Values: [0.00327455 0.00327455]
    Name: Depth
         Units: m
         Shape: (2,)
         Values: [125.  inf]





.. code-block:: default



    P.forward(mod)
    plt.figure()
    P.plot()
    P.plotPredicted()





.. image:: /examples/Data/images/sphx_glr_time_domain_datapoint_003.png
    :class: sphx-glr-single-img





.. code-block:: default



    P.summary()






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Data Point: 
    Channel Names ['Time 2.450e-06 s', 'Time 6.095e-06 s', 'Time 8.095e-06 s', 'Time 1.010e-05 s', 'Time 1.210e-05 s', 'Time 1.409e-05 s', 'Time 1.610e-05 s', 'Time 1.909e-05 s', 'Time 2.360e-05 s', 'Time 2.960e-05 s', 'Time 3.709e-05 s', 'Time 4.609e-05 s', 'Time 5.759e-05 s', 'Time 7.209e-05 s', 'Time 9.060e-05 s', 'Time 1.141e-04 s', 'Time 1.431e-04 s', 'Time 1.801e-04 s', 'Time 2.266e-04 s', 'Time 2.846e-04 s', 'Time 3.581e-04 s', 'Time 4.506e-04 s', 'Time 5.671e-04 s', 'Time 7.136e-04 s', 'Time 8.981e-04 s', 'Time 1.131e-03 s', 'Time 1.423e-03 s', 'Time 1.791e-03 s', 'Time 2.255e-03 s', 'Time 2.838e-03 s', 'Time 3.573e-03 s', 'Time 4.498e-03 s', 'Time 5.662e-03 s', 'Time 7.128e-03 s', 'Time 2.450e-06 s', 'Time 6.095e-06 s', 'Time 8.095e-06 s', 'Time 1.010e-05 s', 'Time 1.210e-05 s', 'Time 1.409e-05 s', 'Time 1.610e-05 s', 'Time 1.909e-05 s', 'Time 2.360e-05 s', 'Time 2.960e-05 s', 'Time 3.709e-05 s', 'Time 4.609e-05 s', 'Time 5.759e-05 s', 'Time 7.209e-05 s', 'Time 9.060e-05 s', 'Time 1.141e-04 s', 'Time 1.431e-04 s', 'Time 1.801e-04 s', 'Time 2.266e-04 s', 'Time 2.846e-04 s', 'Time 3.581e-04 s', 'Time 4.506e-04 s', 'Time 5.671e-04 s', 'Time 7.136e-04 s', 'Time 8.981e-04 s', 'Time 1.131e-03 s'] 
    x: [4194346.5] 
    y: [431112.2] 
    z: [36.7] 
    elevation: [2304.] 
    Number of active channels: 26 
    Name: Time domain data
         Units: $\frac{V}{Am^{4}}$
         Shape: (26,)
         Values: [5.5959e-11 3.5732e-11 2.3111e-11 ... 2.6904e-11 1.8231e-11 1.3699e-11]
     Name: Predicted Data
         Units: $\frac{V}{Am^{4}}$
         Shape: (26,)
         Values: [6.91315830e-12 3.56896118e-12 2.07395932e-12 ... 2.73078129e-12
     1.57515387e-12 8.93202913e-13]
     Name: Standard Deviation
         Units: $\frac{V}{Am^{4}}$
         Shape: (26,)
         Values: [1.e-12 1.e-12 1.e-12 ... 1.e-12 1.e-12 1.e-12]
 
    Line number: 100101.0 
    Fiducial: 154.0
    Relative Error Name: $\epsilon_{Relative}x10^{2}$
         Units: %
         Shape: (2,)
         Values: [0. 0.]

    Additive Error Name: $\epsilon_{Additive}$
         Units: $\frac{V}{Am^{4}}$
         Shape: (2,)
         Values: [0. 0.]

    TdemSystem: 
    ..//supplementary//Data//SkytemHM-SLV.stm
    Name: Time
         Units: s
         Shape: (34,)
         Values: [2.4500e-06 6.0950e-06 8.0950e-06 ... 4.4976e-03 5.6621e-03 7.1276e-03]

    TdemSystem: 
    ..//supplementary//Data//SkytemLM-SLV.stm
    Name: Time
         Units: s
         Shape: (26,)
         Values: [2.45000e-06 6.09500e-06 8.09500e-06 ... 7.13595e-04 8.98095e-04
     1.13060e-03]






.. code-block:: default



    plt.figure()
    P.plotDataResidual(xscale='log', log=10)





.. image:: /examples/Data/images/sphx_glr_time_domain_datapoint_004.png
    :class: sphx-glr-single-img




The errors are set to zero right now, so lets change that


.. code-block:: default


    # Set the Prior
    P._predictedData.setPrior('MVNormalLog' ,P._data[P.iActive], P._std[P.iActive]**2.0)
    P.updateErrors(relativeErr=[0.05, 0.05], additiveErr=[1.0e-12, 1.0e-13])







With forward modelling, we can solve for the best fitting halfspace model


.. code-block:: default


    HSconductivity=P.FindBestHalfSpace()
    print(HSconductivity)
    plt.figure()
    P.plot(withErrorBars=True)
    P.plotPredicted()





.. image:: /examples/Data/images/sphx_glr_time_domain_datapoint_005.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    0.01747528400007685




.. code-block:: default



    plt.figure()
    P.plotDataResidual(xscale='log', log=10)





.. image:: /examples/Data/images/sphx_glr_time_domain_datapoint_006.png
    :class: sphx-glr-single-img




Compute the misfit between observed and predicted data


.. code-block:: default


    print(P.dataMisfit())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    102.2179982665892



Plot the misfits for a range of half space conductivities


.. code-block:: default


    plt.figure()
    P.plotHalfSpaceResponses(-6.0,4.0,200)





.. image:: /examples/Data/images/sphx_glr_time_domain_datapoint_007.png
    :class: sphx-glr-single-img




Compute the sensitivity matrix for a given model


.. code-block:: default


    sensitivityMatrix = P.sensitivity(mod)
    J = StatArray(np.abs(sensitivityMatrix),'|Sensitivity|')
    plt.figure()
    J.pcolor(grid=True, log=10, equalize=True, linewidth=1)





.. image:: /examples/Data/images/sphx_glr_time_domain_datapoint_008.png
    :class: sphx-glr-single-img





.. code-block:: default



    sensitivityMatrix = P.sensitivity(mod)







We can save the FdemDataPoint to a HDF file


.. code-block:: default


    with h5py.File('TdemDataPoint.h5','w') as hf:
        P.createHdf(hf, 'tdp')
        P.writeHdf(hf, 'tdp')









And then read it in


.. code-block:: default


    # P1 = hdfRead.readKeyFromFiles('TdemDataPoint.h5','/','tdp', sysPath=join('supplementary','Data'))









.. code-block:: default



    # P1.summary()







.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  5.553 seconds)


.. _sphx_glr_download_examples_Data_time_domain_datapoint.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: time_domain_datapoint.py <time_domain_datapoint.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: time_domain_datapoint.ipynb <time_domain_datapoint.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
