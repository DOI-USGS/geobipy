.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_examples_Frequency_domain_data_fdem_data_point.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_Frequency_domain_data_fdem_data_point.py:


Fdem Data Point Class
---------------------

Fdem Data contains entire data sets

Fdem Data Points can forward model and evaluate themselves


.. code-block:: default


    from os.path import join
    import numpy as np
    import h5py
    import matplotlib.pyplot as plt
    from geobipy import hdfRead
    from geobipy import FdemData
    from geobipy import FdemDataPoint
    from geobipy import Model1D
    from geobipy import StatArray








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







Summarize the Data


.. code-block:: default

    print(D.__doc__)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Class extension to geobipy.Data defining a Fourier domain electro magnetic data set
    
        FdemData(nPoints, nFrequencies, system)

        Parameters
        ----------
        nPoints : int, optional
            Number of observations in the data set
        nFrequencies : int, optional
            Number of measurement frequencies
        system : str or geobipy.FdemSystem, optional
            * If str: Must be a file name from which to read FD system information.
            * If FdemSystem: A deepcopy is made.

        Returns
        -------
        out : FdemData
            Contains x, y, z, elevation, and data values for a frequency domain dataset.

        Notes
        -----
        FdemData.read() requires a data filename and a system class or system filename to be specified.
        The data file is structured using columns with the first line containing header information.
        The header should contain the following entries
        Line [ID or FID] [X or N or northing] [Y or E or easting] [Z or DTM or dem_elev] [Alt or Laser or bheight] [I Q] ... [I Q] 
        Do not include brackets []
        [I Q] are the in-phase and quadrature values for each measurement frequency.

        If a system filename is given, it too is structured using columns with the first line containing header information
        Each subsequent row contains the information for each measurement frequency

        freq  tor  tmom  tx ty tz ror rmom  rx   ry rz
        378   z    1     0  0  0  z   1     7.93 0  0
        1776  z    1     0  0  0  z   1     7.91 0  0
        ...

        where tor and ror are the orientations of the transmitter/reciever loops [x or z].
        tmom and rmom are the moments of the loops.
        t/rx,y,z are the loop offsets from the observation locations in the data file.

    




.. code-block:: default

    D.summary()





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    3D Point Cloud: 
    Number of Points: : 71470 
     Name: Easting
         Units: m
         Shape: (71470,)
         Values: [586852.29 586852.23 586852.17 ... 590160.46 590163.5  590166.53]
     Name: Northing
         Units: m
         Shape: (71470,)
         Values: [4639119.38 4639122.68 4639125.98 ... 4640082.67 4640082.8  4640082.93]
     Name: Height
         Units: m
         Shape: (71470,)
         Values: [36.629 37.012 37.349 ... 33.123 33.021 32.917]
     Name: Elevation
         Units: m
         Shape: (71470,)
         Values: [1246.84 1246.71 1246.61 ... 1247.36 1247.41 1247.46]
    Data:          : 
    # of Channels: 12 
    # of Total Data: 857640 
    Name: Fdem Data
         Units: ppm
         Shape: (71470, 12)
         Values: [[145.3 435.8 260.6 ... 516.5 405.7 255.7]
     [145.7 436.5 257.9 ... 513.6 403.2 252. ]
     [146.4 437.4 255.8 ... 511.2 400.9 248.8]
     ...
     [173.8 511.6 308.6 ... 660.8 638.7 374.7]
     [172.3 513.7 310.  ... 664.8 643.9 378.7]
     [170.4 515.8 311.3 ... 669.1 650.  383.4]]

     Name: Standard Deviation
         Units: ppm
         Shape: (71470, 12)
         Values: [[14.53 43.58 26.06 ... 51.65 40.57 25.57]
     [14.57 43.65 25.79 ... 51.36 40.32 25.2 ]
     [14.64 43.74 25.58 ... 51.12 40.09 24.88]
     ...
     [17.38 51.16 30.86 ... 66.08 63.87 37.47]
     [17.23 51.37 31.   ... 66.48 64.39 37.87]
     [17.04 51.58 31.13 ... 66.91 65.   38.34]]

     Name: Predicted Data
         Units: ppm
         Shape: (71470, 12)
         Values: [[0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     ...
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]]





Grab a measurement from the data set


.. code-block:: default

    P = D.getDataPoint(0)
    P.system[0].summary()
    P.summary()
    plt.figure()
    P.plot()




.. image:: /examples/Frequency_domain_data/images/sphx_glr_fdem_data_point_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    FdemSystem: 
    None 
    Name: Frequencies
         Units: Hz
         Shape: (6,)
         Values: [   380.   1776.   3345.   8171.  41020. 129550.]
 
    Name: Loop Separations
         Units: m
         Shape: (6,)
         Values: [7.93 7.91 9.03 7.91 7.91 7.89]
 

    Data Point: 
    Channel Names ['In-Phase 380.0 (Hz)', 'In-Phase 1776.0 (Hz)', 'In-Phase 3345.0 (Hz)', 'In-Phase 8171.0 (Hz)', 'In-Phase 41020.0 (Hz)', 'In-Phase 129550.0 (Hz)', 'Quadrature 380.0 (Hz)', 'Quadrature 1776.0 (Hz)', 'Quadrature 3345.0 (Hz)', 'Quadrature 8171.0 (Hz)', 'Quadrature 41020.0 (Hz)', 'Quadrature 129550.0 (Hz)'] 
    x: [586852.29] 
    y: [4639119.38] 
    z: [36.629] 
    elevation: [1246.84] 
    Number of active channels: 12 
    Name: Frequency domain data
         Units: ppm
         Shape: (12,)
         Values: [145.3 435.8 260.6 ... 516.5 405.7 255.7]
     Name: Predicted Data
         Units: ppm
         Shape: (12,)
         Values: [0. 0. 0. ... 0. 0. 0.]
     Name: Standard Deviation
         Units: ppm
         Shape: (12,)
         Values: [14.53 43.58 26.06 ... 51.65 40.57 25.57]
 
    Line number: 30010.0 
    Fiducial: 30000.0
    Relative Error Name: $\epsilon_{Relative}x10^{2}$
         Units: %
         Shape: (1,)
         Values: [0.]

    Additive Error Name: $\epsilon_{Additive}$
         Units: ppm
         Shape: (1,)
         Values: [0.]

    FdemSystem: 
    None 
    Name: Frequencies
         Units: Hz
         Shape: (6,)
         Values: [   380.   1776.   3345.   8171.  41020. 129550.]
 
    Name: Loop Separations
         Units: m
         Shape: (6,)
         Values: [7.93 7.91 9.03 7.91 7.91 7.89]
 




We can forward model the EM response of a 1D layered earth <a href="../Model/Model1D.ipynb">Model1D</a>


.. code-block:: default


    nCells = 19
    par = StatArray(np.linspace(0.01, 0.1, nCells), "Conductivity", "$\frac{S}{m}$")
    thk = StatArray(np.ones(nCells-1) * 10.0)
    mod = Model1D(nCells = nCells, parameters=par, thickness=thk)
    mod.summary()
    plt.figure()
    mod.pcolor(grid=True)




.. image:: /examples/Frequency_domain_data/images/sphx_glr_fdem_data_point_002.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    1D Model: 
    Name: # of Cells
         Units: 
         Shape: (1,)
         Values: [19]
    Top of the model: [0.]
    Name: Thickness
         Units: m
         Shape: (19,)
         Values: [10. 10. 10. ... 10. 10. inf]
    Name: Conductivity
         Units: $\frac{S}{m}$
         Shape: (19,)
         Values: [0.01  0.015 0.02  ... 0.09  0.095 0.1  ]
    Name: Depth
         Units: m
         Shape: (19,)
         Values: [ 10.  20.  30. ... 170. 180.  inf]




Compute and plot the data from the model


.. code-block:: default

    P.forward(mod)
    plt.figure()
    P.plot()
    P.plotPredicted()





.. image:: /examples/Frequency_domain_data/images/sphx_glr_fdem_data_point_003.png
    :class: sphx-glr-single-img





.. code-block:: default


    # Set the Prior
    addErrors = StatArray(np.full(2*P.nFrequencies, 10.0))
    P.predictedData.setPrior('MVNormalLog', addErrors, addErrors)
    P.updateErrors(0.05, addErrors[:])







With forward modelling, we can solve for the best fitting halfspace model


.. code-block:: default


    HSconductivity=P.FindBestHalfSpace()
    print('Best half space conductivity is ', HSconductivity, ' $S/m$')
    plt.figure()
    P.plot()
    P.plotPredicted()




.. image:: /examples/Frequency_domain_data/images/sphx_glr_fdem_data_point_004.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Best half space conductivity is  0.1036632928437698  $S/m$



Compute the misfit between observed and predicted data


.. code-block:: default


    print(P.dataMisfit())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    11.275909256512264



Plot the misfits for a range of half space conductivities


.. code-block:: default


    plt.figure()
    P.plotHalfSpaceResponses(-6.0,4.0,200)




.. image:: /examples/Frequency_domain_data/images/sphx_glr_fdem_data_point_005.png
    :class: sphx-glr-single-img




Compute the sensitivity matrix for a given model


.. code-block:: default


    J = P.sensitivity(mod)
    plt.figure()
    np.abs(J).pcolor(equalize=True, log=10);




.. image:: /examples/Frequency_domain_data/images/sphx_glr_fdem_data_point_006.png
    :class: sphx-glr-single-img




We can save the FdemDataPoint to a HDF file


.. code-block:: default


    with h5py.File('FdemDataPoint.h5','w') as hf:
        P.createHdf(hf, 'fdp')
        P.writeHdf(hf, 'fdp')







And then read it in


.. code-block:: default


    P1=hdfRead.readKeyFromFiles('FdemDataPoint.h5','/','fdp')







.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  1.877 seconds)


.. _sphx_glr_download_examples_Frequency_domain_data_fdem_data_point.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: fdem_data_point.py <fdem_data_point.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: fdem_data_point.ipynb <fdem_data_point.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
