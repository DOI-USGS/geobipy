.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_examples_Frequency_domain_data_frequency_domain_data_set.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_Frequency_domain_data_frequency_domain_data_set.py:


Frequency Domain Data Set
-------------------------


.. code-block:: default

    import matplotlib.pyplot as plt
    from geobipy import FdemData
    import numpy as np
    from os.path import join







Let's read in a frequency domain data set


.. code-block:: default


    dataFolder = "..//supplementary//Data//"

    # The data file name
    dataFile = dataFolder + 'Resolve2.txt'
    # The EM system file name
    systemFile = dataFolder + 'FdemSystem2.stm'

    FD1 = FdemData()
    FD1.read(dataFile, systemFile)








.. code-block:: default


    FD1.channelNames








.. code-block:: default


    FD1.getDataPoint(0)







Print out a small summary of the data


.. code-block:: default


    FD1.summary()





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






.. code-block:: default

    plt.figure()
    FD1.scatter2D()




.. image:: /examples/Frequency_domain_data/images/sphx_glr_frequency_domain_data_set_001.png
    :class: sphx-glr-single-img




Plot all the data along the specified line


.. code-block:: default


    plt.figure()
    ax = FD1.plotLine(30010.0, log=10)




.. image:: /examples/Frequency_domain_data/images/sphx_glr_frequency_domain_data_set_002.png
    :class: sphx-glr-single-img




Or, plot specific channels in the data


.. code-block:: default


    plt.figure()
    FD1.plot(channels=[0,11,8], log=10, linewidth=0.5);




.. image:: /examples/Frequency_domain_data/images/sphx_glr_frequency_domain_data_set_003.png
    :class: sphx-glr-single-img




Read in a second data set


.. code-block:: default



    FD2 = FdemData()
    FD2.read(dataFilename=dataFolder + 'Resolve1.txt', systemFilename=dataFolder + 'FdemSystem1.stm')





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Warning: Your data contains values that are <= 0.0



We can create maps of the elevations in two separate figures


.. code-block:: default


    plt.figure()
    FD1.mapPlot(dx=50.0, dy=50.0, mask = 200.0, method='ct');plt.axis('equal')




.. image:: /examples/Frequency_domain_data/images/sphx_glr_frequency_domain_data_set_004.png
    :class: sphx-glr-single-img





.. code-block:: default


    plt.figure()
    FD2.mapPlot(dx=50.0, dy=50.0, mask = 200.0, method = 'ct');plt.axis('equal');




.. image:: /examples/Frequency_domain_data/images/sphx_glr_frequency_domain_data_set_005.png
    :class: sphx-glr-single-img




Or, we can plot both data sets in one figure to see their positions relative
to each other.

In this case, I use a 2D scatter plot of the data point co-ordinates, and pass
one of the channels as the colour.


.. code-block:: default


    plt.figure()
    FD1.scatter2D(s=1.0, c=FD1.getDataChannel(0))
    FD2.scatter2D(s=1.0, c=FD2.getDataChannel(0), cmap='jet');




.. image:: /examples/Frequency_domain_data/images/sphx_glr_frequency_domain_data_set_006.png
    :class: sphx-glr-single-img




Or, I can interpolate the values to create a gridded "map". mapChannel will
interpolate the specified channel number.


.. code-block:: default


    plt.figure()
    FD1.mapData(3, system=0, method='ct', dx=200, dy=200, mask=250)
    plt.axis('equal');




.. image:: /examples/Frequency_domain_data/images/sphx_glr_frequency_domain_data_set_007.png
    :class: sphx-glr-single-img




Export the data to VTK


.. code-block:: default


    # FD1.toVTK('FD_one')
    # FD2.toVTK('FD_two')







We can get a specific line from the data set


.. code-block:: default


    print(np.unique(FD1.line))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [30010. 30020. 30030. ... 30100. 39010. 39020.]




.. code-block:: default

    L = FD1.getLine(30010.0)







A summary will now show the properties of the line.


.. code-block:: default


    L.summary()





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    3D Point Cloud: 
    Number of Points: : 6710 
     Name: Easting
         Units: m
         Shape: (6710,)
         Values: [586852.29 586852.23 586852.17 ... 586123.57 586123.2  586122.82]
     Name: Northing
         Units: m
         Shape: (6710,)
         Values: [4639119.38 4639122.68 4639125.98 ... 4661765.26 4661768.84 4661772.42]
     Name: Height
         Units: m
         Shape: (6710,)
         Values: [36.629 37.012 37.349 ... 28.313 28.218 28.115]
     Name: Elevation
         Units: m
         Shape: (6710,)
         Values: [1246.84 1246.71 1246.61 ... 1337.94 1337.96 1338.02]
    Data:          : 
    # of Channels: 12 
    # of Total Data: 80520 
    Name: Fdem Data
         Units: ppm
         Shape: (6710, 12)
         Values: [[145.3 435.8 260.6 ... 516.5 405.7 255.7]
     [145.7 436.5 257.9 ... 513.6 403.2 252. ]
     [146.4 437.4 255.8 ... 511.2 400.9 248.8]
     ...
     [ 70.7 314.1 220.2 ... 743.3 960.8 910.7]
     [ 71.3 315.3 220.5 ... 745.9 968.3 919.1]
     [ 72.1 316.6 220.7 ... 749.2 976.5 928.3]]

     Name: Standard Deviation
         Units: ppm
         Shape: (6710, 12)
         Values: [[14.53 43.58 26.06 ... 51.65 40.57 25.57]
     [14.57 43.65 25.79 ... 51.36 40.32 25.2 ]
     [14.64 43.74 25.58 ... 51.12 40.09 24.88]
     ...
     [ 7.07 31.41 22.02 ... 74.33 96.08 91.07]
     [ 7.13 31.53 22.05 ... 74.59 96.83 91.91]
     [ 7.21 31.66 22.07 ... 74.92 97.65 92.83]]

     Name: Predicted Data
         Units: ppm
         Shape: (6710, 12)
         Values: [[0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     ...
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]]





And we can scatter2D the points in the line.


.. code-block:: default


    plt.figure()
    L.scatter2D()




.. image:: /examples/Frequency_domain_data/images/sphx_glr_frequency_domain_data_set_008.png
    :class: sphx-glr-single-img





.. code-block:: default


    plt.figure()
    L.plot(xAxis='r2d', log=10)



.. image:: /examples/Frequency_domain_data/images/sphx_glr_frequency_domain_data_set_009.png
    :class: sphx-glr-single-img





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  14.566 seconds)


.. _sphx_glr_download_examples_Frequency_domain_data_frequency_domain_data_set.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: frequency_domain_data_set.py <frequency_domain_data_set.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: frequency_domain_data_set.ipynb <frequency_domain_data_set.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
