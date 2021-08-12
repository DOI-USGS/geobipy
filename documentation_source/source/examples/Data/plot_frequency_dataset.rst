.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_examples_Data_plot_frequency_dataset.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_examples_Data_plot_frequency_dataset.py:


Frequency domain dataset
------------------------


.. code-block:: default

    import matplotlib.pyplot as plt
    from geobipy import CircularLoop
    from geobipy import FdemSystem
    from geobipy import FdemData
    import numpy as np









Defining data using a frequency domain system
+++++++++++++++++++++++++++++++++++++++++++++

We can start by defining the frequencies, transmitter loops, and receiver loops
For each frequency we need to define a pair of loops


.. code-block:: default

    frequencies = np.asarray([395.0, 822.0, 3263.0, 8199.0, 38760.0, 128755.0])








Transmitter positions are defined relative to the observation locations in the data
This is usually a constant offset for all data points.


.. code-block:: default

    transmitters = [CircularLoop(orient="z", moment=1.0,  x=0.0, y=0.0, z=0.0, pitch=0.0, roll=0.0, yaw=0.0, radius=1.0),
                    CircularLoop(orient="z", moment=1.0,  x=0.0, y=0.0, z=0.0, pitch=0.0, roll=0.0, yaw=0.0, radius=1.0),
                    CircularLoop(orient="x", moment=-1.0, x=0.0, y=0.0, z=0.0, pitch=0.0, roll=0.0, yaw=0.0, radius=1.0),
                    CircularLoop(orient="z", moment=1.0,  x=0.0, y=0.0, z=0.0, pitch=0.0, roll=0.0, yaw=0.0, radius=1.0),
                    CircularLoop(orient="z", moment=1.0,  x=0.0, y=0.0, z=0.0, pitch=0.0, roll=0.0, yaw=0.0, radius=1.0),
                    CircularLoop(orient="z", moment=1.0,  x=0.0, y=0.0, z=0.0, pitch=0.0, roll=0.0, yaw=0.0, radius=1.0)]








Receiver positions are defined relative to the transmitter


.. code-block:: default

    receivers = [CircularLoop(orient="z", moment=1.0, x=7.93, y=0.0, z=0.0, pitch=0.0, roll=0.0, yaw=0.0, radius=1.0),
                 CircularLoop(orient="z", moment=1.0, x=7.91, y=0.0, z=0.0, pitch=0.0, roll=0.0, yaw=0.0, radius=1.0),
                 CircularLoop(orient="x", moment=1.0, x=9.03, y=0.0, z=0.0, pitch=0.0, roll=0.0, yaw=0.0, radius=1.0),
                 CircularLoop(orient="z", moment=1.0, x=7.91, y=0.0, z=0.0, pitch=0.0, roll=0.0, yaw=0.0, radius=1.0),
                 CircularLoop(orient="z", moment=1.0, x=7.91, y=0.0, z=0.0, pitch=0.0, roll=0.0, yaw=0.0, radius=1.0),
                 CircularLoop(orient="z", moment=1.0, x=7.89, y=0.0, z=0.0, pitch=0.0, roll=0.0, yaw=0.0, radius=1.0)]








Instantiate the system for the data


.. code-block:: default

    system = FdemSystem(frequencies=frequencies, transmitterLoops=transmitters, receiverLoops=receivers)








Create some data with random co-ordinates


.. code-block:: default

    x = np.random.randn(100)
    y = np.random.randn(100)
    z = np.random.randn(100)

    data = FdemData(x=x, y=-y, z=z, systems = system)








Reading in the Data
+++++++++++++++++++
Of course measured field data is stored on disk. So instead we can read data from file.


.. code-block:: default

    dataFolder = "..//supplementary//Data//"
    # The data file name
    dataFile = dataFolder + 'Resolve2.txt'
    # The EM system file name
    systemFile = dataFolder + 'FdemSystem2.stm'








Read in a data set from file.


.. code-block:: default

    FD1 = FdemData().read(dataFile, systemFile)








Take a look at the channel names


.. code-block:: default

    for name in FD1.channelNames:
        print(name)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    In-Phase 380.0 (Hz)
    In-Phase 1776.0 (Hz)
    In-Phase 3345.0 (Hz)
    In-Phase 8171.0 (Hz)
    In-Phase 41020.0 (Hz)
    In-Phase 129550.0 (Hz)
    Quadrature 380.0 (Hz)
    Quadrature 1776.0 (Hz)
    Quadrature 3345.0 (Hz)
    Quadrature 8171.0 (Hz)
    Quadrature 41020.0 (Hz)
    Quadrature 129550.0 (Hz)




Get data points by slicing


.. code-block:: default

    FDa = FD1[10:]
    FD1 = FD1[:10]








Append data sets together


.. code-block:: default

    FD1.append(FDa)









Plot the locations of the data points


.. code-block:: default

    plt.figure(figsize=(8,6))
    _ = FD1.scatter2D();




.. image:: /examples/Data/images/sphx_glr_plot_frequency_dataset_001.png
    :alt: plot frequency dataset
    :class: sphx-glr-single-img





Plot all the data along the specified line


.. code-block:: default

    plt.figure(figsize=(8,6))
    _ = FD1.plotLine(30010.0, log=10);




.. image:: /examples/Data/images/sphx_glr_plot_frequency_dataset_002.png
    :alt: Line number 30010.0
    :class: sphx-glr-single-img





Or, plot specific channels in the data


.. code-block:: default

    plt.figure(figsize=(8,6))
    _ = FD1.plot(channels=[0,11,8], log=10, linewidth=0.5);




.. image:: /examples/Data/images/sphx_glr_plot_frequency_dataset_003.png
    :alt: plot frequency dataset
    :class: sphx-glr-single-img





Read in a second data set


.. code-block:: default

    FD2 = FdemData()
    FD2.read(dataFilename=dataFolder + 'Resolve1.txt', systemFilename=dataFolder + 'FdemSystem1.stm')





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Warning: Your data contains values that are <= 0.0

    <geobipy.src.classes.data.dataset.FdemData.FdemData object at 0x12cab4d60>



We can create maps of the elevations in two separate figures


.. code-block:: default

    plt.figure(figsize=(8,6))
    _ = FD1.mapPlot(dx=50.0, dy=50.0, mask = 200.0)
    plt.axis('equal');




.. image:: /examples/Data/images/sphx_glr_plot_frequency_dataset_004.png
    :alt: plot frequency dataset
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    (584519.0671621622, 590166.7428378379, 4639079.207593819, 4661808.632406181)




.. code-block:: default


    plt.figure(figsize=(8,6))
    _ = FD2.mapPlot(dx=50.0, dy=50.0, mask = 200.0)
    plt.axis('equal');




.. image:: /examples/Data/images/sphx_glr_plot_frequency_dataset_005.png
    :alt: plot frequency dataset
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    (662847.3094082569, 668366.7995917432, 4560053.628459876, 4600646.676540123)



Or, we can plot both data sets in one figure to see their positions relative
to each other.

In this case, I use a 2D scatter plot of the data point co-ordinates, and pass
one of the channels as the colour.


.. code-block:: default


    plt.figure(figsize=(8,6))
    _ = FD1.scatter2D(s=1.0, c=FD1.dataChannel(0))
    _ = FD2.scatter2D(s=1.0, c=FD2.dataChannel(0), cmap='jet');




.. image:: /examples/Data/images/sphx_glr_plot_frequency_dataset_006.png
    :alt: plot frequency dataset
    :class: sphx-glr-single-img





Or, interpolate the values to create a gridded "map". mapChannel will
interpolate the specified channel number.


.. code-block:: default


    plt.figure(figsize=(8,6))
    _ = FD1.mapData(channel=3, system=0, dx=200, dy=200, mask=250)
    plt.axis('equal');




.. image:: /examples/Data/images/sphx_glr_plot_frequency_dataset_007.png
    :alt: In-Phase 8171.0 (Hz)
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    (584518.405, 590167.405, 4639078.6625, 4661809.1775)



Export the data to VTK


.. code-block:: default


    # FD1.toVTK('FD_one')
    # FD2.toVTK('FD_two')








Obtain a line from the data set
+++++++++++++++++++++++++++++++

Take a look at the line numbers in the dataset


.. code-block:: default

    print(np.unique(FD1.lineNumber))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [30010. 30020. 30030. ... 30100. 39010. 39020.]





.. code-block:: default

    L = FD1.line(30010.0)








A summary will now show the properties of the line.


.. code-block:: default


    print(L.summary)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    3D Point Cloud: 
    Number of Points: : 6710 
     Name: Easting (m)
        Shape: (6710,)
        Values: [586852.29 586852.23 586852.17 ... 586123.57 586123.2  586122.82]
     Name: Northing (m)
        Shape: (6710,)
        Values: [4639119.38 4639122.68 4639125.98 ... 4661765.26 4661768.84 4661772.42]
     Name: Height (m)
        Shape: (6710,)
        Values: [36.629 37.012 37.349 ... 28.313 28.218 28.115]
     Name: Elevation (m)
        Shape: (6710,)
        Values: [1246.84 1246.71 1246.61 ... 1337.94 1337.96 1338.02]
    Data:          : 
    # of Channels: 12 
    # of Total Data: 80520 
    Name: Fdem Data
        Shape: (6710, 12)
        Values: [[145.3 435.8 260.6 ... 516.5 405.7 255.7]
     [145.7 436.5 257.9 ... 513.6 403.2 252. ]
     [146.4 437.4 255.8 ... 511.2 400.9 248.8]
     ...
     [ 70.7 314.1 220.2 ... 743.3 960.8 910.7]
     [ 71.3 315.3 220.5 ... 745.9 968.3 919.1]
     [ 72.1 316.6 220.7 ... 749.2 976.5 928.3]]

     Name: 
        Shape: (6710, 12)
        Values: [[14.53 43.58 26.06 ... 51.65 40.57 25.57]
     [14.57 43.65 25.79 ... 51.36 40.32 25.2 ]
     [14.64 43.74 25.58 ... 51.12 40.09 24.88]
     ...
     [ 7.07 31.41 22.02 ... 74.33 96.08 91.07]
     [ 7.13 31.53 22.05 ... 74.59 96.83 91.91]
     [ 7.21 31.66 22.07 ... 74.92 97.65 92.83]]

     Name: 
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


    plt.figure(figsize=(8,6))
    _ = L.scatter2D();




.. image:: /examples/Data/images/sphx_glr_plot_frequency_dataset_008.png
    :alt: plot frequency dataset
    :class: sphx-glr-single-img





We can specify the axis along which to plot.
xAxis can be index, x, y, z, r2d, r3d


.. code-block:: default

    plt.figure(figsize=(8,6))
    _ = FD1.plot(channels=[0,11,8], log=10, linewidth=0.5);




.. image:: /examples/Data/images/sphx_glr_plot_frequency_dataset_009.png
    :alt: plot frequency dataset
    :class: sphx-glr-single-img





Obtain a single datapoint from the data set
+++++++++++++++++++++++++++++++++++++++++++

Checkout :ref:`Frequency domain datapoint` for an example
about how to use a datapoint once it is instantiated.


.. code-block:: default

    dp = FD1.datapoint(0)








File Format for frequency domain data
+++++++++++++++++++++++++++++++++++++
Here we describe the file format for frequency domain data.

The data columns are read in according to the column names in the first line.

In this description, the column name or its alternatives are given followed by what the name represents.
Optional columns are also described.

Required columns
________________
line
    Line number for the data point
fid
    Unique identification number of the data point
x or northing or n
    Northing co-ordinate of the data point, (m)
y or easting or e
    Easting co-ordinate of the data point, (m)
z or alt
    Altitude of the transmitter coil above ground level (m)
elevation
    Elevation of the ground at the data point (m)
I_<frequency[0]> Q_<frequency[0]> ... I_<frequency[last]> Q_<frequency[last]>  - with the number and square brackets
    The measurements for each frequency specified in the accompanying system file.
    I is the real inphase measurement in (ppm)
    Q is the imaginary quadrature measurement in (ppm)
Optional columns
________________
InphaseErr[0] QuadratureErr[0] ... InphaseErr[nFrequencies] QuadratureErr[nFrequencies]
    Estimates of standard deviation for each inphase and quadrature measurement.
    These must appear after the data colums.

Example Header
______________
Line fid easting northing elevation height I_380 Q_380 ... ... I_129550 Q_129550

File Format for a frequency domain system
+++++++++++++++++++++++++++++++++++++++++
.. role:: raw-html(raw)
   :format: html

The system file is structured using columns with the first line containing header information

Each subsequent row contains the information for each measurement frequency

freq
    Frequency of the channel
tor
    Orientation of the transmitter loop 'x', or 'z'
tmom
    Transmitter moment
tx, ty, tx
    Offset of the transmitter with respect to the observation locations
ror
    Orientation of the receiver loop 'x', or 'z'
rmom
    Receiver moment
rx, ry, rz
    Offset of the receiver with respect to the transmitter location

Example system files are contained in
`the supplementary folder`_ in this repository

.. _the supplementary folder: https://github.com/usgs/geobipy/tree/master/documentation_source/source/examples/supplementary/Data

See the Resolve.stm files.


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  11.508 seconds)


.. _sphx_glr_download_examples_Data_plot_frequency_dataset.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_frequency_dataset.py <plot_frequency_dataset.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_frequency_dataset.ipynb <plot_frequency_dataset.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
