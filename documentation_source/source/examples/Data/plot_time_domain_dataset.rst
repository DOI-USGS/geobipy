.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_examples_Data_plot_time_domain_dataset.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_examples_Data_plot_time_domain_dataset.py:


Time domain dataset
--------------------


.. code-block:: default

    from geobipy import customPlots as cP
    from os.path import join
    import matplotlib.pyplot as plt
    import numpy as np
    from geobipy.src.classes.core.StatArray import StatArray
    from geobipy.src.classes.data.dataset.TdemData import TdemData








Reading in the Data
+++++++++++++++++++


.. code-block:: default

    dataFolder = "..//supplementary//Data//"
    # The data file name
    dataFiles=[dataFolder + 'Skytem_High.txt', dataFolder + 'Skytem_Low.txt']
    # The EM system file name
    systemFiles=[dataFolder + 'SkytemHM-SLV.stm', dataFolder + 'SkytemLM-SLV.stm']








Read in the data from file


.. code-block:: default

    TD = TdemData()
    TD.read(dataFiles, systemFiles)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    <geobipy.src.classes.data.dataset.TdemData.TdemData object at 0x127020f10>



Plot the locations of the data points


.. code-block:: default

    plt.figure(figsize=(8,6))
    _ = TD.scatter2D()





.. image:: /examples/Data/images/sphx_glr_plot_time_domain_dataset_001.png
    :alt: plot time domain dataset
    :class: sphx-glr-single-img





Plot all the data along the specified line


.. code-block:: default

    plt.figure(figsize=(8,6))
    _ = TD.plotLine(100101.0, log=10)




.. image:: /examples/Data/images/sphx_glr_plot_time_domain_dataset_002.png
    :alt: plot time domain dataset
    :class: sphx-glr-single-img





Or, plot specific channels in the data


.. code-block:: default

    plt.figure(figsize=(8,6))
    _ = TD.plot(system=0, channels=[17, 18, 19], log=10)




.. image:: /examples/Data/images/sphx_glr_plot_time_domain_dataset_003.png
    :alt: plot time domain dataset
    :class: sphx-glr-single-img






.. code-block:: default

    plt.figure()
    plt.subplot(211)
    _ = TD.pcolor(system=0, log=10, xscale='log')
    plt.subplot(212)
    _ = TD.pcolor(system=1, log=10, xscale='log')




.. image:: /examples/Data/images/sphx_glr_plot_time_domain_dataset_004.png
    :alt: plot time domain dataset
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customPlots.py:649: MatplotlibDeprecationWarning: You are modifying the state of a globally registered colormap. In future versions, you will not be able to modify a registered colormap in-place. To remove this warning, you can make a copy of the colormap first. cmap = copy.copy(mpl.cm.get_cmap("viridis"))
      kwargs['cmap'].set_bad(color='white')





.. code-block:: default

    plt.figure()
    ax = TD.scatter2D(s=1.0, c=TD.dataChannel(system=0, channel=23), equalize=True)
    plt.axis('equal')




.. image:: /examples/Data/images/sphx_glr_plot_time_domain_dataset_005.png
    :alt: plot time domain dataset
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    (429026.56, 454054.24, 4160662.0, 4200460.0)



TD.toVTK('TD1', format='binary')

Obtain a line from the data set
+++++++++++++++++++++++++++++++


.. code-block:: default

    line = TD.line(100601.0)









.. code-block:: default

    plt.figure()
    _ = line.scatter2D(c = line.dataChannel(17, system=1))




.. image:: /examples/Data/images/sphx_glr_plot_time_domain_dataset_006.png
    :alt: plot time domain dataset
    :class: sphx-glr-single-img






.. code-block:: default

    plt.figure()
    _ = line.plot(xAxis='x', log=10)





.. image:: /examples/Data/images/sphx_glr_plot_time_domain_dataset_007.png
    :alt: plot time domain dataset
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customFunctions.py:664: RuntimeWarning: All-NaN axis encountered
      if (np.nanmin(values) <= 0.0):
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customFunctions.py:664: RuntimeWarning: All-NaN axis encountered
      if (np.nanmin(values) <= 0.0):
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customFunctions.py:664: RuntimeWarning: All-NaN axis encountered
      if (np.nanmin(values) <= 0.0):
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customFunctions.py:664: RuntimeWarning: All-NaN axis encountered
      if (np.nanmin(values) <= 0.0):
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customFunctions.py:664: RuntimeWarning: All-NaN axis encountered
      if (np.nanmin(values) <= 0.0):
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customFunctions.py:664: RuntimeWarning: All-NaN axis encountered
      if (np.nanmin(values) <= 0.0):
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customFunctions.py:664: RuntimeWarning: All-NaN axis encountered
      if (np.nanmin(values) <= 0.0):
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customFunctions.py:664: RuntimeWarning: All-NaN axis encountered
      if (np.nanmin(values) <= 0.0):
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customFunctions.py:664: RuntimeWarning: All-NaN axis encountered
      if (np.nanmin(values) <= 0.0):
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customFunctions.py:664: RuntimeWarning: All-NaN axis encountered
      if (np.nanmin(values) <= 0.0):
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customFunctions.py:664: RuntimeWarning: All-NaN axis encountered
      if (np.nanmin(values) <= 0.0):
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customFunctions.py:664: RuntimeWarning: All-NaN axis encountered
      if (np.nanmin(values) <= 0.0):
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customFunctions.py:664: RuntimeWarning: All-NaN axis encountered
      if (np.nanmin(values) <= 0.0):
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customFunctions.py:664: RuntimeWarning: All-NaN axis encountered
      if (np.nanmin(values) <= 0.0):
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customFunctions.py:664: RuntimeWarning: All-NaN axis encountered
      if (np.nanmin(values) <= 0.0):
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customFunctions.py:664: RuntimeWarning: All-NaN axis encountered
      if (np.nanmin(values) <= 0.0):
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customFunctions.py:664: RuntimeWarning: All-NaN axis encountered
      if (np.nanmin(values) <= 0.0):
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customFunctions.py:664: RuntimeWarning: All-NaN axis encountered
      if (np.nanmin(values) <= 0.0):
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customFunctions.py:664: RuntimeWarning: All-NaN axis encountered
      if (np.nanmin(values) <= 0.0):
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customFunctions.py:664: RuntimeWarning: All-NaN axis encountered
      if (np.nanmin(values) <= 0.0):
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customFunctions.py:664: RuntimeWarning: All-NaN axis encountered
      if (np.nanmin(values) <= 0.0):
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customFunctions.py:664: RuntimeWarning: All-NaN axis encountered
      if (np.nanmin(values) <= 0.0):
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customFunctions.py:664: RuntimeWarning: All-NaN axis encountered
      if (np.nanmin(values) <= 0.0):
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customFunctions.py:664: RuntimeWarning: All-NaN axis encountered
      if (np.nanmin(values) <= 0.0):




File Format for time domain data
++++++++++++++++++++++++++++++++
Here we describe the file format for time domain data.

The data columns are read in according to the column names in the first line

In this description, the column name or its alternatives are given followed by what the name represents
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
txrx_dx
    Distance in x between transmitter and reciever (m)
txrx_dy
    Distance in y between transmitter and reciever (m)
txrx_dz
    Distance in z between transmitter and reciever (m)
TxPitch
    Pitch of the transmitter loop
TxRoll
    Roll of the transmitter loop
TxYaw
    Yaw of the transmitter loop
RxPitch
    Pitch of the receiver loop
RxRoll
    Roll of the receiver loop
RxYaw
    Yaw of the receiver loop
Off[0] Off[1] ... Off[last]  - with the number and square brackets
    The measurements for each time gate specified in the accompanying system file under Receiver Window Times
Optional columns
________________
OffErr[0] OffErr[1] ... OffErr[last]
    Estimates of standard deviation for each off time measurement
Example Header
______________
Line fid easting northing elevation height txrx_dx txrx_dy txrx_dz TxPitch TxRoll TxYaw RxPitch RxRoll RxYaw Off[0] Off[1]

File Format for a time domain system
++++++++++++++++++++++++++++++++++++
Please see Page 13 of Ross Brodie's `instructions`_

.. _instructions: https://github.com/GeoscienceAustralia/ga-aem/blob/master/docs/GA%20AEM%20Programs%20User%20Manual.pdf

We use GA-AEM for our airborne time domain forward modeller.

Example system files are contained in
`the supplementary folder`_ in this repository

.. _the supplementary folder: https://github.com/usgs/geobipy/tree/master/documentation_source/source/examples/supplementary/Data


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  6.465 seconds)


.. _sphx_glr_download_examples_Data_plot_time_domain_dataset.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_time_domain_dataset.py <plot_time_domain_dataset.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_time_domain_dataset.ipynb <plot_time_domain_dataset.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
