.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_examples_Statistics_plot_histogram_1d.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_Statistics_plot_histogram_1d.py:


Histogram 1D
------------

This histogram class allows efficient updating of histograms, plotting and
saving as HDF5


.. code-block:: default

    import h5py
    from geobipy import hdfRead
    from geobipy import StatArray
    from geobipy import Histogram1D
    import numpy as np
    import matplotlib.pyplot as plt








Histogram with regular bins
+++++++++++++++++++++++++++


.. code-block:: default


    # Create regularly spaced bins
    bins = StatArray(np.linspace(-3, 3, 101), 'Regular bins')









.. code-block:: default


    # Set the histogram using the bins, and update
    H = Histogram1D(bins = bins)









.. code-block:: default


    # We can update the histogram with some new values
    x = np.random.randn(1000)
    H.update(x, clip=True, trim=True)

    # Plot the histogram
    plt.figure()
    _ = H.plot()




.. image:: /examples/Statistics/images/sphx_glr_plot_histogram_1d_001.png
    :class: sphx-glr-single-img





Get the median, and 95% confidence values


.. code-block:: default

    print(H.credibleIntervals(percent=95.0))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    (-0.030000000000000027, -1.71, 1.5299999999999998)





.. code-block:: default


    # We can write the histogram to a HDF file
    with h5py.File('Histogram.h5','w') as hf:
        H.toHdf(hf,'Histogram')









.. code-block:: default


    # And read it back in from Hdf5
    H1 = hdfRead.readKeyFromFiles('Histogram.h5','/','Histogram')

    plt.figure()
    _ = H1.plot()





.. image:: /examples/Statistics/images/sphx_glr_plot_histogram_1d_002.png
    :class: sphx-glr-single-img





Histogram with irregular bins
+++++++++++++++++++++++++++++


.. code-block:: default


    # Create irregularly spaced bins
    x = np.cumsum(np.arange(10))
    irregularBins = np.hstack([-x[::-1], x[1:]]) 









Create a named StatArray


.. code-block:: default

    edges = StatArray(irregularBins, 'irregular bins')









Instantiate the histogram with bin edges


.. code-block:: default

    H = Histogram1D(bins=edges)









Generate random numbers


.. code-block:: default

    x = (np.random.randn(10000)*20.0) - 10.0









Update the histogram


.. code-block:: default

    H.update(x)










.. code-block:: default

    plt.figure()
    _ = H.plot()





.. image:: /examples/Statistics/images/sphx_glr_plot_histogram_1d_003.png
    :class: sphx-glr-single-img





We can plot the histogram as a pcolor plot
.


.. code-block:: default

    plt.figure()
    _ = H.pcolor(grid=True, transpose=True)





.. image:: /examples/Statistics/images/sphx_glr_plot_histogram_1d_004.png
    :class: sphx-glr-single-img





Histogram with linear space entries that are logged internally
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Create some bins spaced logarithmically


.. code-block:: default

    positiveBins = StatArray(np.logspace(-5, 3), 'positive bins')









.. code-block:: default

    print(positiveBins)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [1.00000000e-05 1.45634848e-05 2.12095089e-05 3.08884360e-05
     4.49843267e-05 6.55128557e-05 9.54095476e-05 1.38949549e-04
     2.02358965e-04 2.94705170e-04 4.29193426e-04 6.25055193e-04
     9.10298178e-04 1.32571137e-03 1.93069773e-03 2.81176870e-03
     4.09491506e-03 5.96362332e-03 8.68511374e-03 1.26485522e-02
     1.84206997e-02 2.68269580e-02 3.90693994e-02 5.68986603e-02
     8.28642773e-02 1.20679264e-01 1.75751062e-01 2.55954792e-01
     3.72759372e-01 5.42867544e-01 7.90604321e-01 1.15139540e+00
     1.67683294e+00 2.44205309e+00 3.55648031e+00 5.17947468e+00
     7.54312006e+00 1.09854114e+01 1.59985872e+01 2.32995181e+01
     3.39322177e+01 4.94171336e+01 7.19685673e+01 1.04811313e+02
     1.52641797e+02 2.22299648e+02 3.23745754e+02 4.71486636e+02
     6.86648845e+02 1.00000000e+03]




Instantiate the Histogram with log=10


.. code-block:: default

    H = Histogram1D(bins=positiveBins, log=10)








Generate random 10**x numbers


.. code-block:: default

    x = 10.0**(np.random.randn(1000)*2.0)








The update takes in the numbers in linear space and takes their log=10


.. code-block:: default

    H.update(x, trim=True)









.. code-block:: default

    plt.figure()
    _ = H.plot()



.. image:: /examples/Statistics/images/sphx_glr_plot_histogram_1d_005.png
    :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.764 seconds)


.. _sphx_glr_download_examples_Statistics_plot_histogram_1d.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_histogram_1d.py <plot_histogram_1d.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_histogram_1d.ipynb <plot_histogram_1d.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
