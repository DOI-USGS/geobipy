.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_examples_Statistics_plot_histogram_2d.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_Statistics_plot_histogram_2d.py:


Histogram 2D
------------

This 2D histogram class allows efficient updating of histograms, plotting and
saving as HDF5.


.. code-block:: default

    import geobipy
    from geobipy import StatArray
    from geobipy import Histogram2D
    import matplotlib.pyplot as plt
    import numpy as np









Create some histogram bins in x and y


.. code-block:: default

    x = StatArray(np.linspace(-4.0, 4.0, 101), 'Variable 1')
    y = StatArray(np.linspace(-4.0, 4.0, 101), 'Variable 2')








Instantiate


.. code-block:: default

    H = Histogram2D(xBins=x, yBins=y)









Generate some random numbers


.. code-block:: default

    a = np.random.randn(1000000)
    b = np.random.randn(1000000)
    x = np.asarray([a, b])









Update the histogram counts


.. code-block:: default

    H.update(x)










.. code-block:: default

    plt.figure()
    _ = H.pcolor(cmap='gray_r')





.. image:: /examples/Statistics/images/sphx_glr_plot_histogram_2d_001.png
    :class: sphx-glr-single-img





Generate marginal histograms along an axis


.. code-block:: default

    h1 = H.marginalHistogram(axis=0)
    h2 = H.marginalHistogram(axis=1)









Note that the names of the variables are automatically displayed


.. code-block:: default

    plt.figure()
    plt.subplot(121)
    h1.plot()
    plt.subplot(122)
    _ = h2.plot()





.. image:: /examples/Statistics/images/sphx_glr_plot_histogram_2d_002.png
    :class: sphx-glr-single-img





Create a combination plot with marginal histograms.
sphinx_gallery_thumbnail_number = 3


.. code-block:: default

    plt.figure()
    _ = H.comboPlot(cmap='gray_r')





.. image:: /examples/Statistics/images/sphx_glr_plot_histogram_2d_003.png
    :class: sphx-glr-single-img





We can overlay the histogram with its credible intervals


.. code-block:: default

    plt.figure()
    H.pcolor(cmap='gray_r')
    H.plotCredibleIntervals(axis=0, percent=95.0)
    _ = H.plotCredibleIntervals(axis=1, percent=95.0)





.. image:: /examples/Statistics/images/sphx_glr_plot_histogram_2d_004.png
    :class: sphx-glr-single-img





Take the mean or median estimates from the histogram


.. code-block:: default

    mean = H.mean()
    median = H.median()









Or plot the mean and median


.. code-block:: default

    plt.figure()
    H.pcolor(cmap='gray_r')
    H.plotMean()
    H.plotMedian()
    plt.legend()




.. image:: /examples/Statistics/images/sphx_glr_plot_histogram_2d_005.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    <matplotlib.legend.Legend object at 0x12bba0910>




.. code-block:: default

    plt.figure(figsize=(9.5, 5))
    ax = plt.subplot(121)
    H.pcolor(cmap='gray_r', noColorbar=True)
    H.plotCredibleIntervals(axis=0)
    H.plotMedian()
    H.plotMean(color='y')

    plt.subplot(122, sharex=ax, sharey=ax)
    H.pcolor(cmap='gray_r', noColorbar=True)
    H.plotCredibleIntervals(axis=1)
    H.plotMedian(axis=1)
    H.plotMean(axis=1, color='y')





.. image:: /examples/Statistics/images/sphx_glr_plot_histogram_2d_006.png
    :class: sphx-glr-single-img






.. code-block:: default

    plt.figure(figsize=(9.5, 5))
    ax = plt.subplot(121)
    H1 = H.intervalStatistic([-4.0, -2.0, 2.0, 4.0], statistic='mean', axis=0)
    H1.pcolor(cmap='gray_r', equalize=True, noColorbar=True)
    H1.plotCredibleIntervals(axis=0)
    plt.subplot(122, sharex=ax, sharey=ax)
    H1 = H.intervalStatistic([-4.0, -2.0, 2.0, 4.0], statistic='mean', axis=1)
    H1.pcolor(cmap='gray_r', equalize=True, noColorbar=True)
    H1.plotCredibleIntervals(axis=1)





.. image:: /examples/Statistics/images/sphx_glr_plot_histogram_2d_007.png
    :class: sphx-glr-single-img





Get the range between credible intervals


.. code-block:: default

    H.credibleRange(percent=95.0)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    StatArray([3.44, 3.12, 3.2 , 2.8 , 3.52, 3.44, 4.16, 2.88, 2.8 , 3.44,
               3.36, 3.28, 3.2 , 3.28, 3.36, 3.28, 3.36, 3.36, 3.36, 3.28,
               3.36, 3.2 , 3.2 , 3.28, 3.36, 3.28, 3.28, 3.28, 3.2 , 3.36,
               3.28, 3.28, 3.36, 3.28, 3.28, 3.28, 3.28, 3.28, 3.36, 3.28,
               3.36, 3.28, 3.28, 3.36, 3.28, 3.28, 3.28, 3.28, 3.36, 3.28,
               3.36, 3.28, 3.28, 3.36, 3.28, 3.28, 3.2 , 3.28, 3.2 , 3.28,
               3.36, 3.36, 3.28, 3.28, 3.2 , 3.36, 3.28, 3.28, 3.36, 3.28,
               3.2 , 3.2 , 3.36, 3.36, 3.28, 3.28, 3.28, 3.36, 3.36, 3.36,
               3.28, 3.12, 3.28, 3.44, 3.36, 3.36, 3.36, 3.36, 2.96, 3.28,
               2.88, 3.36, 3.6 , 2.72, 3.12, 3.44, 4.08, 5.2 , 4.24, 3.6 ])



We can map the credible range to an opacity or transparency


.. code-block:: default

    H.opacity()
    H.transparency()




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    StatArray([0.29032258, 0.16129032, 0.19354839, 0.03225806, 0.32258065,
               0.29032258, 0.58064516, 0.06451613, 0.03225806, 0.29032258,
               0.25806452, 0.22580645, 0.19354839, 0.22580645, 0.25806452,
               0.22580645, 0.25806452, 0.25806452, 0.25806452, 0.22580645,
               0.25806452, 0.19354839, 0.19354839, 0.22580645, 0.25806452,
               0.22580645, 0.22580645, 0.22580645, 0.19354839, 0.25806452,
               0.22580645, 0.22580645, 0.25806452, 0.22580645, 0.22580645,
               0.22580645, 0.22580645, 0.22580645, 0.25806452, 0.22580645,
               0.25806452, 0.22580645, 0.22580645, 0.25806452, 0.22580645,
               0.22580645, 0.22580645, 0.22580645, 0.25806452, 0.22580645,
               0.25806452, 0.22580645, 0.22580645, 0.25806452, 0.22580645,
               0.22580645, 0.19354839, 0.22580645, 0.19354839, 0.22580645,
               0.25806452, 0.25806452, 0.22580645, 0.22580645, 0.19354839,
               0.25806452, 0.22580645, 0.22580645, 0.25806452, 0.22580645,
               0.19354839, 0.19354839, 0.25806452, 0.25806452, 0.22580645,
               0.22580645, 0.22580645, 0.25806452, 0.25806452, 0.25806452,
               0.22580645, 0.16129032, 0.22580645, 0.29032258, 0.25806452,
               0.25806452, 0.25806452, 0.25806452, 0.09677419, 0.22580645,
               0.06451613, 0.25806452, 0.35483871, 0.        , 0.16129032,
               0.29032258, 0.5483871 , 1.        , 0.61290323, 0.35483871])




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  2.191 seconds)


.. _sphx_glr_download_examples_Statistics_plot_histogram_2d.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_histogram_2d.py <plot_histogram_2d.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_histogram_2d.ipynb <plot_histogram_2d.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
