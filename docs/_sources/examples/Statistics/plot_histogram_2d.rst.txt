.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_examples_Statistics_plot_histogram_2d.py>`     to download the full example code
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
    :alt: plot histogram 2d
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customPlots.py:649: MatplotlibDeprecationWarning: You are modifying the state of a globally registered colormap. In future versions, you will not be able to modify a registered colormap in-place. To remove this warning, you can make a copy of the colormap first. cmap = copy.copy(mpl.cm.get_cmap("gray_r"))
      kwargs['cmap'].set_bad(color='white')




Generate marginal histograms along an axis


.. code-block:: default

    h1 = H.marginalize(axis=0)
    h2 = H.marginalize(axis=1)









Note that the names of the variables are automatically displayed


.. code-block:: default

    plt.figure()
    plt.subplot(121)
    h1.plot()
    plt.subplot(122)
    _ = h2.plot()





.. image:: /examples/Statistics/images/sphx_glr_plot_histogram_2d_002.png
    :alt: plot histogram 2d
    :class: sphx-glr-single-img





Create a combination plot with marginal histograms.
sphinx_gallery_thumbnail_number = 3


.. code-block:: default

    plt.figure()
    _ = H.comboPlot(cmap='gray_r')





.. image:: /examples/Statistics/images/sphx_glr_plot_histogram_2d_003.png
    :alt: plot histogram 2d
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customPlots.py:649: MatplotlibDeprecationWarning: You are modifying the state of a globally registered colormap. In future versions, you will not be able to modify a registered colormap in-place. To remove this warning, you can make a copy of the colormap first. cmap = copy.copy(mpl.cm.get_cmap("gray_r"))
      kwargs['cmap'].set_bad(color='white')




We can overlay the histogram with its credible intervals


.. code-block:: default

    plt.figure()
    H.pcolor(cmap='gray_r')
    H.plotCredibleIntervals(axis=0, percent=95.0)
    _ = H.plotCredibleIntervals(axis=1, percent=95.0)





.. image:: /examples/Statistics/images/sphx_glr_plot_histogram_2d_004.png
    :alt: plot histogram 2d
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customPlots.py:649: MatplotlibDeprecationWarning: You are modifying the state of a globally registered colormap. In future versions, you will not be able to modify a registered colormap in-place. To remove this warning, you can make a copy of the colormap first. cmap = copy.copy(mpl.cm.get_cmap("gray_r"))
      kwargs['cmap'].set_bad(color='white')




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
    :alt: plot histogram 2d
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customPlots.py:649: MatplotlibDeprecationWarning: You are modifying the state of a globally registered colormap. In future versions, you will not be able to modify a registered colormap in-place. To remove this warning, you can make a copy of the colormap first. cmap = copy.copy(mpl.cm.get_cmap("gray_r"))
      kwargs['cmap'].set_bad(color='white')

    <matplotlib.legend.Legend object at 0x1307aa490>




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
    :alt: plot histogram 2d
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customPlots.py:649: MatplotlibDeprecationWarning: You are modifying the state of a globally registered colormap. In future versions, you will not be able to modify a registered colormap in-place. To remove this warning, you can make a copy of the colormap first. cmap = copy.copy(mpl.cm.get_cmap("gray_r"))
      kwargs['cmap'].set_bad(color='white')
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customPlots.py:649: MatplotlibDeprecationWarning: You are modifying the state of a globally registered colormap. In future versions, you will not be able to modify a registered colormap in-place. To remove this warning, you can make a copy of the colormap first. cmap = copy.copy(mpl.cm.get_cmap("gray_r"))
      kwargs['cmap'].set_bad(color='white')





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
    :alt: plot histogram 2d
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customPlots.py:649: MatplotlibDeprecationWarning: You are modifying the state of a globally registered colormap. In future versions, you will not be able to modify a registered colormap in-place. To remove this warning, you can make a copy of the colormap first. cmap = copy.copy(mpl.cm.get_cmap("gray_r"))
      kwargs['cmap'].set_bad(color='white')
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customPlots.py:690: MatplotlibDeprecationWarning: shading='flat' when X and Y have the same dimensions as C is deprecated since 3.3.  Either specify the corners of the quadrilaterals with X and Y, or pass shading='auto', 'nearest' or 'gouraud', or set rcParams['pcolor.shading'].  This will become an error two minor releases later.
      pm = ax.pcolormesh(X, Y, Zm, alpha = alpha, **kwargs)
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customPlots.py:649: MatplotlibDeprecationWarning: You are modifying the state of a globally registered colormap. In future versions, you will not be able to modify a registered colormap in-place. To remove this warning, you can make a copy of the colormap first. cmap = copy.copy(mpl.cm.get_cmap("gray_r"))
      kwargs['cmap'].set_bad(color='white')
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customPlots.py:690: MatplotlibDeprecationWarning: shading='flat' when X and Y have the same dimensions as C is deprecated since 3.3.  Either specify the corners of the quadrilaterals with X and Y, or pass shading='auto', 'nearest' or 'gouraud', or set rcParams['pcolor.shading'].  This will become an error two minor releases later.
      pm = ax.pcolormesh(X, Y, Zm, alpha = alpha, **kwargs)




Get the range between credible intervals


.. code-block:: default

    H.credibleRange(percent=95.0)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    StatArray([2.96, 2.56, 2.4 , 2.8 , 2.88, 2.96, 3.68, 3.04, 4.  , 3.44,
               3.44, 3.28, 3.28, 3.36, 3.2 , 3.28, 3.2 , 3.36, 3.2 , 3.44,
               3.28, 3.28, 3.28, 3.2 , 3.28, 3.36, 3.28, 3.28, 3.2 , 3.28,
               3.2 , 3.2 , 3.28, 3.36, 3.36, 3.2 , 3.36, 3.2 , 3.28, 3.2 ,
               3.28, 3.28, 3.28, 3.2 , 3.36, 3.2 , 3.28, 3.28, 3.36, 3.36,
               3.28, 3.36, 3.36, 3.36, 3.28, 3.28, 3.36, 3.28, 3.28, 3.36,
               3.28, 3.36, 3.36, 3.28, 3.36, 3.28, 3.36, 3.36, 3.2 , 3.36,
               3.2 , 3.28, 3.28, 3.2 , 3.28, 3.36, 3.36, 3.28, 3.28, 3.36,
               3.28, 3.44, 3.2 , 3.28, 3.12, 3.04, 2.96, 3.44, 3.52, 3.28,
               3.12, 3.12, 3.68, 3.04, 2.64, 2.64, 3.28, 2.72, 1.36, 3.04])



We can map the credible range to an opacity or transparency


.. code-block:: default

    H.opacity()
    H.transparency()




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    StatArray([0.60606061, 0.45454545, 0.39393939, 0.54545455, 0.57575758,
               0.60606061, 0.87878788, 0.63636364, 1.        , 0.78787879,
               0.78787879, 0.72727273, 0.72727273, 0.75757576, 0.6969697 ,
               0.72727273, 0.6969697 , 0.75757576, 0.6969697 , 0.78787879,
               0.72727273, 0.72727273, 0.72727273, 0.6969697 , 0.72727273,
               0.75757576, 0.72727273, 0.72727273, 0.6969697 , 0.72727273,
               0.6969697 , 0.6969697 , 0.72727273, 0.75757576, 0.75757576,
               0.6969697 , 0.75757576, 0.6969697 , 0.72727273, 0.6969697 ,
               0.72727273, 0.72727273, 0.72727273, 0.6969697 , 0.75757576,
               0.6969697 , 0.72727273, 0.72727273, 0.75757576, 0.75757576,
               0.72727273, 0.75757576, 0.75757576, 0.75757576, 0.72727273,
               0.72727273, 0.75757576, 0.72727273, 0.72727273, 0.75757576,
               0.72727273, 0.75757576, 0.75757576, 0.72727273, 0.75757576,
               0.72727273, 0.75757576, 0.75757576, 0.6969697 , 0.75757576,
               0.6969697 , 0.72727273, 0.72727273, 0.6969697 , 0.72727273,
               0.75757576, 0.75757576, 0.72727273, 0.72727273, 0.75757576,
               0.72727273, 0.78787879, 0.6969697 , 0.72727273, 0.66666667,
               0.63636364, 0.60606061, 0.78787879, 0.81818182, 0.72727273,
               0.66666667, 0.66666667, 0.87878788, 0.63636364, 0.48484848,
               0.48484848, 0.72727273, 0.51515152, 0.        , 0.63636364])




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  2.164 seconds)


.. _sphx_glr_download_examples_Statistics_plot_histogram_2d.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_histogram_2d.py <plot_histogram_2d.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_histogram_2d.ipynb <plot_histogram_2d.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
