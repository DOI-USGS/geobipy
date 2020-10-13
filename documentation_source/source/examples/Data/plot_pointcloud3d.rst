.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_examples_Data_plot_pointcloud3d.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_examples_Data_plot_pointcloud3d.py:


3D Point Cloud class
--------------------

The 3D Point Cloud class extracts and utilizes the [Point](Point%20Class.ipynb) Class


.. code-block:: default


    from geobipy import PointCloud3D
    from os.path import join
    import numpy as np
    import matplotlib.pyplot as plt

    nPoints = 10000








Create a quick test example using random points
$z=x(1-x)cos(4\pi x)sin(4\pi y^{2})^{2}$


.. code-block:: default

    x = -np.abs((2.0 * np.random.rand(nPoints)) - 1.0)
    y = -np.abs((2.0 * np.random.rand(nPoints)) - 1.0)
    z = x * (1.0 - x) * np.cos(np.pi * x) * np.sin(np.pi * y)

    PC3D = PointCloud3D(x=x, y=y, z=z)








Append pointclouds together


.. code-block:: default

    x = np.abs((2.0 * np.random.rand(nPoints)) - 1.0)
    y = np.abs((2.0 * np.random.rand(nPoints)) - 1.0)
    z = x * (1.0 - x) * np.cos(np.pi * x) * np.sin(np.pi * y)

    Other_PC = PointCloud3D(x=x, y=y, z=z)
    PC3D.append(Other_PC)








Write a summary of the contents of the point cloud


.. code-block:: default


    PC3D.summary()





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    3D Point Cloud: 
    Number of Points: : 20000 
     Name: Easting
         Units: m
         Shape: (20000,)
         Values: [-0.49955817 -0.93368476 -0.69809349 ...  0.7588914   0.36178379
      0.09533399]
     Name: Northing
         Units: m
         Shape: (20000,)
         Values: [-0.06031532 -0.4436347  -0.16737524 ...  0.26017353  0.62756379
      0.47234696]
     Name: Height
         Units: m
         Shape: (20000,)
         Values: [ 1.95854160e-04 -1.73879019e+00 -3.46841568e-01 ... -9.69631853e-02
      8.94419868e-02  8.20953454e-02]
     Name: Elevation
         Units: m
         Shape: (20000,)
         Values: [0. 0. 0. ... 0. 0. 0.]





Get a single location from the point as a 3x1 vector


.. code-block:: default


    Point=PC3D.getPoint(50)
    # Print the point to the screen
    print(Point)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    <geobipy.src.classes.pointcloud.Point.Point object at 0x12b674f70>




Plot the locations with Height as colour


.. code-block:: default


    plt.figure()
    PC3D.scatter2D(edgecolor='k')




.. image:: /examples/Data/images/sphx_glr_plot_pointcloud3d_001.png
    :alt: plot pointcloud3d
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    (<AxesSubplot:xlabel='Easting (m)', ylabel='Northing (m)'>, <matplotlib.collections.PathCollection object at 0x12744e6d0>, <matplotlib.colorbar.Colorbar object at 0x12ba8c910>)



Plotting routines take matplotlib arguments for customization

For example, plotting the size of the points according to the absolute value of height


.. code-block:: default

    plt.figure()
    ax = PC3D.scatter2D(s=100*np.abs(PC3D.z), edgecolor='k')





.. image:: /examples/Data/images/sphx_glr_plot_pointcloud3d_002.png
    :alt: plot pointcloud3d
    :class: sphx-glr-single-img





Grid the points using a triangulated CloughTocher interpolation


.. code-block:: default


    plt.figure()
    plt.subplot(121)
    PC3D.mapPlot(dx=0.1, dy=0.1, method='ct')
    plt.subplot(122)
    PC3D.mapPlot(dx=0.1, dy=0.1, method='mc')





.. image:: /examples/Data/images/sphx_glr_plot_pointcloud3d_003.png
    :alt: plot pointcloud3d
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customPlots.py:649: MatplotlibDeprecationWarning: You are modifying the state of a globally registered colormap. In future versions, you will not be able to modify a registered colormap in-place. To remove this warning, you can make a copy of the colormap first. cmap = copy.copy(mpl.cm.get_cmap("viridis"))
      kwargs['cmap'].set_bad(color='white')
    Interpolating with gmt surface tmp.txt -I0.1/0.1 -R-1.04997/1.05003/-1.04988/1.05012 -N2000 -T0.25 -C0.01 -Gtmp.grd -Ll0 -Lu1
    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customPlots.py:649: MatplotlibDeprecationWarning: You are modifying the state of a globally registered colormap. In future versions, you will not be able to modify a registered colormap in-place. To remove this warning, you can make a copy of the colormap first. cmap = copy.copy(mpl.cm.get_cmap("viridis"))
      kwargs['cmap'].set_bad(color='white')

    (<AxesSubplot:xlabel='Easting (m)', ylabel='Northing (m)'>, <matplotlib.collections.QuadMesh object at 0x126fa0f10>, <matplotlib.colorbar.Colorbar object at 0x127972190>)



We can perform spatial searches on the 3D point cloud


.. code-block:: default


    PC3D.setKdTree(nDims=2)
    p = PC3D.nearest((0.0,0.0), k=200, p=2, radius=0.3)
    print(p)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    (array([0.00420115, 0.00640944, 0.00971474, ..., 0.11054256, 0.11103671,
           0.11139794]), array([16626, 16568, 10093, ...,  7117, 17037,   409]))




.nearest returns the distances and indices into the point cloud of the nearest points.
We can then obtain those points as another point cloud


.. code-block:: default


    pNear = PC3D[p[1]]
    plt.figure()
    ax1 = plt.subplot(1,2,1)
    pNear.scatter2D()
    plt.plot(0.0, 0.0, 'x')
    plt.subplot(1,2,2, sharex=ax1, sharey=ax1)
    ax, sc, cb = PC3D.scatter2D(edgecolor='k')
    searchRadius = plt.Circle((0.0, 0.0), 0.3, color='b', fill=False)
    ax.add_artist(searchRadius)
    plt.plot(0.0, 0.0, 'x')





.. image:: /examples/Data/images/sphx_glr_plot_pointcloud3d_004.png
    :alt: plot pointcloud3d
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    [<matplotlib.lines.Line2D object at 0x1272f8fa0>]



Read in the xyz co-ordinates in columns 2,3,4 from a file. Skip 1 header line.


.. code-block:: default


    dataFolder = "..//supplementary//Data//"

    PC3D.read(fileName=dataFolder + 'Resolve1.txt')






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    <geobipy.src.classes.pointcloud.PointCloud3D.PointCloud3D object at 0x12b674b80>




.. code-block:: default



    plt.figure()
    f = PC3D.scatter2D(s=10)




.. image:: /examples/Data/images/sphx_glr_plot_pointcloud3d_005.png
    :alt: plot pointcloud3d
    :class: sphx-glr-single-img





Export the 3D Pointcloud to a VTK file.

In this case, I pass the height as point data so that the points are coloured
when opened in Paraview (or other software)


.. code-block:: default



    # PC3D.toVTK('testPoints', format='binary')








.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  5.513 seconds)


.. _sphx_glr_download_examples_Data_plot_pointcloud3d.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_pointcloud3d.py <plot_pointcloud3d.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_pointcloud3d.ipynb <plot_pointcloud3d.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
