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


    print(PC3D.summary)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    3D Point Cloud: 
    Number of Points: : 20000 
     Name: Easting (m)
        Shape: (20000,)
        Values: [-0.98773319 -0.55023466 -0.13412766 ...  0.84262521  0.3091211
      0.51431745]
     Name: Northing (m)
        Shape: (20000,)
        Values: [-0.68587549 -0.37151232 -0.07661938 ...  0.57331589  0.53308878
      0.70317217]
     Name: Height (m)
        Shape: (20000,)
        Values: [-1.6367953  -0.12328441  0.03309123 ... -0.11364554  0.11987811
     -0.00902055]
     Name: Elevation (m)
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

    <geobipy.src.classes.pointcloud.Point.Point object at 0x12bf328b0>




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


    (<AxesSubplot:xlabel='Easting (m)', ylabel='Northing (m)'>, <matplotlib.collections.PathCollection object at 0x12c427280>, <matplotlib.colorbar.Colorbar object at 0x12c416280>)



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

    Interpolating with gmt surface tmp.txt -I0.1/0.1 -R-1.04988/1.05012/-1.04995/1.05005 -N2000 -T0.25 -C0.01 -Gtmp.grd -Ll0 -Lu1

    (<AxesSubplot:xlabel='Easting (m)', ylabel='Northing (m)'>, <matplotlib.collections.QuadMesh object at 0x12c2e4490>, <matplotlib.colorbar.Colorbar object at 0x12c7a4130>)



We can perform spatial searches on the 3D point cloud


.. code-block:: default


    PC3D.setKdTree(nDims=2)
    p = PC3D.nearest((0.0,0.0), k=200, p=2, radius=0.3)
    print(p)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    (array([0.01599939, 0.02188347, 0.02220646, ..., 0.11441974, 0.11456269,
           0.11582736]), array([18456,  6029,   577, ..., 13206,  1778,  2833]))




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


    [<matplotlib.lines.Line2D object at 0x12ee76700>]



Read in the xyz co-ordinates in columns 2,3,4 from a file. Skip 1 header line.


.. code-block:: default


    dataFolder = "..//supplementary//Data//"

    PC3D.read(fileName=dataFolder + 'Resolve1.txt')






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    <geobipy.src.classes.pointcloud.PointCloud3D.PointCloud3D object at 0x12bf32730>




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

   **Total running time of the script:** ( 0 minutes  5.245 seconds)


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
