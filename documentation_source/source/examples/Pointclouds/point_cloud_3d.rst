.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_examples_Pointclouds_point_cloud_3d.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_Pointclouds_point_cloud_3d.py:


3D Point Cloud class
--------------------

The 3D Point Cloud class extracts and utilizes the [Point](Point%20Class.ipynb) Class


.. code-block:: default


    from geobipy import PointCloud3D
    from os.path import join
    import numpy as np
    import matplotlib.pyplot as plt
    # Initialize a 3D point cloud with N elements
    N=10
    # Instantiation pointcloud with an integer size N
    PC3D=PointCloud3D(N)







Create a quick test example using random points
$z=x(1-x)cos(4\pi x)sin(4\pi y^{2})^{2}$


.. code-block:: default


    PC3D.maketest(8000)







Write a summary of the contents of the point cloud


.. code-block:: default



    PC3D.summary()





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    3D Point Cloud: 
    Number of Points: : 8000 
     Name: Easting
         Units: m
         Shape: (8000,)
         Values: [ 0.86714772 -0.82057241  0.30236651 ... -0.78351773  0.66318959
     -0.83950573]
     Name: Northing
         Units: m
         Shape: (8000,)
         Values: [ 0.81318253  0.47732508  0.95057799 ... -0.99134299  0.25963456
     -0.78335139]
     Name: Height
         Units: m
         Shape: (8000,)
         Values: [-0.05832113  1.259586    0.01897687 ... -0.02954588 -0.07978319
     -0.85084587]
     Name: Elevation
         Units: m
         Shape: (8000,)
         Values: [0. 0. 0. ... 0. 0. 0.]




Get a single location from the point as a 3x1 vector


.. code-block:: default


    Point=PC3D.getPoint(50)
    # Print the point to the screen
    print(Point)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    <geobipy.src.classes.pointcloud.Point.Point object at 0x12110e160>



Plot the locations with Height as colour


.. code-block:: default


    plt.figure()
    PC3D.scatter2D(edgecolor='k')




.. image:: /examples/Pointclouds/images/sphx_glr_point_cloud_3d_001.png
    :class: sphx-glr-single-img




Plotting routines take matplotlib arguments for customization

For example, plotting the size of the points according to the absolute value of height


.. code-block:: default



    plt.figure()
    ax = PC3D.scatter2D(s=100*np.abs(PC3D.z),edgecolor='k')





.. image:: /examples/Pointclouds/images/sphx_glr_point_cloud_3d_002.png
    :class: sphx-glr-single-img




Grid the points using a triangulated CloughTocher interpolation


.. code-block:: default


    plt.figure()
    PC3D.mapPlot(method='ct')





.. image:: /examples/Pointclouds/images/sphx_glr_point_cloud_3d_003.png
    :class: sphx-glr-single-img




We can perform spatial searches on the 3D point cloud


.. code-block:: default


    PC3D.setKdTree(nDims=2)
    p = PC3D.nearest((0.0,0.0), k=200, p=2, radius=0.3)
    print(p)








.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    (array([0.01055274, 0.02684564, 0.02760331, ..., 0.16645792, 0.16760366,
           0.16789218]), array([5082, 3935, 4407, ..., 1086, 6479, 3661]))



.nearest returns the distances and indices into the point cloud of the nearest points.
We can then obtain those points as another point cloud


.. code-block:: default


    pNear = PC3D[p[1]]
    plt.figure()
    ax1 = plt.subplot(1,2,1)
    pNear.scatter2D()
    plt.plot(0.0, 0.0, 'x')
    plt.subplot(1,2,2, sharex=ax1, sharey=ax1)
    ax = PC3D.scatter2D(edgecolor='k')
    searchRadius = plt.Circle((0.0, 0.0), 0.3, color='b', fill=False)
    ax.add_artist(searchRadius)
    plt.plot(0.0, 0.0, 'x')





.. image:: /examples/Pointclouds/images/sphx_glr_point_cloud_3d_004.png
    :class: sphx-glr-single-img




Read in the xyz co-ordinates in columns 2,3,4 from a file. Skip 1 header line.


.. code-block:: default


    dataFolder = "..//supplementary//Data//"

    PC3D.read(fileName=dataFolder + 'Resolve1.txt', nHeaderLines=1, columnIndices=[2,3,4])









.. code-block:: default



    plt.figure()
    f = PC3D.scatter2D(s=10)




.. image:: /examples/Pointclouds/images/sphx_glr_point_cloud_3d_005.png
    :class: sphx-glr-single-img




Export the 3D Pointcloud to a VTK file.

In this case, I pass the height as point data so that the points are coloured
when opened in Paraview (or other software)


.. code-block:: default



    PC3D.toVTK('testPoints', format='binary')







.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  5.508 seconds)


.. _sphx_glr_download_examples_Pointclouds_point_cloud_3d.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: point_cloud_3d.py <point_cloud_3d.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: point_cloud_3d.ipynb <point_cloud_3d.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
