.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_examples_Meshes_plot_rectilinear_mesh_2d.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_examples_Meshes_plot_rectilinear_mesh_2d.py:


2D Rectilinear Mesh
-------------------
This 2D rectilinear mesh defines a grid with straight cell boundaries.

It can be instantiated in two ways.  

The first is by providing the cell centres or
cell edges in two dimensions.

The second embeds the 2D mesh in 3D by providing the cell centres or edges in three dimensions.  
The first two dimensions specify the mesh coordinates in the horiztontal cartesian plane
while the third discretizes in depth. This allows us to characterize a mesh whose horizontal coordinates
do not follow a line that is parallel to either the "x" or "y" axis.


.. code-block:: default

    from geobipy import StatArray
    from geobipy import RectilinearMesh2D
    import matplotlib.pyplot as plt
    import numpy as np









Specify some cell centres in x and y


.. code-block:: default

    x = StatArray(np.arange(10.0), 'Easting', 'm')
    y = StatArray(np.arange(10.0), 'Northing', 'm')
    rm = RectilinearMesh2D(xCentres=x, yCentres=y)









We can plot the grid lines of the mesh.


.. code-block:: default

    plt.figure()
    _  = rm.plotGrid(linewidth=0.5)





.. image:: /examples/Meshes/images/sphx_glr_plot_rectilinear_mesh_2d_001.png
    :alt: plot rectilinear mesh 2d
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customPlots.py:649: MatplotlibDeprecationWarning: You are modifying the state of a globally registered colormap. In future versions, you will not be able to modify a registered colormap in-place. To remove this warning, you can make a copy of the colormap first. cmap = copy.copy(mpl.cm.get_cmap("viridis"))
      kwargs['cmap'].set_bad(color='white')




2D Mesh embedded in 3D
++++++++++++++++++++++


.. code-block:: default

    z = StatArray(np.cumsum(np.arange(15.0)), 'Depth', 'm')
    rm = RectilinearMesh2D(xCentres=x, yCentres=y, zCentres=z)








Plot the x-y coordinates of the mesh


.. code-block:: default

    plt.figure()
    _ = rm.plotXY()




.. image:: /examples/Meshes/images/sphx_glr_plot_rectilinear_mesh_2d_002.png
    :alt: plot rectilinear mesh 2d
    :class: sphx-glr-single-img





Again, plot the grid. This time the z-coordinate dominates the plot.


.. code-block:: default

    plt.figure()
    _ = rm.plotGrid(xAxis='r', flipY=True, linewidth=0.5)




.. image:: /examples/Meshes/images/sphx_glr_plot_rectilinear_mesh_2d_003.png
    :alt: plot rectilinear mesh 2d
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customPlots.py:649: MatplotlibDeprecationWarning: You are modifying the state of a globally registered colormap. In future versions, you will not be able to modify a registered colormap in-place. To remove this warning, you can make a copy of the colormap first. cmap = copy.copy(mpl.cm.get_cmap("viridis"))
      kwargs['cmap'].set_bad(color='white')




We can pcolor the mesh by providing cell values.


.. code-block:: default

    arr = StatArray(np.random.random(rm.shape), 'Name', 'Units')

    plt.figure()
    _ = rm.pcolor(arr, xAxis='r', grid=True, flipY=True, linewidth=0.5)




.. image:: /examples/Meshes/images/sphx_glr_plot_rectilinear_mesh_2d_004.png
    :alt: plot rectilinear mesh 2d
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/customPlots.py:649: MatplotlibDeprecationWarning: You are modifying the state of a globally registered colormap. In future versions, you will not be able to modify a registered colormap in-place. To remove this warning, you can make a copy of the colormap first. cmap = copy.copy(mpl.cm.get_cmap("viridis"))
      kwargs['cmap'].set_bad(color='white')




We can perform some interval statistics on the cell values of the mesh
Generate some values


.. code-block:: default

    a = np.repeat(np.arange(1.0, np.float(rm.x.nCells+1))[:, np.newaxis], rm.z.nCells, 1).T









Compute the mean over an interval for the mesh.


.. code-block:: default

    rm.intervalStatistic(a, intervals=[6.8, 12.4], axis=0, statistic='mean')






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    (array([[ 1.,  2.,  3., ...,  8.,  9., 10.]]), [6.8, 12.4])



Compute the mean over multiple intervals for the mesh.


.. code-block:: default

    rm.intervalStatistic(a, intervals=[6.8, 12.4, 20.0, 40.0], axis=0, statistic='mean')






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    (array([[ 1.,  2.,  3., ...,  8.,  9., 10.],
           [ 1.,  2.,  3., ...,  8.,  9., 10.],
           [ 1.,  2.,  3., ...,  8.,  9., 10.]]), [6.8, 12.4, 20.0, 40.0])



We can specify either axis


.. code-block:: default

    rm.intervalStatistic(a, intervals=[2.8, 4.2], axis=1, statistic='mean')






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    (array([[4.5],
           [4.5],
           [4.5],
           ...,
           [4.5],
           [4.5],
           [4.5]]), [2.8, 4.2])




.. code-block:: default

    rm.intervalStatistic(a, intervals=[2.8, 4.2, 5.1, 8.4], axis=1, statistic='mean')






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    (array([[4.5, 6. , 8. ],
           [4.5, 6. , 8. ],
           [4.5, 6. , 8. ],
           ...,
           [4.5, 6. , 8. ],
           [4.5, 6. , 8. ],
           [4.5, 6. , 8. ]]), [2.8, 4.2, 5.1, 8.4])



rm.toVTK('test', cellData=StatArray(np.random.randn(z.size, x.size), "Name"))


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.377 seconds)


.. _sphx_glr_download_examples_Meshes_plot_rectilinear_mesh_2d.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_rectilinear_mesh_2d.py <plot_rectilinear_mesh_2d.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_rectilinear_mesh_2d.ipynb <plot_rectilinear_mesh_2d.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
