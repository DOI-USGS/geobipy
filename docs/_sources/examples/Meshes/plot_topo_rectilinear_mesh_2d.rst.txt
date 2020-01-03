.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_examples_Meshes_plot_topo_rectilinear_mesh_2d.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_Meshes_plot_topo_rectilinear_mesh_2d.py:


Topo Rectilinear Mesh 2D
------------------------
The Topo version of the rectilinear mesh has the same functionality as the
:ref:`Rectilinear Mesh 2D` but the top surface of the mesh can undulate.


.. code-block:: default

    from geobipy import StatArray
    from geobipy import TopoRectilinearMesh2D
    import matplotlib.pyplot as plt
    import numpy as np









Specify some cell centres in x and y


.. code-block:: default

    x = StatArray(np.arange(10.0), 'Easting', 'm')
    y = StatArray(np.arange(10.0), 'Height', 'm')
    # Create a height profile for the mesh
    height = StatArray(np.asarray([5,4,3,2,1,1,2,3,4,5])*3.0, 'Height', 'm')
    # Instantiate the mesh
    rm = TopoRectilinearMesh2D(xCentres=x, yCentres=y, heightCentres=height)








Plot only the grid lines of the mesh


.. code-block:: default

    plt.figure()
    _ = rm.plotGrid(linewidth=0.5)




.. image:: /examples/Meshes/images/sphx_glr_plot_topo_rectilinear_mesh_2d_001.png
    :class: sphx-glr-single-img





Create some cell values


.. code-block:: default

    values = StatArray(np.random.random(rm.shape), 'Name', 'Units')









.. code-block:: default

    plt.figure()
    _ = rm.pcolor(values, grid=True, linewidth=0.1, xAxis='x')




.. image:: /examples/Meshes/images/sphx_glr_plot_topo_rectilinear_mesh_2d_002.png
    :class: sphx-glr-single-img





Compute the mean over an interval for the mesh.


.. code-block:: default

    rm.intervalStatistic(values, intervals=[6.8, 12.4], axis=0)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    (array([[0.58441549, 0.5535671 , 0.16548102, ..., 0.48121875, 0.42072505,
            0.45478709]]), [6.8, 12.4])



Compute the mean over multiple intervals for the mesh.


.. code-block:: default

    rm.intervalStatistic(values, intervals=[6.8, 12.4, 20.0, 40.0], axis=0)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    (array([[0.58441549, 0.5535671 , 0.16548102, ..., 0.48121875, 0.42072505,
            0.45478709]]), [6.8, 12.4])



We can apply the interval statistics to either axis


.. code-block:: default

    rm.intervalStatistic(values, intervals=[2.8, 4.2], axis=1)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    (array([[0.47300654],
           [0.20643969],
           [0.45939022],
           ...,
           [0.86799162],
           [0.41946334],
           [0.2940569 ]]), [2.8, 4.2])




.. code-block:: default

    rm.intervalStatistic(values, intervals=[2.8, 4.2, 5.1, 8.4], axis=1)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    (array([[0.47300654, 0.97363323, 0.56656719],
           [0.20643969, 0.7953032 , 0.49412092],
           [0.45939022, 0.03773808, 0.47252983],
           ...,
           [0.86799162, 0.34502044, 0.57176244],
           [0.41946334, 0.887073  , 0.2040596 ],
           [0.2940569 , 0.89267787, 0.52299616]]), [2.8, 4.2, 5.1, 8.4])




.. code-block:: default

    rm.ravelIndices([[3, 4], [5, 5]])






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    array([35, 45])




.. code-block:: default

    rm.unravelIndex([35, 45])






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    (array([3, 4]), array([5, 5]))



2D Topo rectlinear mesh embedded in 3D
++++++++++++++++++++++++++++++++++++++


.. code-block:: default

    z = StatArray(np.cumsum(np.arange(10.0)), 'Depth', 'm')
    rm = TopoRectilinearMesh2D(xCentres=x, yCentres=y, zCentres=z, heightCentres=height)
    values = StatArray(np.arange(rm.nCells, dtype=np.float).reshape(rm.shape), 'Name', 'Units')










.. code-block:: default

    plt.figure()
    rm.plotGrid(linewidth=1)




.. image:: /examples/Meshes/images/sphx_glr_plot_topo_rectilinear_mesh_2d_003.png
    :class: sphx-glr-single-img





Plot the x-y co-ordinates


.. code-block:: default

    plt.figure()
    rm.plotXY()




.. image:: /examples/Meshes/images/sphx_glr_plot_topo_rectilinear_mesh_2d_004.png
    :class: sphx-glr-single-img





The pcolor function can now be plotted against distance


.. code-block:: default

    plt.figure()
    rm.pcolor(values, grid=True, xAxis='r', linewidth=0.5)





.. image:: /examples/Meshes/images/sphx_glr_plot_topo_rectilinear_mesh_2d_005.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    (<matplotlib.axes._subplots.AxesSubplot object at 0x12cbd64d0>, <matplotlib.collections.QuadMesh object at 0x12cefa390>, <matplotlib.colorbar.Colorbar object at 0x131a2aad0>)



rm.toVTK('test', cellData=values)


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.446 seconds)


.. _sphx_glr_download_examples_Meshes_plot_topo_rectilinear_mesh_2d.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_topo_rectilinear_mesh_2d.py <plot_topo_rectilinear_mesh_2d.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_topo_rectilinear_mesh_2d.ipynb <plot_topo_rectilinear_mesh_2d.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
