.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_examples_Meshes_plot_rectilinear_mesh_1d.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_examples_Meshes_plot_rectilinear_mesh_1d.py:


1D Rectilinear Mesh
-------------------


.. code-block:: default

    from geobipy import StatArray
    from geobipy import RectilinearMesh1D
    import matplotlib.pyplot as plt
    import numpy as np









Instantiate a new 1D rectilinear mesh by specifying cell centres or edges.
Here we use edges


.. code-block:: default

    x = StatArray(np.cumsum(np.arange(0.0, 10.0)), 'Depth', 'm')









.. code-block:: default

    rm = RectilinearMesh1D(edges=x)










.. code-block:: default

    print(rm.centres)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [ 0.5  2.   4.5 ... 24.5 32.  40.5]





.. code-block:: default

    print(rm.edges)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [ 0.  1.  3. ... 28. 36. 45.]





.. code-block:: default

    print(rm.internaledges)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [ 1.  3.  6. ... 21. 28. 36.]





.. code-block:: default

    print(rm.cellWidths)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [1. 2. 3. ... 7. 8. 9.]




Get the cell indices


.. code-block:: default

    print(rm.cellIndex(np.r_[1.0, 5.0, 20.0]))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [1 2 5]




We can plot the grid of the mesh


.. code-block:: default

    plt.figure()
    _ = rm.plotGrid(flipY=True)





.. image:: /examples/Meshes/images/sphx_glr_plot_rectilinear_mesh_1d_001.png
    :alt: plot rectilinear mesh 1d
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/plotting.py:874: MatplotlibDeprecationWarning: shading='flat' when X and Y have the same dimensions as C is deprecated since 3.3.  Either specify the corners of the quadrilaterals with X and Y, or pass shading='auto', 'nearest' or 'gouraud', or set rcParams['pcolor.shading'].  This will become an error two minor releases later.
      pm = ax.pcolormesh(X, Y, v, color=c, **kwargs)




Or Pcolor the mesh showing. An array of cell values is used as the colour.


.. code-block:: default

    plt.figure()
    arr = StatArray(np.random.randn(rm.nCells), "Name", "Units")
    _ = rm.pcolor(arr, grid=True, flipY=True)





.. image:: /examples/Meshes/images/sphx_glr_plot_rectilinear_mesh_1d_002.png
    :alt: plot rectilinear mesh 1d
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/plotting.py:874: MatplotlibDeprecationWarning: shading='flat' when X and Y have the same dimensions as C is deprecated since 3.3.  Either specify the corners of the quadrilaterals with X and Y, or pass shading='auto', 'nearest' or 'gouraud', or set rcParams['pcolor.shading'].  This will become an error two minor releases later.
      pm = ax.pcolormesh(X, Y, v, color=c, **kwargs)




Instantiate a new 1D rectilinear mesh by specifying cell centres or edges.
Here we use edges


.. code-block:: default

    x = StatArray(np.logspace(-3, 3, 10), 'Depth', 'm')









.. code-block:: default

    rm = RectilinearMesh1D(edges=x, log=10)









Access property describing the mesh


.. code-block:: default

    print(rm.centres)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [-2.66666667 -2.         -1.33333333 ...  1.33333333  2.
      2.66666667]





.. code-block:: default

    print(rm.edges)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [-3.         -2.33333333 -1.66666667 ...  1.66666667  2.33333333
      3.        ]





.. code-block:: default

    print(rm.internaledges)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [-2.33333333 -1.66666667 -1.         ...  1.          1.66666667
      2.33333333]





.. code-block:: default

    print(rm.cellWidths)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [0.66666667 0.66666667 0.66666667 ... 0.66666667 0.66666667 0.66666667]




Get the cell indices


.. code-block:: default

    print(rm.cellIndex(np.r_[0.03, 5.0, 200.0]))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [2 5 7]




We can plot the grid of the mesh


.. code-block:: default

    plt.figure()
    _ = rm.plotGrid(flipY=True)





.. image:: /examples/Meshes/images/sphx_glr_plot_rectilinear_mesh_1d_003.png
    :alt: plot rectilinear mesh 1d
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/plotting.py:874: MatplotlibDeprecationWarning: shading='flat' when X and Y have the same dimensions as C is deprecated since 3.3.  Either specify the corners of the quadrilaterals with X and Y, or pass shading='auto', 'nearest' or 'gouraud', or set rcParams['pcolor.shading'].  This will become an error two minor releases later.
      pm = ax.pcolormesh(X, Y, v, color=c, **kwargs)




Or Pcolor the mesh showing. An array of cell values is used as the colour.


.. code-block:: default

    plt.figure()
    arr = StatArray(np.random.randn(rm.nCells), "Name", "Units")
    _ = rm.pcolor(arr, grid=True, flipY=True)


    import h5py
    with h5py.File('rm1d.h5', 'w') as f:
        rm.toHdf(f, 'rm1d')

    with h5py.File('rm1d.h5', 'r') as f:
        rm1 = RectilinearMesh1D().fromHdf(f['rm1d'])


.. image:: /examples/Meshes/images/sphx_glr_plot_rectilinear_mesh_1d_004.png
    :alt: plot rectilinear mesh 1d
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Users/nfoks/codes/repositories/geobipy/geobipy/src/base/plotting.py:874: MatplotlibDeprecationWarning: shading='flat' when X and Y have the same dimensions as C is deprecated since 3.3.  Either specify the corners of the quadrilaterals with X and Y, or pass shading='auto', 'nearest' or 'gouraud', or set rcParams['pcolor.shading'].  This will become an error two minor releases later.
      pm = ax.pcolormesh(X, Y, v, color=c, **kwargs)





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.371 seconds)


.. _sphx_glr_download_examples_Meshes_plot_rectilinear_mesh_1d.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_rectilinear_mesh_1d.py <plot_rectilinear_mesh_1d.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_rectilinear_mesh_1d.ipynb <plot_rectilinear_mesh_1d.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
