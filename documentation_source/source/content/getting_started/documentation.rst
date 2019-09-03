Documentation
=============

Publication
:::::::::::
The code and its processes have been documented in multiple ways.  First we have the publication associated with this software release, the citation is below, and presents the application of this package to frequency and time domain electro-magnetic inversion.

Source code HTML pages
::::::::::::::::::::::
For developers and users of the code, the code itself has been thouroughly documented. The `source code docs can be found here`_

.. _`source code docs can be found here`: https://usgs.github.io/geobipy/

However you can generate the docs locally as well. To do this, you will first need to install sphinx via "pip install sphinx".

Next, head to the documentation folder in this repository and type "make html".  Sphinx generates linux based and windows based make files so this should be a cross-platform procedure.

The html pages will be generated under "build/html", so simply open the "index.html" file to view and navigate the code.

Jupyter notebooks to illustrate the classes
:::::::::::::::::::::::::::::::::::::::::::
For more practical, hands-on documentation, we have also provided jupyter notebooks under the documentation/notebooks folder.  These notebooks illustrate how to use each class in the package.

You will need to install jupyter via "pip install jupyter".

You can then edit and run the notebooks by navigating to the notebooks folder, and typing "jupyter notebook". This will open up a new browser window, and you can play in there.
