Welcome to GeoBIPy
~~~~~~~~~~~~~~~~~~~
Geophysical Bayesian Inference in Python

This package uses a Bayesian formulation and Markov chain Monte Carlo sampling methods to derive posterior distributions of subsurface and measured data properties. The current implementation is applied to time and frequency domain electro-magnetic data. Application outside of these data types is well within scope.

Currently there are two types of data that we have implemented; frequency domain electromagnetic data, and time domain electromagnetic data. The package comes with a frequency domain forward modeller, but it does not come with a time domain forward modeller.  See the section `Installing the time domain forward modeller`_ for more information.

.. contents:: Table of Contents

Getting started
=================
First things first, install a Python 3.5+ distribution.  This is the minimum version that we have tested with.

This package has a few requirements depending on what you wish to do with it.

If you require a serial version of the code, see `Installing a serial version of GeoBIPy`_.

If you require an parallel implementation, you will need to install an MPI library, and Python's mpi4py module. See `Installing MPI and mpi4py`_.

If you require parallel file reading and writing, you will also need to install an MPI enabled HDF5 library, as well as Python's h5py wrapper to that library. It is important to read the notes below on installing h5py on top of a parallel HDF library.  The traditional "pip install h5py" will not work correctly. See `Installing parallel HDF5 and h5py`_ to do this correctly.

If you need to install the parallel IO version of the code, we would recommend that you start with a clean install of Python. This makes it easier to determine whether you have installed and linked the correct version of the parallel HDF5 library.


Installing a serial version of GeoBIPy
=======================================
This is the easiest installation and provides access to a serial implementation of the code.

Simply clone the git repository, navigate to the package folder that contains the setup.py file, and type "pip install -e ."

You should then be able to import modules from geobipy.src.  For this type of installation mpi will not need to be installed, and the serial version of h5py will suffice i.e. the standard "pip install h5py" is fine.

**Side note:**  Let's say you ran a production run on a parallel machine with MPI and parallel HDF capabilities. You generated all the results, copied them back to your local machine, and wish to make plots and images.  You will only need to install the serial version of the code on your local machine to do this.

Installing a parallel version of GeoBIPy
=========================================
Installing the parallel version of the code is a little trickier due to the dependencies necessary between the OpenMPI and/or HDF libraries, and how Python's mpi4py and h5py wrap around those.


Installing MPI and mpi4py
:::::::::::::::::::::::::
To run this code in parallel you will need both an MPI library and the python wrapper, mpi4py.  You must install MPI first before mpi4py.

MPI
---

If you are installing GeoBIPy on a parallel machine, I would think that you have access to prebuilt MPI libraries.  If you are on a local laptop, you will need to install one. This package has been tested with OpenMPI version 1.10.2. Be careful if you want to use a newer version as mpi4py may not communicate with it correctly (at the time of this writing, OpenMPI v2 was having issues).


mpi4py
------

At this point, if you have an mpi4py module already installed, please remove it (you can check with "pip list"). If you started with a clean installation you should not have to worry about this. To test whether a new install of mpi4py will see the mpi library you have, just type "which mpicc".  The path that you see should point to the implementation that you want mpi4py to link to.  Make sure you are about to install mpi4py to the correct python installation.  If you type 'which python' it should return the path to the correct python distribution.  If you are using environments, make sure you have activated the correct one.

Next, use "pip install mpi4py --no-cache-dir".  This last option is very important, without it, pip might install its own MPI library called MPICH2. I would try to avoid this because if you need to install the HDF5 library you will need know which directories to link to (see `Installing parallel HDF5 and h5py`_).

At the end of the day,  h5py needs to communicate with both the correct HDF5 library and mpi4py, and both of those need to communicate with the same MPI library.

Installing parallel HDF5 and h5py
:::::::::::::::::::::::::::::::::
If a parallel HDF5 library is not available, you will need to install one. First make sure you follow `Installing MPI and mpi4py`_ so that an MPI library is available to you. You must install a HDF5 library first before h5py.

HDF5
----
When you install HDF5, make sure that the correct MPI library can be seen by typing "which mpicc".  When you configure the HDF5 library, be sure to use the --enable-parallel option.

h5py
----
Once the HDF5 library is installed you will need to clone the `h5py repository`_

.. _`h5py repository`: https://github.com/h5py/h5py

Make sure you are about to install h5py to the correct python installation.  If you type 'which python' it should return the path to the correct python installation.

Next, copy the following code into a file called install.sh in the h5py folder and run it.  You will need to edit 3 entries.

- In H5PY_PATH change the path to the location where you want h5py installed.
- In HDF5_PATH change the path to the location of the installed parallel HDF5 library (i.e. the directory above /lib/)
- Check that 'which mpicc' returns the correct version.

.. code:: bash

    #!/bin/bash
    module load HDF5-Parallel openmpi-library
    export H5PY_PATH=/path/to/install/h5py
    mkdir -p $H5PY_PATH
    export HDF5_PATH=/path/to/parallelHdf5
    python setup.py clean --all
    python setup.py configure -r
    python setup.py configure --hdf5-version=1.10.2
    python setup.py configure --mpi
    export CC=mpicc
    export gcc=mpicc
    export PYTHONPATH=$PYTHONPATH:$H5PY_PATH
    python setup.py configure --hdf5=$HDF5_PATH
    HDF5_DIR=$HDF5_PATH python setup.py build
    python setup.py install


Installing GeoBIPy on Yeti
:::::::::::::::::::::::::::
If you are installing this package on the USGS machine,  you can bypass all installations regarding MPI, HDF5, and Python's mpi4py and h5py modules.  On Yeti we have a module that you can load using "module load python/pPython3".  This module comes with the bare essentials for parallel python with a working parallel h5py wrapper.

First, you need to create your own environment after you have loaded the pPython module.  This creates a brand new installation directory for you and allows you to install any extra modules yourself.  Do this using 'conda create --name aName'.

You can the activate that environment using 'source activate aName'.

Next pull the GeoBIPy repository and navigate to that folder.  There should be a setup.py file.  In this folder type "pip install -e ." to install the package to python.

You will also need to install the time domain forward modeller.


Installing the time domain forward modeller
:::::::::::::::::::::::::::::::::::::::::::
Ross Brodie at Geoscience Australia has written a great forward modeller, gatdaem1D,  in C++ with a python interface.  You can obtain that code here at the `GA repository`_

.. _`GA repository`: https://github.com/GeoscienceAustralia/ga-aem

So go ahead and "git clone" that repository.

These instructions only describe how to install Ross' forward modeller, but it is part of a larger code base for deterministic inversion. If you wish to install his entire package, please follow his instructions.

Prerequisites
-------------

To compile this forward modeller, you will need a c++ compiler, and `FFTW`_

.. _`FFTW`: http://www.fftw.org/

On a Mac, installing these two items is easy if you use a package manager such as `homebrew`_

.. _`homebrew`: https://brew.sh/

If you use brew, simply do the following

.. code:: bash

   brew install gcc
   brew install fftw

Installing FFTW from Source
+++++++++++++++++++++++++++

If you do not have brew, or use a package manager, you can install fftw from source instead.

Download fftw-3.3.7.tar.gz from the `FFTW downloads`_ .

.. _`FFTW downloads`: http://www.fftw.org/download.html

Untar the folder and install fftw using the following.

.. code:: bash

  tar -zxvf fftw-3.3.7.tar.gz
  cd fftw-3.3.7
  mkdir build
  cd build
  ../configure --prefix=path-to-install-to/fftw-3.3.7 --enable-threads
  make
  make install

where, path-to-install-to is the location where you want fftw to be installed.


Compile the gatdaem1d shared library
------------------------------------
Next, within the gatdaem1d folder, navigate to the makefiles folder modify the top part of the file "gatdaem1d_python.make" to the following

.. code:: bash

  SHELL = /bin/sh
  .SUFFIXES:
  .SUFFIXES: .cpp .o
  cxx = g++
  cxxflags = -std=c++11 -O3 -Wall -fPIC
  FFTW_DIR = path-to-fftw

  ldflags    += -shared
  bindir     = ../python/gatdaem1d

  srcdir     = ../src
  objdir     = ./obj
  includes   = -I$(srcdir) -I$(FFTW_DIR)/include
  libs       = -L$(FFTW_DIR)/lib -lfftw3
  library    = $(bindir)/gatdaem1d.so

You can find out where brew installed fftw by typing

.. code:: bash

  brew info fftw

Which may return something like "/usr/local/Cellar/fftw/3.3.5"

In this case, path-to-fftw is "/usr/local/Cellar/fftw/3.3.5"

If you installed fftw from source, then path-to-fftw is that install path.

Next, type the following to compile the gatdaem1d c++ code.

.. code:: bash

  make -f gatdaem1d_python.make

Installing the Python Bindings
::::::::::::::::::::::::::::::

Finally, to install the python wrapper to gatdaem1d, navigate to the python folder of the gatdaem1d repository.
Type,

.. code:: bash

  pip install .

You should now have access to the time domain forward modeller within geobipy.

Documentation
=============

Publication
:::::::::::
The code and its processes have been documented in multiple ways.  First we have the publication associated with this software release, the citation is below, and presents the application of this package to frequency and time domain electro-magnetic inversion.

Source code HTML pages
::::::::::::::::::::::
For developers and users of the code, the code itself has been thouroughly documented. However you can generate easy to read html pages. To do this, you will first need to install sphinx via "pip install sphinx".

Next, head to the documentation folder in this repository and type "make html".  Sphinx generates linux based and windows based make files so this should be a cross-platform procedure.

The html pages will be generated under "html", so simply open the "index.html" file to view and navigate the code.

Jupyter notebooks to illustrate the classes
:::::::::::::::::::::::::::::::::::::::::::
For more practical, hands-on documentation, we have also provided jupyter notebooks under the documentation/notebooks folder.  These notebooks illustrate how to use each class in the package.

You will need to install jupyter via "pip install jupyter".

You can then edit and run the notebooks by navigating to the notebooks folder, and typing "jupyter notebook". This will open up a new browser window, and you can play in there.
