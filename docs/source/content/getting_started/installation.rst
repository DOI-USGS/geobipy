Installing GeoBIPy
==================

First things first, install a Python 3.5+ distribution.  This is the minimum version that we have tested with.

This package has a few requirements depending on what you wish to do with it.

If you require a serial version of the code, see `Installing a serial version of GeoBIPy`_.

If you require an parallel implementation, you will need to install an MPI library, and Python's mpi4py module. See `Installing MPI and mpi4py`_.

If you require parallel file reading and writing, you will also need to install an MPI enabled HDF5 library, as well as Python's h5py wrapper to that library. It is important to read the notes below on installing h5py on top of a parallel HDF library.  The traditional "pip install h5py" will not work correctly. See `Installing parallel HDF5 and h5py`_ to do this correctly.

If you need to install the parallel IO version of the code, we would recommend that you start with a clean install of Python. This makes it easier to determine whether you have installed and linked the correct version of the parallel HDF5 library.


There are two versions when installing GeoBIPy, a serial version, and a parallel version. Since GeoBIPy uses a Fortran backend for forward modelling frequency domain data, you will need to have a Fortran compiler installed. Make sure that the compiler can handle derived data types since I make use of object oriented programming in Fortran.

Installing a serial version of GeoBIPy
::::::::::::::::::::::::::::::::::::::
This is the easiest installation and provides access to a serial implementation of the code.

Simply clone the git repository, navigate to the package folder that contains the setup.py file, and type "pip install -e ."

You should then be able to import modules from geobipy.  For this type of installation mpi will not need to be installed, and the serial version of h5py will suffice i.e. the standard "pip install h5py" is fine.  h5py will automatically be installed during the install of GeoBIPy since it is a dependency.

**Side note:**  Let's say you ran a production run on a parallel machine with MPI and parallel HDF capabilities. You generated all the results, copied them back to your local machine, and wish to make plots and images.  You will only need to install the serial version of the code on your local machine to do this.

Installing a parallel version of GeoBIPy
::::::::::::::::::::::::::::::::::::::::
Installing the parallel version of the code is a little trickier due to the dependencies necessary between the OpenMPI and/or HDF libraries, and how Python's mpi4py and h5py wrap around those.


Installing MPI and mpi4py
-------------------------
To run this code in parallel you will need both an MPI library and the python wrapper, mpi4py.  You must install MPI first before mpi4py.

MPI
+++

If you are installing GeoBIPy on a parallel machine, I would think that you have access to prebuilt MPI libraries.  If you are on a local laptop, you will need to install one. This package has been tested with OpenMPI version 1.10.2. Be careful if you want to use a newer version as mpi4py may not communicate with it correctly (at the time of this writing, OpenMPI v2 was having issues).


mpi4py
++++++

At this point, if you have an mpi4py module already installed, please remove it (you can check with "pip list"). If you started with a clean installation you should not have to worry about this. To test whether a new install of mpi4py will see the mpi library you have, just type "which mpicc".  The path that you see should point to the implementation that you want mpi4py to link to.  Make sure you are about to install mpi4py to the correct python installation.  If you type 'which python' it should return the path to the correct python distribution.  If you are using environments, make sure you have activated the correct one.

Next, use "pip install mpi4py --no-cache-dir".  This last option is very important, without it, pip might install its own MPI library called MPICH2. I would try to avoid this because if you need to install the HDF5 library you will need know which directories to link to (see `Installing parallel HDF5 and h5py`_).

At the end of the day,  h5py needs to communicate with both the correct HDF5 library and mpi4py, and both of those need to communicate with the same MPI library.

Installing parallel HDF5 and h5py
---------------------------------
If a parallel HDF5 library is not available, you will need to install one. First make sure you follow `Installing MPI and mpi4py`_ so that an MPI library is available to you. You must install a HDF5 library first before h5py.

HDF5
++++
When you install HDF5, make sure that the correct MPI library can be seen by typing "which mpicc".  When you configure the HDF5 library, be sure to use the --enable-parallel option.

h5py
++++
Once the HDF5 library is installed you will need to clone the `h5py repository`_

.. _`h5py repository`: https://github.com/h5py/h5py

Make sure you are about to install h5py to the correct python installation.  If you type 'which python' it should return the path to the correct python installation.

Next, copy the following code into a file called install.sh in the h5py folder and run it.  You will need to edit 3 entries.

- In H5PY_PATH change the path to the location where you want h5py installed.
- In HDF5_PATH change the path to the location of the installed parallel HDF5 library (i.e. the directory above /lib/)
- Check that 'which mpicc' returns the correct version.

.. code:: bash

    #!/bin/bash
    export HDF5_PATH=<Your path to HDF5>
    python setup.py clean --all
    python setup.py configure -r --hdf5-version=<Your version of HDF5> --mpi --hdf5=$HDF5_PATH
    export gcc=gcc
    CC=mpicc HDF5_DIR=$HDF5_PATH python setup.py build
    python setup.py install


Installing the time domain forward modeller
:::::::::::::::::::::::::::::::::::::::::::
Ross Brodie at Geoscience Australia has written a great forward modeller, gatdaem1D,  in C++ with a python interface.  You can obtain that code here at the `GA repository`_

.. _`GA repository`: https://github.com/GeoscienceAustralia/ga-aem

However, for use with GeoBIPy, use `this fork of gataem1D`_ if there are open pull requests at the original repository.

.. _`this fork of gataem1D`: https://github.com/leonfoks/ga-aem

Go ahead and "git clone" that repository.

These instructions only describe how to install Ross' forward modeller, but it is part of a larger code base for inversion. If you wish to install his entire package, please follow his instructions.

Prerequisites
-------------

To compile his forward modeller, you will need a c++ compiler, and `FFTW`_

.. _`FFTW`: http://www.fftw.org/

On a Mac, installing these two items is easy if you use a package manager such as `homebrew`_

.. _`homebrew`: https://brew.sh/

If you use brew, simply do the following

.. code:: bash

   brew install gcc
   brew install fftw

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
Next, within the gatdaem1d folder, navigate to the makefiles folder and modify the top part of the file "gatdaem1d_python.make" to the following

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
------------------------------

Finally, to install the python wrapper to gatdaem1d, navigate to the python folder of the gatdaem1d repository.
Type,

.. code:: bash

  pip install .

You should now have access to the time domain forward modeller within geobipy.