#!/usr/bin/env python

import sys
# Test Python's version
major, minor = sys.version_info[0:2]
if (major, minor) < (3, 5):
    sys.stderr.write('\nPython 3.5 or later is needed to use this package\n')
    sys.exit(1)
from setuptools import find_packages
try:
    from numpy.distutils.core import setup
    from numpy.distutils.core import Extension
except ImportError:
    pass

def readme():
    with open('README.rst', encoding='utf-8', mode='r') as f:
        return f.read()

setup(name='geobipy',
    packages=find_packages(),
    scripts=[],
    version=1.0,
    description='Markov chain Monte Carlo inversion',
    long_description=readme(),
    url = 'https://github.com/usgs/geobipy',
    classifiers=[
        'Development Status :: 0 - Alpha',
        'License :: OSI Approved :: CC0 and GPLv2 and BSD3',
        'Programming Language :: Python :: 3.x',
        'Topic :: Geophysical Inversion :: Bayesian :: McMC',
    ],
    author='Leon Foks',
    author_email='nfoks@contractor.usgs.gov',
    install_requires=[
        'numpy >= 1.11',
        'scipy >= 0.18.1',
        'h5py >= 2.6.0',
        'netcdf4',
        'sklearn',
        'matplotlib',
        'pyvtk',
        'sphinx',
        'progressbar2',
        'numba',
        'cached-property',
        'empymod',
        'smm'
    ],
    # ext_modules=[Extension(name='geobipy.src.classes.forwardmodelling.ipforward1d_fortran',
    #             extra_f90_compile_args = ['-ffree-line-length-none','-O3', '-finline-functions', '-funroll-all-loops'],
    #             extra_link_args = ['-ffast-math','-ffree-line-length-none', '-O3', '-finline-functions', '-funroll-all-loops', '-g0'],
    #             sources=['geobipy/src/classes/forwardmodelling/ipforward1D_fortran/m_ipforward1D.f90'],
	# 	 ),
    #              ],
    entry_points = {
        'console_scripts':[
            'geobipy=geobipy:geobipy',
            'geobipy_mpi=geobipy:geobipy_mpi',
        ],
    }
)

