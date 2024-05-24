#!/usr/bin/env python

import sys
# Test Python's version
major, minor = sys.version_info[0:2]
from setuptools import find_packages, setup
from distutils.command.sdist import sdist
cmdclass={'sdist': sdist}
#try:
#    from numpy.distutils.core import setup
#    from numpy.distutils.core import Extension
#except ImportError:
#    pass

def readme():
    with open('README.rst', encoding='utf-8', mode='r') as f:
        return f.read()

__version__ = '2.0.0'

setup(name='geobipy',
    packages=find_packages(),
    scripts=[],
    version=__version__,
    description='Markov chain Monte Carlo inversion',
    long_description=readme(),
    url = 'https://github.com/DOI-USGS/geobipy',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved',
        'License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ],
    author='Leon Foks',
    author_email='nfoks@contractor.usgs.gov',
    install_requires=[
        'numpy',
        'scipy',
        'h5py',
        'numba',
        'pandas',
        'netcdf4',
#        'sklearn',
        'matplotlib',
        'pyvista',
        'sphinx',
        'progressbar2',
        'cached-property',
        'empymod',
        'smm',
        'lmfit',
        'scikit-learn',
        'randomgen',
        'numba_kdtree',
        'pygmt'
    ],
    extra_requires=['sphinx_gallery',
                    "sphinx_rtd_theme"

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
        ],
    }
)
