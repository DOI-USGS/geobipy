""" @MTfor1D_Class
Module for forward modeling magneto telluric data.
Leon Foks
June 2015
"""
import numpy as np
from .mtforward1d_fortran import mtforward1d

def mt1dfwd(S, mod, z0):
    """ Forward Model a single EM data point from a 1D layered earth conductivity model
    S:    :EmSystem Class describing the aquisition system
    mod: : Model1D Class describing the 1D layered earth model
    z0:  : Altitude of the sensor above the top of the 1D model
    TODO: SPEED UP THIS CODE
    """
    assert z0 <= mod.top, "Sensor altitude must be above the top of the model"

    prd = np.zeros(2 * S.nFreq)
    c = np.zeros([mod.nCells[0], S.nFreq], dtype=np.complex128, order='F')
    mtforward1d.forward1d(S.freq, mod.par, mod.thk, prd[:S.nFreq], prd[S.nFreq:], c, S.nFreq, mod.nCells[0])
    return prd, c


def mt1dsen(S, mod, z0, c):
    """


    """
    assert z0 <= mod.top, "Sensor altitude must be above the top of the model"       

    J = np.zeros([2 * S.nFreq, mod.nCells[0]], order='F')
    mtforward1d.sensitivity1d(S.freq, mod.par, mod.thk, c, J, S.nFreq, mod.nCells[0])
    return J