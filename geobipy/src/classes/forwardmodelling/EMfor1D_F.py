""" @EMfor1D_F_Class
Module for forward modeling Frequency domain electro magnetic data.
Leon Foks
June 2015
"""
import numpy as np
from .fdemforward1d_fortran import fdemforward1d

def fdem1dfwd(S, mod, z0):
    """ Forward Model a single EM data point from a 1D layered earth conductivity model
    S:    :EmSystem Class describing the aquisition system
    mod: : Model1D Class describing the 1D layered earth model
    z0:  : Altitude of the sensor above the top of the 1D model
    TODO: SPEED UP THIS CODE
    """
    assert z0 <= mod.top, "Sensor altitude must be above the top of the model"

    # Create the indices of the coil orientations for the frequencies.
    tid = S.getTensorID()
    
    tHeight = np.zeros(S.nFrequencies)
    rHeight = np.zeros(S.nFrequencies)
    tMom = np.zeros(S.nFrequencies)
    rMom = np.zeros(S.nFrequencies)
    rx = np.zeros(S.nFrequencies)
    for i in range(S.nFrequencies):
        tHeight[i] = -z0 + S.T[i].z
        rHeight[i] = z0 + S.R[i].z
        tMom[i] = S.T[i].moment
        rMom[i] = S.R[i].moment
        rx[i] = S.R[i].x
    scl = tMom * rMom
       
    prd = np.zeros(S.nFrequencies, dtype=np.complex128)

    fdemforward1d.forward1d(tid, S.frequencies, tHeight, rHeight, tMom, rx, S.dist, scl, mod.par, mod.thk, prd, S.nFrequencies,  mod.nCells[0])

    return prd


def fdem1dsen(S, mod, z0):
    """ Forward Model a single EM data point from a 1D layered earth conductivity model
    S:    :EmSystem Class describing the aquisition system
    mod: : Model1D Class describing the 1D layered earth model
    z0:  : Altitude of the sensor above the top of the 1D model
    TODO: SPEED UP THIS CODE
    """
    assert z0 <= mod.top, "Sensor altitude must be above the top of the model"

    nLayers = mod.nCells[0]
    # Create the indices of the coil orientations for the frequencies.
    tid = S.getTensorID()

    tHeight = np.zeros(S.nFrequencies)
    rHeight = np.zeros(S.nFrequencies)
    tMom = np.zeros(S.nFrequencies)
    rMom = np.zeros(S.nFrequencies)
    rx = np.zeros(S.nFrequencies)
    for i in range(S.nFrequencies):
        tHeight[i] = -z0 + S.T[i].z
        rHeight[i] = z0 + S.R[i].z
        tMom[i] = S.T[i].moment
        rMom[i] = S.R[i].moment
        rx[i] = S.R[i].x
    scl = tMom * rMom

    J = np.zeros([S.nFrequencies, nLayers], dtype=np.complex128, order='F')

    fdemforward1d.sensitivity1d(tid, S.frequencies, tHeight, rHeight, tMom, rx, S.dist, scl, mod.par, mod.thk, J, S.nFrequencies,  mod.nCells[0])

    return J