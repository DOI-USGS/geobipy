""" @EMfor1D_F_Class
Module for forward modeling Frequency domain electro magnetic data.
Leon Foks
June 2015
"""
import numpy as np
from .fdem1d_numba import (nbFdem1dfwd, nbFdem1dsen)
# from ...ipforward1d_fortran import ipforward1d

def fdem1dfwd(system, model1d, altitude):
    """Wrapper to freqeuency domain EM forward modellers

    Parameters
    ----------
    system : geobipy.FdemSystem
        Acquisition system information
    model1d : geobipy.Model1D
        1D layered earth geometry
    altitude : float
        Acquisition height above the model

    Returns
    -------
    predictedData : array_like
        Frequency domain data.

    """
    assert altitude >= model1d.top, "Sensor altitude must be above the top of the model"

    # Create the indices of the coil orientations for the frequencies.
    tid = system.getTensorID()

    tHeight = np.empty(system.nFrequencies, dtype=np.float64)
    rHeight = np.empty(system.nFrequencies, dtype=np.float64)
    tMom = np.empty(system.nFrequencies, dtype=np.float64)
    rMom = np.empty(system.nFrequencies, dtype=np.float64)
    rx = np.empty(system.nFrequencies, dtype=np.float64)

    for i in range(system.nFrequencies):
        tHeight[i] = altitude + system.transmitterLoops[i].z
        rHeight[i] = -altitude + system.receiverLoops[i].z
        tMom[i] = system.transmitterLoops[i].moment
        rMom[i] = system.receiverLoops[i].moment
        rx[i] = system.loopOffsets[i, 0]
    scl = tMom * rMom

    frequencies = np.asarray(system.frequencies)
    conductivity = np.asarray(model1d.par)
    kappa = np.asarray(model1d.magnetic_susceptibility)
    perm = np.asarray(model1d.magnetic_permeability)
    thickness = np.asarray(model1d.thk)
    loopSeparation = np.asarray(system.loopSeparation)

    return nbFdem1dfwd(tid, frequencies, tHeight, rHeight, tMom, rx, loopSeparation, system.w0, system.lamda0, system.lamda02, system.w1, system.lamda1, system.lamda12, scl, conductivity, kappa, perm, thickness)


# def ip1dfwd(S, mod, z0):
#     """ Forward Model a single EM data point from a 1D layered earth conductivity model
#     S:    :EmSystem Class describing the aquisition system
#     mod: : Model1D Class describing the 1D layered earth model
#     z0:  : Altitude of the sensor above the top of the 1D model
#     TODO: SPEED UP THIS CODE
#     """
#     assert z0 <= model1d.top, "Sensor altitude must be above the top of the model"

#     # Create the indices of the coil orientations for the frequencies.
#     tid = system.getTensorID()

#     tHeight = np.zeros(system.nFrequencies)
#     rHeight = np.zeros(system.nFrequencies)
#     tMom = np.zeros(system.nFrequencies)
#     rMom = np.zeros(system.nFrequencies)
#     rx = np.zeros(system.nFrequencies)
#     for i in range(system.nFrequencies):
#         tHeight[i] = -z0 + system.T[i].z
#         rHeight[i] = z0 + system.R[i].z
#         tMom[i] = system.T[i].moment
#         rMom[i] = system.R[i].moment
#         rx[i] = system.R[i].x
#     scl = tMom * rMom

#     prd = np.zeros(system.nFrequencies, dtype=np.complex128)

#     ipforward1d.forward1d(tid, system.frequencies, tHeight, rHeight, tMom, rx, system.loopSeparation, scl, model1d.par, model1d.thk, prd, system.nFrequencies,  model1d.nCells[0])

#     return prd


def fdem1dsen(system, model1d, altitude):
    """Wrapper to freqeuency domain EM forward modellers

    Parameters
    ----------
    system : geobipy.FdemSystem
        Acquisition system information
    model1d : geobipy.Model1D
        1D layered earth geometry
    altitude : float
        Acquisition height above the model

    Returns
    -------
    predictedData : array_like
        Frequency domain data.

    """

    assert altitude >= model1d.top, "Sensor altitude must be above the top of the model"

    # Create the indices of the coil orientations for the frequencies.
    tid = system.getTensorID()

    tHeight = np.zeros(system.nFrequencies)
    rHeight = np.zeros(system.nFrequencies)
    tMom = np.zeros(system.nFrequencies)
    rMom = np.zeros(system.nFrequencies)
    rx = np.zeros(system.nFrequencies)
    for i in range(system.nFrequencies):
        tHeight[i] = altitude + system.transmitterLoops[i].z
        rHeight[i] = -altitude + system.transmitterLoops[i].z
        tMom[i] = system.transmitterLoops[i].moment
        rMom[i] = system.receiverLoops[i].moment
        rx[i] = system.loopOffsets[i, 0]
    scl = tMom * rMom

    frequencies = np.asarray(system.frequencies)
    conductivity = np.asarray(model1d.par)
    kappa = np.asarray(model1d.magnetic_susceptibility)
    perm = np.asarray(model1d.magnetic_permeability)
    thickness = np.asarray(model1d.thk)
    loopSeparation = np.asarray(system.loopSeparation)

    return nbFdem1dsen(tid, frequencies, tHeight, rHeight, tMom, rx, loopSeparation, system.w0, system.lamda0, system.lamda02, system.w1, system.lamda1, system.lamda12, scl, conductivity, kappa, perm, thickness)

