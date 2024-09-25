""" @EMfor1D_F_Class
Module for forward modeling Frequency domain electro magnetic data.
Leon Foks
June 2015
"""
from numpy import complex128, float64, zeros
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

    assert altitude >= model1d.mesh.relative_to, "Sensor altitude must be above the top of the model"

    transmitter_height = altitude + system.transmitter.z
    receiver_height = -transmitter_height + system.receiver.z
    scl = system.transmitter.moment * system.receiver.moment
    x_separation = system.loop_offsets[0, :]

    kappa = zeros(model1d.values.size, dtype=float64)
    perm = zeros(model1d.values.size, dtype=float64)

    return nbFdem1dfwd(system.tensor_id,
                       system.frequencies,
                       transmitter_height,
                       receiver_height,
                       system.transmitter.moment,
                       x_separation,
                       system.loop_separation,
                       system.w0, system.lamda0, system.lamda02,
                       system.w1, system.lamda1, system.lamda12,
                       scl,
                       model1d.values,
                       kappa,
                       perm,
                       model1d.mesh.widths)


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

#     tHeight = zeros(system.nFrequencies)
#     rHeight = zeros(system.nFrequencies)
#     tMom = zeros(system.nFrequencies)
#     rMom = zeros(system.nFrequencies)
#     rx = zeros(system.nFrequencies)
#     for i in range(system.nFrequencies):
#         tHeight[i] = -z0 + system.T[i].z
#         rHeight[i] = z0 + system.R[i].z
#         tMom[i] = system.T[i].moment
#         rMom[i] = system.R[i].moment
#         rx[i] = system.R[i].x
#     scl = tMom * rMom

#     prd = zeros(system.nFrequencies, dtype=complex128)

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

    assert altitude >= model1d.mesh.relative_to, "Sensor altitude must be above the top of the model"

    transmitter_height = altitude + system.transmitter.z
    receiver_height = -transmitter_height + system.receiver.z
    scl = system.transmitter.moment * system.receiver.moment
    x_separation = system.loop_offsets[0, :]

    kappa = zeros(model1d.values.size, dtype=float64)
    perm = zeros(model1d.values.size, dtype=float64)

    return nbFdem1dsen(system.tensor_id,
                       system.frequencies,
                       transmitter_height,
                       receiver_height,
                       system.transmitter.moment,
                       x_separation,
                       system.loop_separation,
                       system.w0, system.lamda0, system.lamda02,
                       system.w1, system.lamda1, system.lamda12,
                       scl,
                       model1d.values,
                       kappa,
                       perm,
                       model1d.mesh.widths)
