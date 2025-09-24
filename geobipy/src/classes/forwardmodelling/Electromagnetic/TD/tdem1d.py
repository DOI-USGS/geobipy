""" @EMfor1D_F_Class
Module for forward modeling time domain electro magnetic data.
Leon Foks
June 2020
"""
from copy import deepcopy
from numpy import hstack, size, zeros, vstack
from .....base import utilities as cf
from ....system.TdemSystem_GAAEM import TdemSystem_GAAEM
# from .empymod_walktem import empymod_walktem


def tdem1dfwd(datapoint, model1d):
    """Wrapper to freqeuency domain EM forward modellers

    Parameters
    ----------
    system : geobipy.TdemSystem
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
    assert datapoint.z[0] >= model1d.mesh.relative_to, "Sensor altitude must be above the top of the model"

    heightTolerance = 0.0
    if (datapoint.transmitter.z > heightTolerance):
        assert isinstance(datapoint.system[0], TdemSystem_GAAEM), TypeError(
            "For airborne data, system must be type TdemSystem_GAAEM")
        return gaTdem1dfwd(datapoint, model1d)

    else:
        raise NotImplementedError("Cant do ground stuff yet")
        # return empymod_tdem1dfwd(datapoint, model1d)


def tdem1dsen(datapoint, model1d, ix=None, model_changed=True):

    heightTolerance = 0.0
    if (datapoint.transmitter.z > heightTolerance):
        assert isinstance(datapoint.system[0], TdemSystem_GAAEM), TypeError(
            "For airborne data, system must be type TdemSystem_GAAEM")
        return gaTdem1dsen(datapoint, model1d, ix, model_changed)
    # else:
    #     return empymod_tdem1dsen(datapoint, model1d, ix)


# def empymod_tdem1dfwd(datapoint, model1d):

#     for i in range(datapoint.nSystems):
#         iSys = datapoint._systemIndices(i)
#         fm = empymod_walktem(datapoint.system[i], model1d)
#         datapoint._predictedData[iSys] = fm


# def empymod_tdem1dsen(datapoint, model1d, ix=None):

#     if (ix is None):  # Generate a full matrix if the layers are not specified
#         ix = range(model1d.mesh.nCells[0])
#         J = zeros((datapoint.nWindows, model1d.mesh.nCells[0]))
#     else:  # Partial matrix for specified layers
#         J = zeros((datapoint.nWindows, size(ix)))

#     for j in range(datapoint.nSystems):  # For each system
#         iSys = datapoint._systemIndices(j)

#         d0 = empymod_walktem(datapoint.system[j], model1d)
#         m1 = deepcopy(model1d)

#         for i in range(size(ix)):  # For the specified layers
#             iLayer = ix[i]
#             dSigma = 0.02 * model1d.values[iLayer]
#             m1.values[:] = model1d.values[:]
#             m1.values[iLayer] += dSigma
#             d1 = empymod_walktem(datapoint.system[j], m1)
#             # Store the necessary component
#             J[iSys, i] = (d1 - d0) / dSigma

#     datapoint.J = J[datapoint.active, :]
#     return datapoint.J


def gaTdem1dfwd(datapoint, model1d):
    # Generate the Brodie Earth class
    E = model1d.Earth

    G = datapoint.loop_pair.Geometry

    # Forward model the data for each system
    return [datapoint.system[i].forwardmodel(G, E) for i in range(datapoint.nSystems)]

def ga_fm_dlogc(datapoint, model1d):
    # Generate the Brodie Earth class
    E = model1d.Earth

    G = datapoint.loop_pair.Geometry

    # Forward model and sensitivity from each system
    values = [datapoint.system[i].fm_dlogc(G, E) for i in range(datapoint.nSystems)]

    fm = [values[i][0] for i in range(datapoint.nSystems)]

    # J = zeros((datapoint.nChannels, model1d.mesh.nCells.item()))
    comps = []
    for i in range(datapoint.nSystems):  # For each system
        iSys = datapoint._systemIndices(i)
        # Store the necessary component
        if 'x' in datapoint.components:
            comps.append(values[i][1].T)
        if 'y' in datapoint.components:
            comps.append(values[i][2].T)
        if 'z' in datapoint.components:
            comps.append(-values[i][3].T)

    J = vstack(comps)

    return fm, J

def gaTdem1dsen(datapoint, model1d, ix=None, model_changed=True):
    """ Compute the sensitivty matrix for a 1D layered earth model,
    optionally compute the responses for only the layers in ix """
    # Unfortunately the code requires forward modelled data to compute the
    # sensitivity if the model has changed since last time
    if model_changed:
        _ = gaTdem1dfwd(datapoint, model1d)

    if (ix is None):  # Generate a full matrix if the layers are not specified
        ix = range(model1d.mesh.nCells.item())
        J = zeros((datapoint.nChannels, model1d.mesh.nCells.item()))
    else:  # Partial matrix for specified layers
        J = zeros((datapoint.nChannels, size(ix)))

    for j in range(datapoint.nSystems):  # For each system
        iSys = datapoint._systemIndices(j)
        for i in range(size(ix)):  # For the specified layers
            tmp = datapoint.system[j].derivative(datapoint.system[j].CONDUCTIVITYDERIVATIVE, ix[i] + 1)

            # Store the necessary component
            comps = []
            if 'x' in datapoint.components:
                comps.append(tmp.SX)
            if 'y' in datapoint.components:
                comps.append(tmp.SY)
            if 'z' in datapoint.components:
                comps.append(-tmp.SZ)
            J[iSys, i] = model1d.values[ix[i]] * hstack(comps)

    return J

# except Exception as e:
#     def gaTdem1dfwd(*args, **kwargs):
#         raise Exception("{}\n gatdaem1d is not installed. Please see instructions".format(e))

#     def gaTdem1dsen(*args, **kwargs):
#         raise Exception("{}\n gatdaem1d is not installed. Please see instructions".format(e))
