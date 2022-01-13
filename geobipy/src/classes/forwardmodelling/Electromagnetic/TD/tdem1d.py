""" @EMfor1D_F_Class
Module for forward modeling time domain electro magnetic data.
Leon Foks
June 2020
"""
from copy import deepcopy
import numpy as np
from ....system.TdemSystem_GAAEM import TdemSystem_GAAEM
from .empymod_walktem import empymod_walktem


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
    assert datapoint.z[0] >= model1d.mesh.relativeTo, "Sensor altitude must be above the top of the model"

    heightTolerance = 0.0
    if (datapoint.z > heightTolerance):
        assert isinstance(datapoint.system[0], TdemSystem_GAAEM), TypeError(
            "For airborne data, system must be type TdemSystem_GAAEM")
        return gaTdem1dfwd(datapoint, model1d)

    else:
        return empymod_tdem1dfwd(datapoint, model1d)


def tdem1dsen(datapoint, model1d, ix=None, modelChanged=True):

    heightTolerance = 0.0
    if (datapoint.z > heightTolerance):
        assert isinstance(datapoint.system[0], TdemSystem_GAAEM), TypeError(
            "For airborne data, system must be type TdemSystem_GAAEM")
        return gaTdem1dsen(datapoint, model1d, ix, modelChanged)
    else:
        return empymod_tdem1dsen(datapoint, model1d, ix)


def empymod_tdem1dfwd(datapoint, model1d):

    for i in range(datapoint.nSystems):
        iSys = datapoint._systemIndices(i)
        fm = empymod_walktem(datapoint.system[i], model1d)
        datapoint._predictedData[iSys] = fm


def empymod_tdem1dsen(datapoint, model1d, ix=None):

    if (ix is None):  # Generate a full matrix if the layers are not specified
        ix = range(model1d.mesh.nCells[0])
        J = np.zeros((datapoint.nWindows, model1d.mesh.nCells[0]))
    else:  # Partial matrix for specified layers
        J = np.zeros((datapoint.nWindows, np.size(ix)))

    for j in range(datapoint.nSystems):  # For each system
        iSys = datapoint._systemIndices(j)

        d0 = empymod_walktem(datapoint.system[j], model1d)
        m1 = deepcopy(model1d)

        for i in range(np.size(ix)):  # For the specified layers
            iLayer = ix[i]
            dSigma = 0.02 * model1d.values[iLayer]
            m1.values[:] = model1d.values[:]
            m1.values[iLayer] += dSigma
            d1 = empymod_walktem(datapoint.system[j], m1)
            # Store the necessary component
            J[iSys, i] = (d1 - d0) / dSigma

    datapoint.J = J[datapoint.active, :]
    return datapoint.J


try:
    from gatdaem1d import Earth
    from gatdaem1d import Geometry

    def gaTdem1dfwd(datapoint, model1d):
        # Generate the Brodie Earth class
        E = Earth(model1d.values[:], model1d.mesh.widths[:-1])

        # Generate the Brodie Geometry class
        G = Geometry(datapoint.z[0],
                     datapoint.transmitter.roll.item(), datapoint.transmitter.pitch.item(), datapoint.transmitter.yaw.item(),
                     datapoint.receiver.x.item() - datapoint.transmitter.x.item(),
                     datapoint.receiver.y.item() - datapoint.transmitter.y.item(),
                     datapoint.receiver.z.item() - datapoint.transmitter.z.item(),
                     #  datapoint.loopOffset[0], datapoint.loopOffset[1], datapoint.loopOffset[2],
                     datapoint.receiver.roll.item(), datapoint.receiver.pitch.item(), datapoint.receiver.yaw.item())

        # Forward model the data for each system
        return [datapoint.system[i].forwardmodel(G, E) for i in range(datapoint.nSystems)]

    def ga_fm_dlogc(datapoint, model1d):
        # Generate the Brodie Earth class
        E = Earth(model1d.values[:], model1d.mesh.widths[:-1])

        # Generate the Brodie Geometry class
        G = Geometry(datapoint.z.item(),
                     datapoint.transmitter.roll.item(), datapoint.transmitter.pitch.item(), datapoint.transmitter.yaw.item(),
                     datapoint.receiver.x.item() - datapoint.transmitter.x.item(),
                     datapoint.receiver.y.item() - datapoint.transmitter.y.item(),
                     datapoint.receiver.z.item() - datapoint.transmitter.z.item(),
                     #  datapoint.loopOffset[0], datapoint.loopOffset[1], datapoint.loopOffset[2],
                     datapoint.receiver.roll.item(), datapoint.receiver.pitch.item(), datapoint.receiver.yaw.item())

        # Forward model the data for each system
        return [datapoint.system[i].fm_dlogc(G, E) for i in range(datapoint.nSystems)]

    def gaTdem1dsen(datapoint, model1d, ix=None, modelChanged=True):
        """ Compute the sensitivty matrix for a 1D layered earth model, optionally compute the responses for only the layers in ix """
        # Unfortunately the code requires forward modelled data to compute the
        # sensitivity if the model has changed since last time
        if modelChanged:
            E = Earth(model1d.values[:], model1d.mesh.widths[:-1])

            G = Geometry(datapoint.z.item(),
                         datapoint.transmitter.roll.item(), datapoint.transmitter.pitch.item(), datapoint.transmitter.yaw.item(),
                         datapoint.receiver.x.item() - datapoint.transmitter.x.item(),
                         datapoint.receiver.y.item() - datapoint.transmitter.y.item(),
                         datapoint.receiver.z.item() - datapoint.transmitter.z.item(),
                         datapoint.receiver.roll.item(), datapoint.receiver.pitch.item(), datapoint.receiver.yaw.item())

            for i in range(datapoint.nSystems):
                datapoint.system[i].forwardmodel(G, E)

        if (ix is None):  # Generate a full matrix if the layers are not specified
            ix = range(model1d.mesh.nCells.item())
            J = np.zeros((datapoint.nChannels, model1d.mesh.nCells.item()))
        else:  # Partial matrix for specified layers
            J = np.zeros((datapoint.nChannels, np.size(ix)))

        for j in range(datapoint.nSystems):  # For each system
            iSys = datapoint._systemIndices(j)
            for i in range(np.size(ix)):  # For the specified layers
                tmp = datapoint.system[j].derivative(datapoint.system[j].CONDUCTIVITYDERIVATIVE, ix[i] + 1)

                # Store the necessary component
                comps = []
                if 'x' in datapoint.components:
                    comps.append(tmp.SX)
                if 'y' in datapoint.components:
                    comps.append(tmp.SY)
                if 'z' in datapoint.components:
                    comps.append(-tmp.SZ)
                J[iSys, i] = model1d.values[ix[i]] * np.hstack(comps)

        datapoint.J = J #[datapoint.active, :]
        return datapoint.J

except:
    def gaTdem1dfwd(*args, **kwargs):
        raise Exception("gatdaem1d is not installed. Please see instructions")

    def gaTdem1dsen(*args, **kwargs):
        raise Exception("gatdaem1d is not installed. Please see instructions")
