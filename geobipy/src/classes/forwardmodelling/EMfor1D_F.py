""" @EMfor1D_F_Class
Module for forward modeling Frequency domain electro magnetic data.
Leon Foks
June 2015
"""
#%%
from ...classes.core.StatArray import StatArray
from ..system.FdemSystem import FdemSystem
from ..model.Model1D import Model1D
from scipy.constants import mu_0 as mu0
from scipy.constants import epsilon_0 as eps0
from ...base.logging import myLogger
import numpy as np
from numpy import exp as nExp  # For arrays
from cmath import exp as cExp  # For single numbers
#from ...base import Error as Err
import time as time
from ...base.customFunctions import tanh
from .fdemforward1d_fortran import fdemforward1d

#%%

def fdem1dfwd(S, mod, z0):
    """ Forward Model a single EM data point from a 1D layered earth conductivity model
    S:    :EmSystem Class describing the aquisition system
    mod: : Model1D Class describing the 1D layered earth model
    z0:  : Altitude of the sensor above the top of the 1D model
    TODO: SPEED UP THIS CODE
    """
    # ####lg.myLogger("Global");####lg.indent()
    ####lg.info('Inside EMfor1D')
    assert z0 <= mod.top, "Sensor altitude must be above the top of the model"

    # Create the indices of the coil orientations for the frequencies.
    xx, xy, xz, yx, yy, yz, zx, zy, zz = S.getComponentID()

    useJ0 = (np.max([len(xx), len(xy), len(yx), len(yy), len(zz)]) > 0)
    useJ1 = useJ0 or (np.max([len(xz), len(yz), len(zx), len(zy)]) > 0)

#%%
    tMom = np.zeros(S.nFreq)
    rMom = np.zeros(S.nFreq)
    for i in range(S.nFreq):
        tMom[i] = S.T[i].moment
        rMom[i] = S.R[i].moment
    scl = tMom * rMom
    ####lg.critical("Scale: "+str(scl))
#%%
#==============================================================================
#     print("Coil Orientation Indices to Frequencies")
#     print(xx,xy,xz)
#     print(yx,yy,yz)
#     print(zx,zy,zz)
#==============================================================================

    # calculate reflection coefficient
    if useJ0:
#        rTEj0, u0j0 = calcrTEsens(S, S.H.J0, mod, 0)

        rTEj0 = np.zeros([S.nFreq,S.H.J0.nC], dtype=np.complex128, order='F')
        u0j0 = np.zeros([S.nFreq,S.H.J0.nC], dtype=np.complex128, order='F')
        fdemforward1d.calcfdemforward1d(S.freq, S.H.J0.lam, mod.par, mod.thk, mod.chim, rTEj0, u0j0, S.nFreq, S.H.J0.nC, mod.nCells[0])
    # *** calcrTEsens.py only returns two variables for f####lg.== 0 ***
    if useJ1:
#        rTEj1, u0j1 = calcrTEsens(S, S.H.J1, mod, 0)
        rTEj1 = np.zeros([S.nFreq,S.H.J1.nC], dtype=np.complex128, order='F')
        u0j1 = np.zeros([S.nFreq,S.H.J1.nC], dtype=np.complex128, order='F')
        fdemforward1d.calcfdemforward1d(S.freq, S.H.J1.lam, mod.par, mod.thk, mod.chim, rTEj1, u0j1, S.nFreq, S.H.J1.nC, mod.nCells[0])
    # *** calcrTEsens.py only returns two variables for f####lg.== 0 ***
    H = np.zeros(S.nFreq, dtype=np.complex128)
    H0 = np.zeros(S.nFreq, dtype=np.complex128)
    # Horizontal Co-Planar Coils
    if (len(zz) != 0):
        H[zz], H0[zz] = calcHzz(zz, S, z0, rTEj0, u0j0, S.H.J0)
    # Vertical Co-Axial Coils
    if (len(xx) != 0):
        H[xx], H0[xx] = calcHxx(xx, S, z0, rTEj0, rTEj1, S.H.J0, S.H.J1)
    prd = 1.e6 * ((H - H0) / H0) * scl
    ####lg.debug("Predicted Data: "+str(prd))
    # ####lg.dedent()
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
    xx, xy, xz, yx, yy, yz, zx, zy, zz = S.getComponentID()

    useJ0 = (np.max([len(xx), len(xy), len(yx), len(yy), len(zz)]) > 0)
    useJ1 = useJ0 or (np.max([len(xz), len(yz), len(zx), len(zy)]) > 0)

#%%
    tMom = np.zeros(S.nFreq)
    rMom = np.zeros(S.nFreq)
    for i in range(S.nFreq):
        tMom[i] = S.T[i].moment
        rMom[i] = S.R[i].moment
    scl = tMom * rMom
    ####lg.critical("Scale: "+str(scl))
#%%
#==============================================================================
#     print("Coil Orientation Indices to Frequencies")
#     print(xx,xy,xz)
#     print(yx,yy,yz)
#     print(zx,zy,zz)
#==============================================================================
    if useJ0:
        # *** calcrTEsens.py only returns two variables for f####lg.== 1 ***
#        rTE0, u0j0, dTEdp0 = calcrTEsens(S, S.H.J0, mod, 1)
#        if (nLayers == 1):
#            dTEdp0 = np.repeat(dTEdp0[:, :, np.newaxis], 1, 2)

        rTE0 = np.zeros([S.nFreq, S.H.J0.nC], dtype=np.complex128, order='F')
        u0j0 = np.zeros([S.nFreq, S.H.J0.nC], dtype=np.complex128, order='F')
        dTEdp0 = np.zeros([S.nFreq, S.H.J0.nC, nLayers], dtype=np.complex128, order='F')
        fdemforward1d.calcfdemsensitivity1d(S.freq, S.H.J0.lam, mod.par, mod.thk, mod.chim, rTE0, u0j0, dTEdp0, S.nFreq, S.H.J0.nC, nLayers)

    if useJ1:
        # *** calcrTEsens.py only returns two variables for f####lg.== 1 ***
#        rTE1, u0j1, dTEdp1 = calcrTEsens(S, S.H.J1, mod, 1)
#        if (nLayers == 1):
#            dTEdp1 = np.repeat(dTEdp1[:, :, np.newaxis], 1, 2)

        rTE1 = np.zeros([S.nFreq, S.H.J1.nC], dtype=np.complex128, order='F')
        u0j1 = np.zeros([S.nFreq, S.H.J1.nC], dtype=np.complex128, order='F')
        dTEdp1 = np.zeros([S.nFreq, S.H.J1.nC, nLayers], dtype=np.complex128, order='F')
        fdemforward1d.calcfdemsensitivity1d(S.freq, S.H.J1.lam, mod.par, mod.thk, mod.chim, rTE1, u0j1, dTEdp1, S.nFreq, S.H.J1.nC, nLayers)

    dH = np.zeros([S.nFreq, nLayers], dtype=np.complex128)
    dH0 = np.zeros([S.nFreq, nLayers], dtype=np.complex128)

    for i in range(nLayers):
            # Horizontal Co-Planar Coils
        if (len(zz) != 0):
            dH[zz, i], dH0[zz, i] = calcHzz(zz, S, z0, dTEdp0[:, :, i], u0j0, S.H.J0)
        # Vertical Co-Axial Coils
        if (len(xx) != 0):
            dH[xx, i], dH0[xx, i] = calcHxx(xx, S, z0, dTEdp0[:, :, i], dTEdp1[:, :, i], S.H.J0, S.H.J1)
    scl = tMom * rMom
    J = 1.e6 * ((dH - dH0)) / dH0
    J *= np.repeat(scl[:, np.newaxis], nLayers, 1)
     ####lg.dedent()
    return J


def calcrTEsens(S, J, M, computeSens):
    """ Calculate the reflection coefficient according to Ward and Hohmann, EM theory for geophysical applications
    B. Minsley, June 2010
    S:  : EmSystem Class
    J:  : Bessel Function Filter Parameters
    M:  : 1D Layered Model
    computeSens: : True or False
    """
    # ####lg.myLogger("Global"); ####lg.indent()
    ####lg.critical("Inside calcrTEsens")
    # Get the number of layers in the model
    nLayers = M.nCells[0]
    nL1 = nLayers + 1

    # Create temporary variables and insert parameters at the beginning
    con = np.zeros(nLayers + 1)
    con[0] = 0.0
    con[1:] = M.par[:]
    chim = np.zeros(nLayers + 1)
    chim[0] = 0.0
    chim[1:] = M.chim[:]
    thk = np.zeros(nLayers + 1)
    thk[0] = np.NAN
    thk[1:] = M.thk[:]

    ####lg.critical("Conductivity: "+str(con))
    ####lg.critical("ChiM: "+str(chim))
    ####lg.critical("Thickness: "+str(thk))

    ####lg.critical("nLayers: "+str(nLayers))

    lamSq = J.lam**2.0

    ####lg.critical("Lambda Squared: "+str(lamSq))

    # Compute the angular frequencies from the system frequencies
    omega = np.zeros(S.nFreq)
    omega += 2.0 * np.pi * S.freq[:]

    ####lg.critical("Omega: "+str(omega))

    # Build an array of responses per layer
    tmp = np.dot(1j, omega)
    # Compute the Admitivity, yn=j*omega*eps+sigma
    yn = np.repeat(tmp[:, np.newaxis] * eps0, nL1, 1) + con
    zn = np.repeat(tmp[:, np.newaxis] * mu0, nL1, 1)
    znP = np.repeat(zn[:, np.newaxis, :], J.nC, axis=1)
    zy = np.multiply(zn, yn)

    un = np.sqrt(np.repeat(zy[:, np.newaxis, :], J.nC, axis=1) +
                 lamSq[:, :, np.newaxis], dtype=np.complex128)
    Yn = un / znP

    Y = np.zeros([S.nFreq, J.nC, nL1], dtype=np.complex128)
    Y[:, :, nLayers] = Yn[:, :, nLayers]
    Y[:, :, 0] = np.NAN


    if computeSens == 0:
        if nLayers == 1:
            # ####lg.dedent()
            return M0_0(Yn, Y, un)
        # ####lg.dedent()
        tmp = M1_0(Yn, Y, un, thk)

        ####lg.critical("M1_0: "+str(tmp))
#      if (np.any(np.isnan(tmp))): Err.Emsg("Stopped Here")
        return tmp
    else:
        dydp = np.zeros([S.nFreq, J.nC, nLayers], dtype=np.complex128)
        dydp[:, :, nLayers - 1] = con[-1] / (2.0 * un[:, :, nLayers])
        if nLayers == 1:
            # ####lg.dedent()
            return M0_1(Yn, Y, un, dydp)
        # ####lg.dedent()
        return M1_1(Yn, Y, un, tmp, thk, con, dydp)


def M0_0(Yn, Y, un):
    # ####lg.myLogger("Global"); ####lg.indent()
    # ####lg.critical("M0_0")
    u0 = un[:, :, 0]
    rTE = (Yn[:, :, 0] - Y[:, :, 1]) / (Yn[:, :, 0] + Y[:, :, 1])
    ####lg.critical("rTE: "+str(rTE))
    # ####lg.dedent()
    return (rTE, u0)


def M1_0(Yn, Y, un, thk):
    ####lg=myLogger("Global"); ####lg.indent()
    ####lg.critical("M1_0")
    nL = thk.size - 1

    ####lg.critical("nL: "+str(nL))
    revRange = reversed(range(1, nL))
    for i in revRange:
        ####lg.critical("i: "+str(i))
        i1 = i + 1
        tanuh = tanh(un[:, :, i] * thk[i])
        ####lg.critical("tanuh: "+str(tanuh[0,:]))
        tmpY = Y[:, :, i1]
        tmpYn = Yn[:, :, i]
        num = tmpY + (tmpYn * tanuh)
        den = tmpYn + (tmpY * tanuh)
        Y[:, :, i] = tmpYn * (num / den)
    ####lg.critical("M1_0 Y: "+str(Y))
    ####lg.dedent()
    return M0_0(Yn, Y, un)


def M0_1(Yn, Y, un, dydp):
    rTE, u0 = M0_0(Yn, Y, un)
    sens = -2.0 * Yn[:, :, 0] * dydp[:, :, 0] / (Yn[:, :, 0] + Y[:, :, 1])**2.0
    return (rTE, u0, sens)


def M1_1(Yn, Y, un, tmp, thk, con, dydp):
    dum = np.repeat(tmp[:, np.newaxis], np.size(Yn, 1), 1) * mu0
    dydy = np.ones(dydp.shape, dtype=np.complex128)
    nL = thk.size - 1

    revRange = reversed(range(1, nL))

    for i in revRange:
        i1 = i + 1
        tmpUn = un[:, :, i]
        tanuh = tanh(tmpUn * thk[i])
        tmpY = Y[:, :, i1]
        tmpYn = Yn[:, :, i]
        num = tmpY + (tmpYn * tanuh)
        den = tmpYn + (tmpY * tanuh)
        Y[:, :, i] = tmpYn * (num / den)
        # Temporary variables
        dum1 = dum * thk[i]
        a0 = tmpY**2.0
        b0 = tmpYn**2.0
        c0 = tanuh**2.0
        a1 = tmpYn * a0
        a2 = tmpYn**3.0
        a3 = a1 - a2
        t1 = (con[i] / (2.0 * tmpUn * (den**2.0)))
        t6 = dum1 * a3
        t2 = (2.0 * tmpYn * tmpY * c0)
        t3 = t6 * c0
        t4 = ((a0 - b0) * tanuh) + 2.0 * b0
        t5 = -t6
        # dY(i)/d(ln(con(i)))
        dydp[:, :, i - 1] = t1 * (t2 + t3 + t4 + t5)
        # dY(i)/dY(i+1)
        num1 = b0 * (1.0 - c0)
        den1 = den**-2.0
        dydy[:, :, i] = num1 * den1
    u0 = un[:, :, 0]
    dum1 = Yn[:, :, 0]
    dum = (dum1 + Y[:, :, 1])**-1.0
    rTE = (dum1 - Y[:, :, 1]) * dum
    num = -2.0 * np.repeat(dum1[:, :, np.newaxis], nL, 2)
    dum1 = dum**2.0
    den = np.repeat(dum1[:, :, np.newaxis], nL, 2)

    sens = num * dydp * np.cumprod(dydy, 2) * den
    return (rTE, u0, sens)


def calcHzz(i, S, z, rTEin, u0in, J):
    """ Compute the ZZ component of the EM Field?
    ix:    : Loop Pair to compute this response for
    S:     : EmSystem Class
    z:     : Altitude of the Sensor
    rTEiN: : ??
    u0iN:  : ??
    J:     : Bessel Function Filter Parameters
    """
    # Parse out the necessary components
    nComps = len(i)
    height = np.zeros(nComps)
    rHeight = np.zeros(nComps)
    tMom = np.zeros(nComps)
    for j in range(nComps):
        height[j] = -z + S.T[i[j]].off
        rHeight[j] = z + S.R[i[j]].off
        tMom[j] = S.T[i[j]].moment
    H = np.repeat(height[:, np.newaxis], J.nC, 1)
    Z = np.repeat(rHeight[:, np.newaxis], J.nC, 1)
    dist = S.dist[i]
    rTE = rTEin[i, :]
    u0 = u0in[i, :]
    lam = J.lam[i, :]
    # Temporary variables
    a0 = nExp(-u0 * (Z + H))
    a1 = lam**3.0 / u0
    a2 = (tMom / (4.0 * np.pi)) / dist
    # Equation 4.46K(lam)
#  tmp=np.complex256(u0*(Z-H))
    tmp = u0 * (Z - H)
    K = (a0 + rTE * nExp(tmp)) * a1
    Hzz = a2 * (np.dot(K, J.W[:]))
    # Free-Space response
    K0 = a0 * a1
    Hzz0 = a2 * (np.dot(K0, J.W[:]))
    return Hzz, Hzz0


def calcHxx(i, S, z, rTEin0, rTEin1, J0, J1):
    # Parse out the necessary components
    nComps = len(i)
    height = np.zeros(nComps)
    rHeight = np.zeros(nComps)
    tMom = np.zeros(nComps)
    rx = np.zeros(nComps)
    j = np.arange(nComps)
    for j in range(nComps):
        height[j] = -z + S.T[i[j]].off
        rHeight[j] = z + S.R[i[j]].off
        tMom[j] = S.T[i[j]].moment
        rx[j] = S.R[i[j]].tx
    dist = S.dist[i]
    H0 = np.repeat(height[:, np.newaxis], J0.nC, 1)
    Z0 = np.repeat(rHeight[:, np.newaxis], J0.nC, 1)
    lam0 = J0.lam[i, :]
    rTE0 = rTEin0[i, :]
    H1 = np.repeat(height[:, np.newaxis], J1.nC, 1)
    Z1 = np.repeat(rHeight[:, np.newaxis], J1.nC, 1)
    lam1 = J1.lam[i, :]
    rTE1 = rTEin1[i, :]
    # Temporary variables
    a0 = nExp(-lam0 * (Z0 + H0))
    a1 = lam0**2.0
    b0 = nExp(-lam1 * (Z1 + H1))
    tmp = 1.0 / dist
    c0 = -(tMom / (4.0 * np.pi)) * tmp
    d0 = c0 * ((rx * tmp)**2.0)
    d1 = c0 * ((tmp) - ((2.0 * rx**2.0) / (dist**3.0)))
    # K(lam) equaion 4.46
#  tmp=np.complex256(lam0*(Z0-H0))
    tmp = lam0 * (Z0 - H0)
    Ka = (a0 - rTE0 * nExp(tmp)) * a1
#  tmp=np.complex256(lam1*(Z1-H1))
    tmp = lam1 * (Z1 - H1)
    Kb = (b0 - rTE1 * nExp(tmp)) * lam1
    HxxA = d0 * (np.dot(Ka, J0.W[:]))
    HxxB = d1 * (np.dot(Kb, J1.W[:]))

    Hxx = HxxA + HxxB

    # Temporary variables

    # Free-Space
    Ka = a0 * a1
    Kb = b0 * lam1

    HxxA = d0 * (np.dot(Ka, J0.W[:]))
    HxxB = d1 * (np.dot(Kb, J1.W[:]))

    Hxx0 = HxxA + HxxB
    return Hxx, Hxx0

