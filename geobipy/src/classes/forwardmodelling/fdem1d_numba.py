""" @Fdem1D_Numba
Numba enabled frequency domain forward modelling
Leon Foks
June 2015
"""
from numpy import (real, int64, float64, complex128)
from numpy import (empty, zeros, ones, asarray)
from numpy import (sqrt, exp)
from numpy import pi

pi2 = float64(2.0 * pi)
pi4 = float64(4.0 * pi)
mu0 = float64(4.e-7 * pi)
c = float64(299792458.0)
eps0 = 1.0 / (mu0 * (c**2.0))
nC0 = int64(120)
nC1 = int64(140)

from numba import jit
_numba_settings = {'nopython': True, 'nogil': False, 'fastmath': True, 'cache': True}

@jit(**_numba_settings)
def pyFdem1dfwd(tid, frequencies, tHeight, rHeight, moments, rx, separation, w0, lamda0, lamda02, w1, lamda1, lamda12, scale, conductivity, thickness):

    nFrequencies = int64(len(frequencies))
    nLayers = int64(len(conductivity))

    nL1 = nLayers + 1

    par = zeros(nL1, dtype=float64)
    par[1:] = conductivity

    thk = zeros(nL1, dtype=float64)
    thk[1:] = thickness

    H = empty(nFrequencies, dtype=complex128)
    H0 = empty(nFrequencies, dtype=complex128)

    useJ0 = False
    for i in range(len(tid)):
        if tid[i] in [1, 2, 4, 5, 9]:
            useJ0 = True

    # Compute the angular frequencies from the system frequencies
    omega = empty(nFrequencies, dtype=complex128)
    for i in range(nFrequencies):
        omega[i] = 0.0 + (2.0 * pi * frequencies[i] * mu0)*1j

    rTEj1, u0j1 = calcFdemforward1D(nLayers, nFrequencies, nC1, frequencies, omega, lamda1, lamda12, par, thk)
    if (useJ0):
        rTEj0, u0j0 = calcFdemforward1D(nLayers, nFrequencies, nC0, frequencies, omega, lamda0, lamda02, par, thk)

    for i in range(len(tid)):
        id = tid[i]
        if id == 1:
            H[i], H0[i] =  Hxx(tHeight[i], rHeight[i], moments[i], rx[i], separation[i], rTEj0[i, :], w0, lamda0[i, :], lamda02[i, :], rTEj1[i, :], w1, lamda1[i, :])
        elif id == 3:
            H[i], H0[i] =  Hxz(tHeight[i], rHeight[i], moments[i], rx[i], separation[i], rTEj1[i, :], w1, lamda1[i, :], lamda12[i, :])
        elif id == 7:
            H[i], H0[i] =  Hzx(tHeight[i], rHeight[i], moments[i], rx[i], separation[i], rTEj1[i, :], u0j1[i, :], w1, lamda1[i, :], lamda12[i, :])
        elif id == 9:
            H[i], H0[i] = Hzz(tHeight[i], rHeight[i], moments[i], separation[i], rTEj0[i,  :], u0j0[i,  :], w0, lamda0[i, :])

    return 1.e6 * scale * ((H - H0) / H0)


@jit(**_numba_settings)
def pyFdem1dsen(tid, frequencies, tHeight, rHeight, moments, rx, separation, w0, lamda0, lamda02, w1, lamda1, lamda12, scale, conductivity, thickness):

    nFrequencies = int64(len(frequencies))
    nLayers = int64(len(conductivity))

    nL1 = nLayers + 1

    par = zeros(nL1, dtype=float64)
    par[1:] = conductivity

    thk = zeros(nL1, dtype=float64)
    thk[1:] = thickness

    dH = zeros((nLayers, nFrequencies), dtype=complex128)
    dH0 = zeros((nLayers, nFrequencies), dtype=complex128)

    useJ0 = False
    for i in range(len(tid)):
        if tid[i] in [1, 2, 4, 5, 9]:
            useJ0 = True

    # Compute the angular frequencies from the system frequencies
    omega = empty(nFrequencies, dtype=complex128)
    for i in range(nFrequencies):
        omega[i] = 0.0 + (2.0 * pi * frequencies[i] * mu0)*1j

    u0j1, sens1 = calcFdemSensitivity1D(nLayers, nFrequencies, nC1, frequencies, omega, lamda1, lamda12, par, thk)
    if (useJ0):
        u0j0, sens0 = calcFdemSensitivity1D(nLayers, nFrequencies, nC0, frequencies, omega, lamda0, lamda02, par, thk)

    for k in range(nLayers):
        for i in range(nFrequencies):
            id = tid[i]
            if id == 1:
                dH[k, i], dH0[k, i] = Hxx(tHeight[i], rHeight[i], moments[i], rx[i], separation[i], sens0[k, i, :], w0, lamda0[i, :], lamda02[i, :], sens1[k, i, :], w1, lamda1[i, :])
            elif id == 3:
                dH[k, i], dH0[k, i] = Hxz(tHeight[i], rHeight[i], moments[i], rx[i], separation[i], sens1[k, i, :], w1, lamda1[i, :], lamda12[i, :])
            elif id == 7:
                dH[k, i], dH0[k, i] = Hzx(tHeight[i], rHeight[i], moments[i], rx[i], separation[i], sens1[k, i, :], u0j1[i, :], w1, lamda1[i, :], lamda12[i, :])
            elif id == 9:
                dH[k, i], dH0[k, i] = Hzz(tHeight[i], rHeight[i], moments[i], separation[i], sens0[k, i, :], u0j0[i, :], w0, lamda0[i, :])


    J = empty((nFrequencies, nLayers), dtype=complex128)
    for k in range(nLayers):
        for i in range(nFrequencies):
            J[i, k] = 1.e6 * scale[i] * (dH[k, i] - dH0[k, i]) / dH0[k, i]

    return J

    
@jit(**_numba_settings)
def calcFdemforward1D(nLayers, nFrequencies, nCoefficients, frequencies, omega, lamda, lamda2, par, thk):
    Y, Yn, un = initCoefficients(nLayers, nFrequencies, nCoefficients, frequencies, omega, lamda, lamda2, par)
    return M1_0(nLayers, nFrequencies, nCoefficients, Yn, Y, un, thk)


@jit(**_numba_settings)
def calcFdemSensitivity1D(nLayers, nFrequencies, nCoefficients, frequencies, omega, lamda, lamda2, par, thk):

    Y, Yn, un = initCoefficients(nLayers, nFrequencies, nCoefficients, frequencies, omega, lamda, lamda2, par)

    if (nLayers == 1):
        u0 = zeros((nFrequencies, nCoefficients), dtype=complex128)
        sens = zeros((nLayers, nFrequencies, nCoefficients), dtype=complex128)

        p = par[-1]
        for i in range(nFrequencies):
            for jc in range(nCoefficients):
                sens[-1, i, jc] = p / (2.0 * un[-1, i, jc])

        for i in range(nFrequencies):
            for jc in range(nCoefficients):
                u0[i, jc] = un[0, i, jc]
                a0 = Yn[0, i, jc]
                a1 = Y[1, i, jc]
                a2 = 1.0 / (a0 + a1)
                sens[0, i, jc] = -2.0 * a0 * sens[0, i, jc] * a2**2.0
    else:
        u0, sens = M1_1(nLayers, nFrequencies, nCoefficients, Yn, Y, un, omega, thk, par)

    return u0, sens


@jit(**_numba_settings)
def initCoefficients(nLayers, nFrequencies, nCoefficients, frequencies, omega, lamda, lamda2, par):

    nL1 = nLayers + 1

    un = empty((nL1, nFrequencies, nCoefficients), dtype=complex128)
    Y = empty((nL1, nFrequencies, nCoefficients), dtype=complex128)
    Yn = empty((nL1, nFrequencies, nCoefficients), dtype=complex128)

    # Compute the Admitivity, yn=j*omega*eps+sigma
    for i in range(nFrequencies):
        oTmp = 1.0 / omega[i]
        for jc in range(nCoefficients):
            Y[0, i, jc] = oTmp

    p = par[0]
    for i in range(nFrequencies):
        tTmp = (omega[i] * eps0 + p) * omega[i]
        for jc in range(nCoefficients):
            unTmp = sqrt(tTmp + lamda2[i, jc])
            un[0, i, jc] = unTmp
            Yn[0, i, jc] = unTmp * Y[0, i, jc]

    for k in range(1, nL1):
        p = par[k]
        for i in range(nFrequencies):
            tTmp = (omega[i] * eps0 + p) * omega[i]
            for  jc in range(nCoefficients):
                un[k, i, jc] = sqrt(tTmp + lamda2[i, jc])
                Yn[k, i, jc] = un[k, i, jc] * Y[0, i, jc]

    for i in range(nFrequencies):
        for jc in range(nCoefficients):
            Y[-1, i, jc] = Yn[-1, i, jc]

    return Y, Yn, un


@jit(**_numba_settings)
def M1_0(nLayers, nFrequencies, nCoefficients, Yn, Y, un, thk):

    rTE = empty((nFrequencies, nCoefficients), dtype=complex128)
    u0 = empty((nFrequencies, nCoefficients), dtype=complex128)

    for k in range(nLayers-1, 0, -1):
        k1 = k + 1
        t = thk[k]

        for i in range(nFrequencies):
            for jc in range(nCoefficients):
                a3 = Yn[k, i, jc]
                a4 = Y[k1, i, jc]
                z = un[k, i, jc] * t
                a0 = cTanh1(z)
                a1 = a4 + (a3 * a0)
                a2 = 1.0 / (a3 + (a4 * a0))
                Y[k, i, jc] = a3 * a1 * a2

    for i in range(nFrequencies):
        for jc in range(nCoefficients):
            u0[i, jc] = un[0, i, jc]
            a2 = Yn[0, i, jc]
            a1 = Y[1, i, jc]
            rTE[i, jc] = (a2 - a1) / (a2 + a1)

    return rTE, u0
  

@jit(**_numba_settings)
def M1_1(nLayers, nFrequencies, nCoefficients, Yn, Y, un, omega, thk, par):

    u0 = empty((nFrequencies, nCoefficients), dtype=complex128)
    sens = empty((nLayers, nFrequencies, nCoefficients), dtype=complex128)

    p = par[-1]
    for i in range(nFrequencies):
        for jc in range(nCoefficients):
            sens[-1, i, jc] = p / (2.0 * un[-1, i, jc])

    accumulate = empty((nLayers-1, nFrequencies, nCoefficients), dtype=complex128)

    for k in range(nLayers-1, 0, -1):
        k1 = k + 1
        k2 = k - 1
        p = par[k]
        t = thk[k]
        for i in range(nFrequencies):
            oTmp = omega[i]
            for jc in range(nCoefficients):              
                Yn_ = Yn[k, i, jc]
                Y_ = Y[k1, i, jc]
                un_ = un[k, i, jc]
                z = un_ * t
                b4 = cTanh1(z)
                b5 = b4**2.0
                b8 = (2.0 * Yn_ * Y_ * b5)
                b6 = Y_ + (Yn_ * b4)
                b7 = Yn_ + (Y_ * b4)
                b2 = Y_**2.0

                Y[k, i, jc] = Yn_ * (b6 / b7)

                b6 = Yn_**2.0

                b9 = (p / (2.0 * un_ * (b7**2.0)))

                b7 = b7**(-2.0)
                b8 += ((b2 - b6) * b4) + 2.0 * b6
                b4 = b6 * (1.0 - b5)
                b6 = Yn_**3.0
                accumulate[k2, i, jc] = b4 * b7
                b4 = b2 * Yn_
                b7 = b4 - b6
                b1 = oTmp * t * b7
                b8 += (b1 * b5) - b1

                sens[k2, i, jc] = b8 * b9

    for k in range(1, nLayers-1):
        k1 = k - 1
        for i in range(nFrequencies):
            for jc in range(nCoefficients):         
                accumulate[k, i, jc] = accumulate[k, i, jc] * accumulate[k1, i, jc]
    
    for i in range(nFrequencies):
        for jc in range(nCoefficients):         
            u0[i, jc] = un[0, i, jc]

            a0 = Yn[0, i, jc]
            a1 = Y[1, i, jc]
            a2 = 1.0 / (a0 + a1)

            Y[1, i, jc] = a2**2.0
            Yn[0, i, jc] = -2.0 * a0 * Y[1, i, jc]

    for i in range(nFrequencies):
        for jc in range(nCoefficients): 
            sens[0,  i, jc] *= Yn[0, i, jc]

    for k in range(1, nLayers):
        k2 = k - 1
        for i in range(nFrequencies):
            for jc in range(nCoefficients): 
                sens[k,  i, jc] *= Yn[0, i, jc] * accumulate[k2, i, jc]

    return u0, sens


@jit(**_numba_settings)
def Hxx(tHeight, rHeight, moments, rx, separation, rTEj0, w0, lamda0, lamda02, rTEj1, w1, lamda1):

    hSum = rHeight + tHeight
    hDiff = rHeight - tHeight
    r = 1.0 / separation

    c0 = -(moments / pi4) * r
    d0 = c0 * ((rx * r)**2.0)
    d1 = c0 * (r - ((2.0 * rx**2.0) * (r**3.0)))

    H = complex128(0.0 + 0j)
    H0 = complex128(0.0 + 0j)

    for jc in range(nC0):
        w0_ = d0 * w0[jc]
        J0_ = lamda0[jc]
        rTE0_ = rTEj0[jc]
        a1 = lamda02[jc]
        # Temporary variables
        a0 = exp(-J0_ * (hSum))
    
        k = (a0 - (rTE0_ * exp(J0_ * hDiff))) * a1

        H += k * w0_ # First bessel contribution
        k1 = a0 * a1
        H0 += k1 * w0_ # First bessel contribution to free space

        w1_ = d1 * w1[jc]
        J1_ = lamda1[jc]
        rTE1_ = rTEj1[jc]
        # Use the first 120 entries of the second bessel in this loop.
        b0 = exp(-J1_ * (hSum))
        k2 = (b0 - (rTE1_ * exp(J1_ * hDiff))) * J1_
        H += k2 * w1_
        k3 = b0 * J1_
        H0 += k3 * w1_

    for jc in range(nC0, nC1):
        w1_ = d1 * w1[jc]
        J1_ = lamda1[jc]
        rTE1_ = rTEj1[jc]
        # Use the first 120 entries of the second bessel in this loop.
        b0 = exp(-J1_ * (hSum))
        k2 = (b0 - (rTE1_ * exp(J1_ * hDiff))) * J1_
        H += k2 * w1_
        k3 = b0 * J1_
        H0 += k3 * w1_

    return H,  H0


@jit(**_numba_settings)
def Hxz(tHeight, rHeight, moments, rx, separation, rTE1, w1, lamda1, lamda12):

    hSum = rHeight + tHeight
    hDiff = rHeight - tHeight

    d1 = (rx * moments) / (pi4 * separation)

    H = complex128(0.0 + 0j)
    H0 = complex128(0.0 + 0j)

    for jc in range(nC1):
        w1_ = d1 * w1[jc]
        J1_ = lamda1[jc]
        rTE1_ = rTE1[jc]
        a1 = lamda12[jc]

        b0 = exp(-J1_ * hSum)
        k = (b0 - (rTE1_ * exp(J1_ * hDiff))) * a1
        H += k * w1_
        k = b0 * a1
        H0 += k * w1_
    
    return H, H0
    

@jit(**_numba_settings)
def Hzx(tHeight, rHeight, moments, rx, separation, rTE1, u1, w1, lamda1, lamda12):

    hSum = rHeight + tHeight
    hDiff = rHeight - tHeight

    d1 = (rx * moments) / (pi4 * separation)

    H = complex128(0.0 + 0j)
    H0 = complex128(0.0 + 0j)

    for jc in range(nC1):
        w1_ = d1 * w1[jc]
        u1_ = u1[jc]
        J1_ = lamda1[jc]
        rTE1_ = rTE1[jc]
        a1 = lamda12[jc]

        b0 = exp(-u1_ * hSum)
        k = (b0 - (rTE1_ * exp(u1_ * hDiff))) * a1
        H += k * w1_
        k = b0 * a1
        H0 += k * w1_

    return H, H0

@jit(**_numba_settings)
def Hzz(tHeight, rHeight, moments, separation, rTE, u0, w0, lamda0):

    hSum = rHeight + tHeight
    hDiff = rHeight - tHeight
    a2 = moments / (pi4 * separation)

    H = complex128(0.0 + 0j)
    H0 = complex128(0.0 + 0j)

    for jc in range(nC0):
        w0_ = a2 * w0[jc]
        u0_ = u0[jc]
        J0_ = lamda0[jc]
        rTE_ = rTE[jc]

        # Temporary variables
        a0 = exp(-u0_ * hSum)
        a1 = J0_**3.0 / u0_

        # Equation 4.46K(lam)
        k = (a0 + (rTE_ * exp(u0_ * hDiff))) * a1
        H += k * w0_

        # Free Space response
        k = a0 * a1
        H0 += k * w0_

    return H, H0


@jit(**_numba_settings)
def cTanh1(z):
    if (real(z) > 0.0):
        tmp = exp(-2.0 * z)
        return (1.0 - tmp) / (1.0 + tmp)
    else:
        tmp = exp(2.0 * z)
        return (tmp - 1.0) / (tmp + 1.0)
