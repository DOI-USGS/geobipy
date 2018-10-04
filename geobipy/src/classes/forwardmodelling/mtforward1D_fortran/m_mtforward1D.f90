module mtforward1D

implicit none

complex(kind=8), parameter :: cOne = complex(1.d0, 1.d0)
complex(kind=8), parameter :: j = complex(0.d0, 1.d0)
real(kind=8), parameter :: pi = dacos(-1.d0)
real(kind=8), parameter :: pi2 = 2.d0*pi
real(kind=8), parameter :: mu0 = 4.d-7 * pi ! H/m
real(kind=8), parameter :: toDegress = 180.d0 / pi

contains

!====================================================================!
subroutine forward1D(nFreq, nLayers, freqs, parIn, thkIn, apparentRes, phase, c)
    !! Computes the forward magneto-telluric response of a 1D layered earth.
    !! This function is callable from python.
!====================================================================!
    integer, intent(in) :: nFreq
    integer, intent(in) :: nLayers
    real(kind=8), intent(in) :: freqs(nFreq)
    !f2py intent(in) :: freqs
    real(kind=8), intent(in) :: parIn(nLayers)
    !f2py intent(in) ::  parIn
    real(kind=8), intent(in) :: thkIn(nLayers)
    !f2py intent(in) :: thkIn
    real(kind=8), intent(inout) :: apparentRes(nFreq)
    !f2py intent(in, out) :: apparentRes
    real(kind=8), intent(inout) :: phase(nFreq)
    !f2py intent(in, out) :: phase
    complex(kind=8), intent(inout) :: c(nLayers, nFreq)
    !f2py intent(in, out) :: c


    complex(kind=8) :: gamma(nLayers), gammaH, gammaRho
    complex(kind=8) :: z0, z1
    real(kind=8) :: omega, tmp3, rZ0, iZ0
    complex(kind=8) :: tmp, tmp2

    integer :: i
    integer :: k

    do i = 1, nFreq
        omega = pi2 * freqs(i)
        tmp3 = omega * mu0
        tmp = j * tmp3
        gamma = sqrt(tmp / parIn)
        z1 = 1.d0 / gamma(nLayers)
        c(nLayers, i) = z1
        do k = nLayers - 1, 1, -1
            z0 = z1
            z1 = innerKernel(gamma(k), thkIn(k), z0)
            c(k, i) = z1
        enddo
        rZ0 = real(z1)
        iZ0 = aimag(z1)
        apparentRes(i) = tmp3 * (rZ0 * rZ0 + iZ0 * iZ0)
        phase(i) = toDegress * atan(-rZ0 / iZ0)
    enddo

end subroutine
!====================================================================!
!====================================================================!
subroutine sensitivity1D(nFreq, nLayers, freqs, parIn, thkIn, c, sens)
    !! Computes the sensitivity of the MT response of a 1D layered earth w.r.t. resistivity.
    !! This function is callable from python.
!====================================================================!
    integer, intent(in) :: nFreq
    integer, intent(in) :: nLayers
    real(kind=8), intent(in) :: freqs(nFreq)
    !f2py intent(in) :: freqs
    real(kind=8), intent(in) :: parIn(nLayers)
    !f2py intent(in) ::  parIn
    real(kind=8), intent(in) :: thkIn(nLayers)
    !f2py intent(in) :: thkIn
    complex(kind=8), intent(in) :: c(nLayers, nFreq)
    !f2py intent(in) :: c
    real(kind=8), intent(inout) :: sens(2 * nFreq, nLayers)
    !f2py intent(in, out) :: sens

    complex(kind=8) :: dC(nLayers-1)

    integer :: i, nF2
    integer :: k
    complex(kind=8) :: gamma(nLayers)
    complex(kind=8) :: dCdK(nLayers)
    complex(kind=8) :: factor(nLayers)
    complex(kind=8) :: dCjdKj, cjcj1
    real(kind=8) :: rC1, iC1
    real(kind=8) :: rdC1(nLayers), idC1(nLayers)

    real(kind=8) :: tmp
    complex(kind=8) :: tmp1, z0

    do i = 1, nFreq
        nF2 = i + nFreq
        tmp = pi2 * freqs(i) * mu0
        tmp1 = j * tmp
        gamma = sqrt(tmp1 / parIn)
        factor = -0.5d0 * sqrt(tmp1 / (parIn**3.d0))

        z0 = cOne
        do k = 1, nLayers - 1
            call nextTerms(gamma(k), thkIn(k), c(k+1, i), dCjdKj, cjcj1)
            dCdK(k) = z0 * dCjdKj
            z0 = z0 * cjcj1
        enddo

        dCdK(nLayers) = -z0 / gamma(nLayers)**2.d0
        dCdK = factor * dCdK

        rC1 = real(c(1, i))
        iC1 = aimag(c(1, i))
        rdC1 = real(dCdK)
        idC1 = aimag(dCdK)
        sens(i, :) = tmp * ((rC1 * rdC1) + (iC1 * idC1))
        sens(nF2, :) = ((rC1 * idC1) - (iC1 * rdC1)) / (rC1 * rC1 + iC1 * iC1)**2.d0

    enddo

end subroutine
!====================================================================!
!====================================================================!
subroutine nextTerms(k, thk, c, dCjdKj, cjcj1)
!====================================================================!
    complex(kind=8), intent(in) :: k
    real(kind=8), intent(in) :: thk
    complex(kind=8), intent(in) :: c
    complex(kind=8) :: dCjdKj
    complex(kind=8) :: cjcj1

    complex(kind=8) :: tmp0, tmp1, tmp2, tmp3

    tmp0 = 1.d0 / k
    tmp1 = k * c
    tmp2 = k * thk + acoth(tmp1)
    tmp3 = acothPrime(tmp1)

    dCjdKj = (-tmp0**2.d0 * coth(tmp2)) + (tmp0 * cothPrime(tmp2)) * (thk + tmp3 * c)
    cjcj1 = cothPrime(tmp2) * tmp3
end subroutine
!====================================================================!
!====================================================================!
function innerKernel(k, thk, x) result(val)
!====================================================================!
    complex(kind=8), intent(in) :: k
    real(kind=8), intent(in) :: thk
    complex(kind=8), intent(in) :: x
    complex(kind=8) :: val

    complex(kind=8) :: tmp1, tmp2

    val = coth((k * thk) + acoth(k * x)) / k
end function
!====================================================================!
!====================================================================!
function coth(x) result(val)
    !! Computes tanh(x) = (e^x + e^-x) / (e^x - e^-x)
!====================================================================!
    complex(kind=8), intent(in) :: x
    complex(kind=8) :: val
    complex(kind=8) :: ex, emx
    ex = exp(x)
    emx = exp(-x)

    val = (ex + emx) / (ex - emx)
end function
!====================================================================!
!====================================================================!
function cothPrime(x) result(val)
    !! Computes derivate of coth w.r.t. x
!====================================================================!
    complex(kind=8), intent(in) :: x
    complex(kind=8) :: val
    complex(kind=8) :: ex, emx
    ex = exp(x)
    emx = exp(-x)

    val = (2.d0 / (ex - emx))**2.d0
end function
!====================================================================!
!====================================================================!
function acoth(x) result(val)
    !! Computes arccoth
!====================================================================!
    complex(kind=8), intent(in) :: x
    complex(kind=8) :: val
    complex(kind=8) :: ex, emx
    ex = x + 1.d0
    emx = x - 1.d0

    val = 0.5d0 * log((ex) / (emx))
end function
!====================================================================!
!====================================================================!
function acothPrime(x) result(val)
    !! Computes derivate of arccoth w.r.t. x
!====================================================================!
    complex(kind=8), intent(in) :: x
    complex(kind=8) :: val
    
    val = 1.d0 / (1.d0 - x**2.d0)
    
end function
!====================================================================!
end module