module fdemforward1D
implicit none

! Define some fixed parameters
real(kind=8), parameter :: NaN = transfer((/ Z'00000000', Z'7FF80000' /), 1.0_8)
complex(kind=8), parameter :: cNaN = complex(NaN, NaN)
real(kind=8), parameter :: pi = dacos(-1.d0)
real(kind=8), parameter :: pi2 = 2.d0 * pi
real(kind=8), parameter :: pi4 = 4.d0 * pi
real(kind=8), parameter :: mu0 = 4.d-7 * pi ! H/m
real(kind=8), parameter :: c = 299792458.d0 ! m/s
real(kind=8), parameter :: eps0 = 1.d0 / (mu0 * (c**2.d0)) ! F/m
integer, parameter :: nCJ0 = 120
integer, parameter :: nCJ1 = 140

! Using a class allows us to create coefficients once when the forward modeller is first used.
type t_lambdas
    real(kind=8), allocatable :: w0(:), w1(:)
    real(kind=8), allocatable :: lam0(:, :), lam1(:, :)
end type

! Private so f2py doesnt barf...
type(t_lambdas), private :: lam

contains

!====================================================================!
subroutine forward1D(nFreq, nLayers, tid, freqs, tHeight, rHeight, moments, rx, separation, scale, parIn, thkIn, predicted)
    !! Computes the forward frequency domain electromagnetic response of a 1D layered earth.
    !! This function is callable from python.
!====================================================================!
    integer, intent(in) :: nFreq
    integer, intent(in) :: nLayers
    integer, intent(in) :: tid(nFreq)
    !f2py intent(in) :: tid
    real(kind=8), intent(in) :: freqs(nFreq)
    !f2py intent(in) :: freqs
    real(kind=8), intent(in) :: tHeight(nFreq)
    !f2py intent(in) :: tHeight
    real(kind=8), intent(in) :: rHeight(nFreq)
    !f2py intent(in) :: rHeight
    real(kind=8), intent(in) :: moments(nFreq)
    !f2py intent(in) :: moments
    real(kind=8), intent(in) :: rx(nFreq)
    !f2py intent(in) :: rx
    real(kind=8), intent(in) :: separation(nFreq)
    !f2py intent(in) :: separation
    real(kind=8), intent(in) :: scale(nFreq)
    !f2py intent(in) :: scale
    real(kind=8), intent(in) :: parIn(nLayers)
    !f2py intent(in) ::  parIn
    real(kind=8), intent(in) :: thkIn(nLayers)
    !f2py intent(in) :: thkIn
    complex(kind=8), intent(inout) :: predicted(nFreq)
    !f2py intent(in, out) :: predicted

    integer :: i
    logical :: useJ0
    complex(kind=8) :: H(nFreq), H0(nFreq)
    complex(kind=8) :: rTEj0(nCJ0, nFreq), rTEj1(nCJ1, nFreq)
    complex(kind=8) :: u0j0(nCJ0, nFreq), u0j1(nCJ1, nFreq)
    real(kind=8) :: par(nLayers+1), thk(nLayers+1)

    call setLambdas(lam, nFreq, separation)

    H = complex(0.d0, 0.d0)
    H0 = complex(0.d0, 0.d0)
    rTEj0 = complex(0.d0, 0.d0)
    u0j0 = complex(0.d0, 0.d0)
    rTEj1 = complex(0.d0, 0.d0)
    u0j1 = complex(0.d0, 0.d0)

    useJ0 = .false.
    i = 1
    do while (.not. useJ0 .and. i <= nFreq)
        select case(tid(i))
        case (1,2,4,5,9)
            useJ0 = .true.
            exit
        end select
        i = i + 1
    enddo

    call initModel(parIn, thkIn, nLayers, par, thk)

    call calcFdemforward1D(nFreq, nCJ1, nLayers, freqs, lam%lam1, par, thk, rTEj1, u0j1)
    if (useJ0) call calcFdemforward1D(nFreq, nCJ0, nLayers, freqs, lam%lam0, par, thk, rTEj0, u0j0)

    do i = 1, nFreq
        select case (tid(i))
        case(1)
            call calcHxx(nFreq, nCJ0, nCJ1, i, tHeight, rHeight, moments, rx, separation, rTEj0, lam%w0, lam%lam0, rTEj1, lam%w1, lam%lam1, H, H0)
        case(3)
            call calcHxz(nFreq, nCJ1, i, tHeight, rHeight, moments, rx, separation, rTEj1, lam%w1, lam%lam1, H, H0)
        case(7)
            call calcHzx(nFreq, nCJ1, i, tHeight, rHeight, moments, rx, separation, rTEj1, u0j1, lam%w1, lam%lam1, H, H0)
        case(9)
            call calcHzz(nFreq, nCJ0, i, tHeight, rHeight, moments, separation, rTEj0, u0j0, lam%w0, lam%lam0, H, H0)
        end select
    enddo

    predicted = 1.d6 * scale * ((H - H0) / H0)

end subroutine
!====================================================================!
!====================================================================!
subroutine sensitivity1D(nFreq, nLayers, tid, freqs, tHeight, rHeight, moments, rx, separation, scale, parIn, thkIn, sens)
    !! Computes the sensitivity of the frequency domain electromagnetic response of a 1D layered earth.
    !! This function is callable from python.
!====================================================================!
    integer, intent(in) :: nFreq
    integer, intent(in) :: nLayers
    integer, intent(in) :: tid(nFreq)
    !f2py intent(in) :: tid
    real(kind=8), intent(in) :: freqs(nFreq)
    !f2py intent(in) :: freqs
    real(kind=8), intent(in) :: tHeight(nFreq)
    !f2py intent(in) :: tHeight
    real(kind=8), intent(in) :: rHeight(nFreq)
    !f2py intent(in) :: rHeight
    real(kind=8), intent(in) :: moments(nFreq)
    !f2py intent(in) :: moments
    real(kind=8), intent(in) :: rx(nFreq)
    !f2py intent(in) :: rx
    real(kind=8), intent(in) :: separation(nFreq)
    !f2py intent(in) :: separation
    real(kind=8), intent(in) :: scale(nFreq)
    !f2py intent(in) :: scale
    real(kind=8), intent(in) :: parIn(nLayers)
    !f2py intent(in) ::  parIn
    real(kind=8), intent(in) :: thkIn(nLayers)
    !f2py intent(in) :: thkIn
    complex(kind=8), intent(inout) :: sens(nFreq, nLayers)
    !f2py intent(in, out) :: sens

    integer :: i, j
    logical :: useJ0
    complex(kind=8) :: dH(nFreq, nLayers), dH0(nFreq, nLayers)
    complex(kind=8) :: rTEj0(nCJ0, nFreq), rTEj1(nCJ1, nFreq)
    complex(kind=8) :: u0j0(nCJ0, nFreq), u0j1(nCJ1, nFreq)
    complex(kind=8) :: sens0(nCJ0, nFreq, nLayers), sens1(nCJ1, nFreq, nLayers)
    real(kind=8) :: par(nLayers+1), thk(nLayers+1)

    call setLambdas(lam, nFreq, separation)

    dH = complex(0.d0, 0.d0)
    dH0 = complex(0.d0, 0.d0)
    rTEj0 = complex(0.d0, 0.d0)
    u0j0 = complex(0.d0, 0.d0)
    rTEj1 = complex(0.d0, 0.d0)
    u0j1 = complex(0.d0, 0.d0)
    sens0 = complex(0.d0, 0.d0)
    sens1 = complex(0.d0, 0.d0)

    useJ0 = .false.
    i = 1
    do while (.not. useJ0 .and. i <= nFreq)
        select case(tid(i))
        case (1,2,4,5,9)
            useJ0 = .true.
            exit
        end select
        i = i + 1
    enddo

    call initModel(parIn, thkIn, nLayers, par, thk)

    call calcFdemSensitivity1D(nFreq, nCJ1, nLayers, freqs, lam%lam1, par, thk, rTEj1, u0j1, sens1)
    if (useJ0) call calcFdemSensitivity1D(nFreq, nCJ0, nLayers, freqs, lam%lam0, par, thk, rTEj0, u0j0, sens0)

    do i = 1, nFreq
        select case (tid(i))
        case(1)
            do j = 1, nLayers
                call calcHxx(nFreq, nCJ0, nCJ1, i, tHeight, rHeight, moments, rx, separation, sens0(:, :, j), lam%w0, lam%lam0, sens1(:, :, j), lam%w1, lam%lam1, dH(:, j), dH0(:, j))
            enddo
        case(3)
            do j = 1, nLayers
                call calcHxz(nFreq, nCJ1, i, tHeight, rHeight, moments, rx, separation, sens1(:, :, j), lam%w1, lam%lam1, dH(:, j), dH0(:, j))
            enddo
        case(7)
            do j = 1, nLayers
                call calcHzx(nFreq, nCJ1, i, tHeight, rHeight, moments, rx, separation, sens1(:, :, j), u0j1, lam%w1, lam%lam1, dH(:, j), dH0(:, j))
            enddo
        case(9)
            do j = 1, nLayers
                call calcHzz(nFreq, nCJ0, i, tHeight, rHeight, moments, separation, sens0(:, :, j), u0j0, lam%w0, lam%lam0, dH(:, j), dH0(:, j))
            enddo
        end select
        sens(i, :) = 1.d6 * scale(i) * (dH(i, :) - dH0(i, :)) / dH0(i, :)
    enddo

end subroutine
!====================================================================!

!====================================================================!
subroutine calcFdemforward1D(nFreq, nC, nLayers, freqs, fCoeffs, par, thk, rTE, u0)
    !! Computes the forward frequency domain electromagnetic response of a 1D layered earth.
    !! This function is callable from python.
!====================================================================!
    integer, intent(in) :: nFreq
    integer, intent(in) :: nC
    integer, intent(in) :: nLayers
    real(kind=8), intent(in) :: freqs(nFreq)
    real(kind=8), intent(in) :: fCoeffs(nC, nFreq)
    real(kind=8), intent(in) :: par(nLayers+1)
    real(kind=8), intent(in) :: thk(nLayers+1)
    complex(kind=8), intent(inout) :: rTE(nC, nFreq)
    complex(kind=8), intent(inout) :: u0(nC, nFreq)

    complex(kind=8) :: omega(nFreq)
    complex(kind=8) :: un(nC, nFreq, nLayers+1)
    complex(kind=8) :: Y(nC, nFreq, nLayers+1)
    complex(kind=8) :: Yn(nC, nFreq, nLayers+1)

    integer :: nL1

    nL1 = nLayers+1

    ! Setup the coefficients?
    call initCoefficients2(freqs, fCoeffs, nFreq, nC, nLayers, par, omega, un, Y, Yn)

    call M1_0(nFreq, nC, nL1, Yn, Y, un, thk, rTE, u0)

end subroutine
!====================================================================!
!====================================================================!
subroutine calcFdemSensitivity1D(nFreq, nC, nLayers, freqs, fCoeffs, par, thk, rTE, u0, sens)
    !! Computes the sensitivity for the frequency domain electromagnetic response of a 1D layered earth.
    !! This function is callable from python.
!====================================================================!
    integer, intent(in) :: nFreq
    integer, intent(in) :: nC
    integer, intent(in) :: nLayers
    real(kind=8), intent(in) :: freqs(nFreq)
    real(kind=8), intent(in) :: fCoeffs(nFreq, nC)
    real(kind=8), intent(in) :: par(nLayers+1)
    real(kind=8), intent(in) :: thk(nLayers+1)
    complex(kind=8), intent(inout) :: rTE(nC, nFreq)
    complex(kind=8), intent(inout) :: u0(nC, nFreq)
    complex(kind=8), intent(inout) :: sens(nC, nFreq, nLayers)

    complex(kind=8) :: a0, a1, a2
    complex(kind=8) :: omega(nFreq)
    complex(kind=8) :: un(nC, nFreq, nLayers+1)
    complex(kind=8) :: Y(nC, nFreq, nLayers+1)
    complex(kind=8) :: Yn(nC, nFreq, nLayers+1)

    integer :: nL1
    integer :: j
    integer :: k

    nL1 = nLayers+1

    ! Setup the coefficients?
    call initCoefficients2(freqs, fCoeffs, nFreq, nC, nLayers, par, omega, un, Y, Yn)

    do k = 1, nFreq
        do j = 1, nC
            sens(j, k, nLayers) = par(nL1) / (2.d0 * un(j, k, nL1))
        enddo
    enddo
    

    if (nLayers == 1) then
        do k = 1, nFreq
            do j = 1, nC
                u0(j, k) = un(j, k, 1)
                a0 = Yn(j, k, 1)
                a1 = Y(j, k, 2)
                a2 = 1.d0 / a0 + a1
                rTE(j, k) = (a0 - a1) *  a2
                sens(j, k, 1) = -2.d0 * a0 * sens(j, k, 1) * a2**2.d0
            enddo
        enddo
    else
        call M1_1(Yn, Y, un, omega, thk, par, nFreq, nC, nL1, rTE, u0, sens)
    endif

end subroutine
!====================================================================!
!====================================================================!
subroutine M1_1(Yn, Y, un, omega, thk, par, nFreq, nC, nL1, rTE, u0, sens)
!====================================================================!
    integer :: nFreq
    integer :: nC
    integer :: nL1
    complex(kind=8), intent(inout) :: Yn(nC, nFreq, nL1)
    complex(kind=8), intent(inout) :: Y(nC, nFreq, nL1)
    complex(kind=8), intent(in) :: un(nC, nFreq, nL1)
    complex(kind=8), intent(inout) :: omega(nFreq)
    real(kind=8), intent(in) :: thk(nL1)
    real(kind=8), intent(in) :: par(nL1)
    complex(kind=8), intent(inout) :: rTE(nC, nFreq)
    complex(kind=8), intent(inout) :: u0(nC, nFreq)
    complex(kind=8), intent(inout) :: sens(nC, nFreq, nL1-1)

    real(kind=8) :: p
    real(kind=8) :: t

    complex(kind=8) :: a0, a1, a2, z
    complex(kind=8) :: b1, b2, b4, b5, b6, b7, b8, b9 ! On the stack, does this matter at large scale?
    complex(kind=8) :: un_, Y_, Yn_
    complex(kind=8) :: accumulate(nC, nFreq, nL1-1)

    integer :: i, i1, i2
    integer :: j
    integer :: k
    integer :: nLayers

    nLayers = nL1 - 1

    ! Initialize cumprod array
    accumulate = 1.d0

    do i = nLayers, 2, -1
        i1 = i + 1
        i2 = i - 1
        p = par(i)
        t = thk(i)

        do k = 1, nFreq
            do j = 1, nC
                Yn_ = Yn(j, k, i)
                Y_ = Y(j, k, i1)
                un_ = un(j, k, i)
                z = un_ * t
                b4 = cTanh(z)
                b5 = b4**2.d0
                b8 = (2.d0 * Yn_ * Y_ * b5)
                b6 = Y_ + (Yn_ * b4)
                b7 = Yn_ + (Y_ * b4)
                b2 = Y_**2.d0

                Y(j, k, i) = Yn_ * (b6 / b7)

                b6 = Yn_**2.d0

                b9 = (p / (2.d0 * un_ * (b7**2.d0)))

                b7 = b7**(-2.d0)
                b8 = b8 + ((b2 - b6) * b4) + 2.d0 * b6
                b4 = b6 * (1.d0 - b5)
                b6 = Yn_**3.d0
                accumulate(j, k, i) = b4 * b7
                b4 = b2 * Yn_
                b7 = b4 - b6
                b1 = omega(k) * t * b7
                b8 = b8 + (b1 * b5) - b1

                sens(j, k, i2) = b8 * b9
            enddo
        enddo
    enddo


    ! Cumulative product over third dimension
    do i = 2, nLayers
        i1 = i - 1
        do k = 1, nFreq
            do j = 1, nC
                accumulate(j, k, i) = accumulate(j, k, i) * accumulate(j, k, i1)
            enddo
        enddo
    enddo

    do k = 1, nFreq
        do j = 1, nC
            u0(j, k) = un(j, k, 1)

            a0 = Yn(j, k, 1)
            a1 = Y(j, k, 2)
            a2 = 1.d0 / (a0 + a1)

            rTE(j, k) = (a0 - a1) * a2

            Y(j, k, 2) = a2**2.d0
            Yn(j, k, 1) = -2.d0 * a0 * Y(j, k, 2)
        enddo
    enddo

    do i = 1, nLayers
        do k = 1, nFreq
            do j = 1, nC
                sens(j, k, i) = sens(j, k, i) * Yn(j, k, 1) * accumulate(j, k, i)
            enddo
        enddo
    enddo

end subroutine
!====================================================================!
!====================================================================!
subroutine M1_0(nFreq, nC, nL1, Yn, Y, un, thk, rTE, u0)
!====================================================================!
    integer, intent(in) :: nFreq
    integer, intent(in) :: nC
    integer, intent(in) :: nL1
    complex(kind=8), intent(in) :: Yn(nC, nFreq, nL1)
    complex(kind=8), intent(inout) :: Y(nC, nFreq, nL1)
    complex(kind=8), intent(in) :: un(nC, nFreq, nL1)
    real(kind=8), intent(in) :: thk(nL1)
    complex(kind=8), intent(inout) :: rTE(nC, nFreq)
    complex(kind=8), intent(inout) :: u0(nC, nFreq)

    integer :: i, i1
    integer :: j
    integer :: k
    integer :: nLayers

    real(kind=8) :: t
    complex(kind=8) :: a0, a1, a2, a3, a4, z

    nLayers = nL1 - 1
    i=nLayers
    do i = nLayers, 2, -1
        i1 = i + 1
        t = thk(i)
        do k = 1, nFreq
            do j = 1, nC
                a3 = Yn(j, k, i)
                a4 = Y(j, k, i1)
                z = un(j, k, i) * t
                a0 = cTanh(z) ! Use existing memory to save space
                a1 = a4 + (a3 * a0) ! Numerator
                a2 = 1.d0 / (a3 + (a4 * a0)) ! Denominator
                Y(j, k, i) = a3 * a1 * a2
            enddo
        enddo
    enddo

    
    do k = 1, nFreq
        do j = 1, nC
            u0(j, k) = un(j, k, 1)
            a0 = Yn(j, k, 1)
            a1 = Y(j, k, 2)
            rTE(j, k) = (a0 - a1) / (a0 + a1)
        enddo
    enddo

    ! call M0_0(Yn, Y, un, rTE, u0)
end subroutine
!====================================================================!
! !====================================================================!
! subroutine M0_0(Yn, Y, un, rTE, u0)
! !====================================================================!
!     complex(kind=8), intent(in) :: Yn(:,:,:)
!     complex(kind=8), intent(in) :: Y(:,:,:)
!     complex(kind=8), intent(in) :: un(:,:,:)
!     complex(kind=8), intent(inout) :: rTE(:,:)
!     complex(kind=8), intent(inout) :: u0(:,:)

!     u0 = un(:,:,1)
!     rTE = Yn(:,:,1) - Y(:,:,2)
!     rTE = rTE / (Yn(:,:,1) + Y(:,:,2))
! end subroutine
! !====================================================================!
!====================================================================!
subroutine calcHxx(nFreq, nC0, nC1, i, tHeight, rHeight, moments, rx, separation, rTE0, Jw0, J0, rTE1, Jw1, J1, H, H0)
!====================================================================!
    integer, intent(in) :: nFreq
    integer, intent(in) :: nC0
    integer, intent(in) :: nC1
    integer, intent(in) :: i
    real(kind=8), intent(in) :: tHeight(nFreq)
    real(kind=8), intent(in) :: rHeight(nFreq)
    real(kind=8), intent(in) :: moments(nFreq)
    real(kind=8), intent(in) :: rx(nFreq)
    real(kind=8), intent(in) :: separation(nFreq)
    complex(kind=8), intent(in) :: rTE0(nC0, nFreq)
    real(kind=8), intent(in) :: Jw0(nC0)
    real(kind=8), intent(in) :: J0(nC0, nFreq)
    complex(kind=8), intent(in) :: rTE1(nC1, nFreq)
    real(kind=8), intent(in) :: Jw1(nC1)
    real(kind=8), intent(in) :: J1(nC1, nFreq)
    complex(kind=8), intent(inout) :: H(nFreq)
    complex(kind=8), intent(inout) :: H0(nFreq)

    integer :: j

    complex(kind=8) :: k, rTE0_, rTE1_
    real(kind=8) :: a0, a1, b0, c0, d0, d1, hDiff, hSum, r, J0_, J1_
    real(kind=8) :: w0, w1


    hSum = rHeight(i) + tHeight(i)
    hDiff = rHeight(i) - tHeight(i)
    r = 1.d0 / separation(i)

    c0 = -(moments(i) / pi4) * r
    d0 = c0 * ((rx(i) * r)**2.d0)
    d1 = c0 * (r - ((2.d0 * rx(i)**2.d0) * (r**3.d0)))
    
    do j = 1, nC0

        w0 = d0 * Jw0(j)
        J0_ = J0(j, i)
        rTE0_ = rTE0(j, i)
        w1 = d1 * Jw1(j)
        J1_ = J1(j, i)
        rTE1_ = rTE1(j, i)

        ! Temporary variables
        a0 = exp(-J0_ * (hSum))
        a1 = J0_**2.d0

        k = (a0 - (rTE0_ * exp(J0_ * hDiff))) * a1
        H(i) = H(i) + (k * w0) ! First bessel contribution
        k = a0 * a1
        H0(i) = H0(i) + (k * w0) ! First bessel contribution to free space

        ! Use the first 120 entries of the second bessel in this loop.
        b0 = exp(-J1_ * (hSum))
        k = (b0 - (rTE1_ * exp(J1_ * hDiff))) * J1_
        H(i) = H(i) + (k * w1)
        k = b0 * J1_
        H0(i) = H0(i) + (k * w1)
    enddo

    do j = nC0+1, nC1

        w1 = d1 * Jw1(j)
        J1_ = J1(j, i)
        rTE1_ = rTE1(j, i)

        ! Add the last 20 entries of the second bessel in this loop.
        b0 = exp(-J1_ * (hSum))
        k = (b0 - (rTE1_ * exp(J1_ * hDiff))) * J1_
        H(i) = H(i) + (k * w1)
        k = b0 * J1_
        H0(i) = H0(i) + (k * w1)
    enddo
end subroutine
!====================================================================!
!====================================================================!
subroutine calcHxz(nFreq, nC1, i, tHeight, rHeight, moments, rx, separation, rTE1, Jw1, J1, H, H0)
!====================================================================!
    integer, intent(in) :: nFreq
    integer, intent(in) :: nC1
    integer, intent(in) :: i
    real(kind=8), intent(in) :: tHeight(nFreq)
    real(kind=8), intent(in) :: rHeight(nFreq)
    real(kind=8), intent(in) :: moments(nFreq)
    real(kind=8), intent(in) :: rx(nFreq)
    real(kind=8), intent(in) :: separation(nFreq)
    complex(kind=8), intent(in) :: rTE1(nC1, nFreq)
    real(kind=8), intent(in) :: Jw1(nC1)
    real(kind=8), intent(in) :: J1(nC1, nFreq)
    complex(kind=8), intent(inout) :: H(nFreq)
    complex(kind=8), intent(inout) :: H0(nFreq)

    integer :: j

    complex(kind=8) :: k, rTE1_
    real(kind=8) :: a1, b0, d1, hDiff, hSum, J1_
    real(kind=8) :: w1

    hSum = rHeight(i) + tHeight(i)
    hDiff = rHeight(i) - tHeight(i)

    d1 = (rx(i) * moments(i)) / (pi4 * separation(i))
    
    do j = 1, nC1

        w1 = d1 * Jw1(j)
        J1_ = J1(j, i)
        rTE1_ = rTE1(j, i)

        b0 = exp(-J1_ * hSum)
        a1 = J1_**2.d0
        k = (b0 - (rTE1_ * exp(J1_ * hDiff))) * a1
        H(i) = H(i) + (k * w1)
        k = b0 * a1
        H0(i) = H0(i) + (k * w1)
    enddo
end subroutine
!====================================================================!
!====================================================================!
subroutine calcHzx(nFreq, nC1, i, tHeight, rHeight, moments, rx, separation, rTE1, u1, Jw1, J1, H, H0)
!====================================================================!
    integer, intent(in) :: nFreq
    integer, intent(in) :: nC1
    integer, intent(in) :: i
    real(kind=8), intent(in) :: tHeight(nFreq)
    real(kind=8), intent(in) :: rHeight(nFreq)
    real(kind=8), intent(in) :: moments(nFreq)
    real(kind=8), intent(in) :: rx(nFreq)
    real(kind=8), intent(in) :: separation(nFreq)
    complex(kind=8), intent(in) :: rTE1(nC1, nFreq)
    complex(kind=8), intent(in) :: u1(nC1, nFreq)
    real(kind=8), intent(in) :: Jw1(nC1)
    real(kind=8), intent(in) :: J1(nC1, nFreq)
    complex(kind=8), intent(inout) :: H(nFreq)
    complex(kind=8), intent(inout) :: H0(nFreq)

    integer :: j

    complex(kind=8) :: b0, k, rTE1_, u1_
    real(kind=8) :: a1, d1, hDiff, hSum, J1_
    real(kind=8) :: w1

    hSum = rHeight(i) + tHeight(i)
    hDiff = rHeight(i) - tHeight(i)

    d1 = (rx(i) * moments(i)) / (pi4 * separation(i))
    
    do j = 1, nC1

        w1 = d1 * Jw1(j)
        u1_ = u1(j, i)
        J1_ = J1(j, i)
        rTE1_ = rTE1(j, i)

        b0 = exp(-u1_ * hSum)
        a1 = J1_**2.d0
        k = (b0 - (rTE1_ * exp(u1_ * hDiff))) * a1
        H(i) = H(i) + (k * w1)
        k = b0 * a1
        H0(i) = H0(i) + (k * w1)
    enddo
end subroutine
!====================================================================!
!====================================================================!
subroutine calcHzz(nFreq, nC0, i, tHeight, rHeight, moments, separation, rTE, u0, Jw0, J0, H, H0)
!====================================================================!
    integer, intent(in) :: nFreq
    integer, intent(in) :: nC0
    integer, intent(in) :: i
    real(kind=8), intent(in) :: tHeight(nFreq)
    real(kind=8), intent(in) :: rHeight(nFreq)
    real(kind=8), intent(in) :: moments(nFreq)
    real(kind=8), intent(in) :: separation(nFreq)
    complex(kind=8), intent(in) :: rTE(nC0, nFreq)
    complex(kind=8), intent(in) :: u0(nC0, nFreq)
    real(kind=8), intent(in) :: Jw0(nC0)
    real(kind=8), intent(in) :: J0(nC0, nFreq)
    complex(kind=8), intent(inout) :: H(nFreq)
    complex(kind=8), intent(inout) :: H0(nFreq)

    integer :: j

    complex(kind=8) :: a0, a1, k, u0_, rTE_
    real(kind=8) :: a2, hDiff, hSum, J0_
    real(kind=8) :: w0_


    hSum = rHeight(i) + tHeight(i)
    hDiff = rHeight(i) - tHeight(i)
    a2 = moments(i) / (pi4 * separation(i))
    
    do j = 1, nC0

        w0_ = a2 * Jw0(j)
        u0_ = u0(j, i)
        J0_ = J0(j, i)
        rTE_ = rTE(j, i)

        ! Temporary variables
        a0 = exp(-u0_ * hSum)
        a1 = J0_**3.d0 / u0_

        ! Equation 4.46K(lam)
        k = (a0 + (rTE_ * exp(u0_ * hDiff))) * a1
        H(i) = H(i) + (k * w0_)

        ! Free Space response
        k = a0 * a1
        H0(i) = H0(i) + (k * w0_)
    enddo

end subroutine
!====================================================================!
!====================================================================!
subroutine initModel(parIn, thkIn, nLayers, par, thk)
!====================================================================!
    integer :: nLayers ! Number of layers
    real(kind=8), intent(in) :: parIn(nLayers) ! Parameter values
    real(kind=8), intent(in) :: thkIn(nLayers) ! Thicknesses
    ! real(kind=8), intent(in) :: chimIn(nLayers) ! Magnetic susceptibility
    real(kind=8), intent(inout) :: par(nLayers+1)
    real(kind=8), intent(inout) :: thk(nLayers+1)
    ! real(kind=8), intent(inout) :: chim(nLayers+1)

    integer :: nL1

    nL1 = nLayers + 1

    par(1) = 0.d0 ; par(2:nL1) = parIn
    ! chim(1) = 0.d0 ; chim(2:nL1) = chimIn
    thk(1) = NaN ; thk(2:nL1) = thkIn

end subroutine
!====================================================================!
!====================================================================!
subroutine initCoefficients2(freqs, fCoeffs, nFreq, nC, nLayers, par, omega, un, Y, Yn)
!====================================================================!
    integer :: nFreq ! Number of frequencies
    integer :: nC ! Number of filter coefficients
    integer :: nLayers ! Number of layers
    real(kind=8), intent(in) :: freqs(nFreq) ! Frequencies
    real(kind=8), intent(in) :: fCoeffs(nC, nFreq) ! Hankel Filter Coefficients
    real(kind=8), intent(in) :: par(nLayers+1)
    complex(kind=8), intent(inout) :: omega(nFreq)
    complex(kind=8), intent(inout) :: un(nC, nFreq, nLayers+1)
    complex(kind=8), intent(inout) :: Y(nC, nFreq, nLayers+1)
    complex(kind=8), intent(inout) :: Yn(nC, nFreq, nLayers+1)

    complex(kind=8) :: b0, b2(nC, nFreq)
    real(kind=8) :: p

    integer :: nL1
    integer :: i
    integer :: j
    integer :: k

    nL1 = nLayers + 1

    ! Compute the angular frequencies from the system frequencies
    do i = 1, nFreq
        omega(i) = complex(0.d0, pi2 * freqs(i)) * mu0
    enddo

    ! Compute the Admitivity, yn=j*omega*eps+sigma
    do k = 1, nFreq
        b0 = 1.d0 / omega(k)
        do j = 1, nC
            Y(j, k, 1) = b0
        enddo
    enddo

    p = par(1)
    do k = 1, nFreq
        do j = 1, nC
            b2(j, k) = fCoeffs(j, k)**2.d0
            un(j, k, 1) = sqrt(((omega(k) * eps0 + p) * omega(k)) + b2(j, k))
            Yn(j, k, 1) = un(j, k, 1) * Y(j, k, 1)
        enddo
    enddo

    do i = 2, nL1
        p = par(i)
        do k = 1, nFreq
            do j = 1, nC
                un(j, k, i) = sqrt(((omega(k) * eps0 + p) * omega(k)) + b2(j, k))
                Yn(j, k, i) = un(j, k, i) * Y(j, k, 1)
            enddo
        enddo
    enddo

    
    Y(:, :, 1) = cNaN
    Y(:, :, 2:nLayers) = 0.d0
    Y(:, :, nL1) = Yn(:, :, nL1)

end subroutine
!====================================================================!
!====================================================================!
subroutine setLambdas(this, nFreq, separation)
!====================================================================!
    class(t_lambdas) :: this
    integer, intent(in) :: nFreq
    real(kind=8), intent(in) :: separation(nFreq)

    if (allocated(this%w0)) then
        if (size(this%lam0, 2) /= nFreq) then
            deallocate(this%lam0, this%lam1)
            allocate(this%lam0(nCJ0, nFreq))
            allocate(this%lam1(nCJ1, nFreq))
            call set_lambda(this, nFreq, separation)
        endif
        return
    endif    

    allocate(this%w0(nCJ0), this%w1(nCJ1))
    call set_w0(this)
    call set_w1(this)

    allocate(this%lam0(nCJ0, nFreq))
    allocate(this%lam1(nCJ1, nFreq))
    call set_lambda(this, nFreq, separation)

end subroutine
!====================================================================!
!====================================================================!
subroutine set_lambda(this, nFreq, separation)
!====================================================================!
    class(t_lambdas) :: this
    integer, intent(in) :: nFreq
    real(kind=8), intent(in) :: separation(nFreq)

    real(kind=8) :: r
    real(kind=8), parameter :: a0 = -8.3885d0
    real(kind=8), parameter :: s0 = 9.04226468670d-02
    real(kind=8), parameter :: a1 = -7.91001919d0
    real(kind=8), parameter :: s1 = 8.7967143957d-02
    real(kind=8) :: tmp(nCJ1)

    integer :: j, k

    do j = 1, nCJ1
        tmp(j) = real(j-1, kind=8)
    enddo

    do k = 1, nFreq
        r = 1.d0 / separation(k)
        do j = 1, nCJ0
            this%lam0(j, k) = 10.d0**((tmp(j) * s0) + a0) * r
            this%lam1(j, k) = 10.d0**((tmp(j) * s1) + a1) * r
        enddo
        do j = 121, nCJ1
            this%lam1(j, k) = 10.d0**((tmp(j) * s1) + a1) * r
        enddo
    enddo
end subroutine
!====================================================================!
!====================================================================!
subroutine set_w0(this)
!====================================================================!
    class(t_lambdas) :: this
    this%w0 = [&
        9.62801364263e-07, -5.02069203805e-06, 1.25268783953e-05, -1.99324417376e-05, 2.29149033546e-05,&
        -2.04737583809e-05, 1.49952002937e-05, -9.37502840980e-06, 5.20156955323e-06, -2.62939890538e-06,&
        1.26550848081e-06, -5.73156151923e-07, 2.76281274155e-07, -1.09963734387e-07, 7.38038330280e-08,&
        -9.31614600001e-09, 3.87247135578e-08, 2.10303178461e-08, 4.10556513877e-08, 4.13077946246e-08,&
        5.68828741789e-08, 6.59543638130e-08, 8.40811858728e-08, 1.01532550003e-07, 1.26437360082e-07,&
        1.54733678097e-07, 1.91218582499e-07, 2.35008851918e-07, 2.89750329490e-07, 3.56550504341e-07,&
        4.39299297826e-07, 5.40794544880e-07, 6.66136379541e-07, 8.20175040653e-07, 1.01015545059e-06,&
        1.24384500153e-06, 1.53187399787e-06, 1.88633707689e-06, 2.32307100992e-06, 2.86067883258e-06,&
        3.52293208580e-06, 4.33827546442e-06, 5.34253613351e-06, 6.57906223200e-06, 8.10198829111e-06,&
        9.97723263578e-06, 1.22867312381e-05, 1.51305855976e-05, 1.86329431672e-05, 2.29456891669e-05,&
        2.82570465155e-05, 3.47973610445e-05, 4.28521099371e-05, 5.27705217882e-05, 6.49856943660e-05,&
        8.00269662180e-05, 9.85515408752e-05, 1.21361571831e-04, 1.49454562334e-04, 1.84045784500e-04,&
        2.26649641428e-04, 2.79106748890e-04, 3.43716968725e-04, 4.23267056591e-04, 5.21251001943e-04,&
        6.41886194381e-04, 7.90483105615e-04, 9.73420647376e-04, 1.19877439042e-03, 1.47618560844e-03,&
        1.81794224454e-03, 2.23860214971e-03, 2.75687537633e-03, 3.39471308297e-03, 4.18062141752e-03,&
        5.14762977308e-03, 6.33918155348e-03, 7.80480111772e-03, 9.61064602702e-03, 1.18304971234e-02,&
        1.45647517743e-02, 1.79219149417e-02, 2.20527911163e-02, 2.71124775541e-02, 3.33214363101e-02,&
        4.08864842127e-02, 5.01074356716e-02, 6.12084049407e-02, 7.45146949048e-02, 9.00780900611e-02,&
        1.07940155413e-01, 1.27267746478e-01, 1.46676027814e-01, 1.62254276550e-01, 1.68045766353e-01,&
        1.52383204788e-01, 1.01214136498e-01, -2.44389126667e-03, -1.54078468398e-01, -3.03214415655e-01,&
        -2.97674373379e-01, 7.93541259524e-03, 4.26273267393e-01, 1.00032384844e-01, -4.94117404043e-01,&
        3.92604878741e-01, -1.90111691178e-01, 7.43654896362e-02, -2.78508428343e-02, 1.09992061155e-02,&
        -4.69798719697e-03, 2.12587632706e-03, -9.81986734159e-04, 4.44992546836e-04, -1.89983519162e-04,&
        7.31024164292e-05, -2.40057837293e-05, 6.23096824846e-06, -1.12363896552e-06, 1.04470606055e-07]
end subroutine
!====================================================================!
!====================================================================!
subroutine set_w1(this)
!====================================================================!
    class(t_lambdas) :: this
    this%w1 = [&
        -6.76671159511e-14, 3.39808396836e-13, -7.43411889153e-13, 8.93613024469e-13, -5.47341591896e-13,&
        -5.84920181906e-14, 5.20780672883e-13, -6.92656254606e-13, 6.88908045074e-13, -6.39910528298e-13,&
        5.82098912530e-13, -4.84912700478e-13, 3.54684337858e-13, -2.10855291368e-13, 1.00452749275e-13,&
        5.58449957721e-15, -5.67206735175e-14, 1.09107856853e-13, -6.04067500756e-14, 8.84512134731e-14,&
        2.22321981827e-14, 8.38072239207e-14, 1.23647835900e-13, 1.44351787234e-13, 2.94276480713e-13,&
        3.39965995918e-13, 6.17024672340e-13, 8.25310217692e-13, 1.32560792613e-12, 1.90949961267e-12,&
        2.93458179767e-12, 4.33454210095e-12, 6.55863288798e-12, 9.78324910827e-12, 1.47126365223e-11,&
        2.20240108708e-11, 3.30577485691e-11, 4.95377381480e-11, 7.43047574433e-11, 1.11400535181e-10,&
        1.67052734516e-10, 2.50470107577e-10, 3.75597211630e-10, 5.63165204681e-10, 8.44458166896e-10,&
        1.26621795331e-09, 1.89866561359e-09, 2.84693620927e-09, 4.26886170263e-09, 6.40104325574e-09,&
        9.59798498616e-09, 1.43918931885e-08, 2.15798696769e-08, 3.23584600810e-08, 4.85195105813e-08,&
        7.27538583183e-08, 1.09090191748e-07, 1.63577866557e-07, 2.45275193920e-07, 3.67784458730e-07,&
        5.51470341585e-07, 8.26916206192e-07, 1.23991037294e-06, 1.85921554669e-06, 2.78777669034e-06,&
        4.18019870272e-06, 6.26794044911e-06, 9.39858833064e-06, 1.40925408889e-05, 2.11312291505e-05,&
        3.16846342900e-05, 4.75093313246e-05, 7.12354794719e-05, 1.06810848460e-04, 1.60146590551e-04,&
        2.40110903628e-04, 3.59981158972e-04, 5.39658308918e-04, 8.08925141201e-04, 1.21234066243e-03,&
        1.81650387595e-03, 2.72068483151e-03, 4.07274689463e-03, 6.09135552241e-03, 9.09940027636e-03,&
        1.35660714813e-02, 2.01692550906e-02, 2.98534800308e-02, 4.39060697220e-02, 6.39211368217e-02,&
        9.16763946228e-02, 1.28368795114e-01, 1.73241920046e-01, 2.19830379079e-01, 2.51193131178e-01,&
        2.32380049895e-01, 1.17121080205e-01, -1.17252913088e-01, -3.52148528535e-01, -2.71162871370e-01,&
        2.91134747110e-01, 3.17192840623e-01, -4.93075681595e-01, 3.11223091821e-01, -1.36044122543e-01,&
        5.12141261934e-02, -1.90806300761e-02, 7.57044398633e-03, -3.25432753751e-03, 1.49774676371e-03,&
        -7.24569558272e-04, 3.62792644965e-04, -1.85907973641e-04, 9.67201396593e-05, -5.07744171678e-05,&
        2.67510121456e-05, -1.40667136728e-05, 7.33363699547e-06, -3.75638767050e-06, 1.86344211280e-06,&
        -8.71623576811e-07, 3.61028200288e-07, -1.05847108097e-07, -1.51569361490e-08, 6.67633241420e-08,&
        -8.33741579804e-08, 8.31065906136e-08, -7.53457009758e-08, 6.48057680299e-08, -5.37558016587e-08,&
        4.32436265303e-08, -3.37262648712e-08, 2.53558687098e-08, -1.81287021528e-08, 1.20228328586e-08,&
        -7.10898040664e-09, 3.53667004588e-09, -1.36030600198e-09, 3.52544249042e-10, -4.53719284366e-11]
end subroutine
!====================================================================!
!====================================================================!
function cTanh(z) result(res)
  !! Evaluate tanh(z) for complex numbers. Prevents NaN overflow on some systems
!====================================================================!
    complex(kind=8), intent(in) :: z
    complex(kind=8) :: res
    complex(kind=8) :: tmp
    
    if (real(z) > 0.d0) then
        tmp = exp(-2.d0 * z)
        res = (1.d0 - tmp) / (1.d0 + tmp)
    else
        tmp = exp(2.d0 * z)
        res = (tmp - 1.d0) / (tmp + 1.d0)
    endif
end function
!====================================================================!
end module

















