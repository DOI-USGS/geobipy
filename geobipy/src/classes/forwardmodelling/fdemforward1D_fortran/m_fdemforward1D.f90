module fdemforward1D
implicit none

! Define some fixed parameters
real(kind=8),parameter :: NaN = transfer((/ Z'00000000', Z'7FF80000' /),1.0_8)
complex(kind=8),parameter :: cNaN = complex(NaN,NaN)
real(kind=8),parameter :: pi = dacos(-1.d0)
real(kind=8),parameter :: mu0 = 4.d-7*pi ! H/m
real(kind=8),parameter :: c = 299792458.d0 ! m/s
real(kind=8),parameter :: eps0 = 1.d0/(mu0*(c**2.d0)) ! F/m

contains

!====================================================================!
subroutine calcFdemforward1D(nFreq, nC, nLayers, freqs, fCoeffs, parIn, thkIn, chimIn, rTE, u0)
    !! Computes the forward frequency domain electromagnetic response of a 1D layered earth.
    !! This function is callable from python.
!====================================================================!
    integer, intent(in) :: nFreq
    integer, intent(in) :: nC
    integer, intent(in) :: nLayers
    real(kind=8), intent(in) :: freqs(nFreq)
    !f2py intent(in) :: freqs
    real(kind=8), intent(in) :: fCoeffs(nFreq, nC)
    !f2py intent(in) :: fCoeffs
    real(kind=8), intent(in) :: parIn(nLayers)
    !f2py intent(in) ::  parIn
    real(kind=8), intent(in) :: thkIn(nLayers)
    !f2py intent(in) :: thkIn
    real(kind=8), intent(in) :: chimIn(nLayers)
    !f2py intent(in) :: chimIn
    complex(kind=8), intent(inout) :: rTE(nFreq, nC)
    !f2py intent(in,out) :: rTE
    complex(kind=8), intent(inout) :: u0(nFreq, nC)
    !f2py intent(in,out) :: u0

    real(kind=8), allocatable :: par(:)
    real(kind=8), allocatable :: thk(:)
    real(kind=8), allocatable :: chim(:)

    complex(kind=8), allocatable :: omega(:)
    complex(kind=8), allocatable :: un(:,:,:)
    complex(kind=8), allocatable :: Y(:,:,:)
    complex(kind=8), allocatable :: Yn(:,:,:)

    integer :: nL1
    integer :: istat

    nL1 = nLayers+1

    ! Allocate memory (for some reason f2py won't let me do this inside subroutines)
    allocate(par(nL1), thk(nL1), chim(nL1), stat=istat)
    if (istat /= 0) stop "Could not allocate par, thk, chim"
    allocate(omega(nFreq), stat=istat)
    if (istat /= 0) stop "Could not allocate omega"
    allocate(un(nFreq,nC,nL1), stat=istat)
    if (istat /= 0) stop "Could not allocate un"
    un = 0.d0
    allocate(Yn(nFreq,nC,nL1), stat=istat)
    if (istat /= 0) stop "Could not allocate Yn"
    Yn = 0.d0
    allocate(Y(nFreq,nC,nL1), stat=istat)
    if (istat /= 0) stop "Could not allocate Y"

    ! Setup the coefficients?
    call initCoefficients(freqs, fCoeffs, parIn, thkIn, chimIn, nFreq, nC, nLayers, omega, par, thk, chim, un, Y, Yn)

    call M1_0(Yn, Y, un, thk, nL1, rTE, u0)

    deallocate(omega, stat=istat)
    if (istat /= 0) stop "Could not deallocate omega"
    deallocate(par, thk, chim, stat=istat)
    if (istat /= 0) stop "Could not deallocate par, thk, chim"
    deallocate(un, stat=istat)
    if (istat /= 0) stop "Could not deallocate un"
    deallocate(Yn, stat=istat)
    if (istat /= 0) stop "Could not deallocate Yn"
    deallocate(Y, stat=istat)
    if (istat /= 0) stop "Could not deallocate Y"
end subroutine
!====================================================================!
!====================================================================!
subroutine calcFdemSensitivity1D(nFreq, nC, nLayers, freqs, fCoeffs, parIn, thkIn, chimIn, rTE, u0, sens)
    !! Computes the sensitivity for the frequency domain electromagnetic response of a 1D layered earth.
    !! This function is callable from python.
!====================================================================!
    integer, intent(in) :: nFreq
    integer, intent(in) :: nC
    integer, intent(in) :: nLayers
    real(kind=8), intent(in) :: freqs(nFreq)
    !f2py intent(in) :: freqs
    real(kind=8), intent(in) :: fCoeffs(nFreq, nC)
    !f2py intent(in) :: fCoeffs
    real(kind=8), intent(in) :: parIn(nLayers)
    !f2py intent(in) ::  parIn
    real(kind=8), intent(in) :: thkIn(nLayers)
    !f2py intent(in) :: thkIn
    real(kind=8), intent(in) :: chimIn(nLayers)
    !f2py intent(in) :: chimIn
    complex(kind=8), intent(inout) :: rTE(nFreq, nC)
    !f2py intent(in,out) :: rTE
    complex(kind=8), intent(inout) :: u0(nFreq, nC)
    !f2py intent(in,out) :: u0
    complex(kind=8), intent(inout) :: sens(nFreq, nC, nLayers)
    !f2py intent(in,out) :: sens

    real(kind=8), allocatable :: par(:)
    real(kind=8), allocatable :: thk(:)
    real(kind=8), allocatable :: chim(:)

    complex(kind=8), allocatable :: omega(:)
    complex(kind=8), allocatable :: un(:,:,:)
    complex(kind=8), allocatable :: Y(:,:,:)
    complex(kind=8), allocatable :: Yn(:,:,:)

    integer :: nL1
    integer :: istat

    nL1 = nLayers+1

    ! Allocate memory (for some reason f2py won't let me do this inside subroutines)
    allocate(par(nL1), thk(nL1), chim(nL1), stat=istat)
    if (istat /= 0) stop "Could not allocate par, thk, chim"
    allocate(omega(nFreq), stat=istat)
    if (istat /= 0) stop "Could not allocate omega"
    allocate(un(nFreq,nC,nL1), stat=istat)
    if (istat /= 0) stop "Could not allocate un"
    un = 0.d0
    allocate(Yn(nFreq,nC,nL1), stat=istat)
    if (istat /= 0) stop "Could not allocate Yn"
    Yn = 0.d0
    allocate(Y(nFreq,nC,nL1), stat=istat)
    if (istat /= 0) stop "Could not allocate Y"

    ! Setup the coefficients?
    call initCoefficients(freqs, fCoeffs, parIn, thkIn, chimIn, nFreq, nC, nLayers, omega, par, thk, chim, un, Y, Yn)

    sens(:,:,nLayers) = par(nL1) / (2.d0 * un(:,:,nL1))

    if (nLayers == 1) then
        call M0_0(Yn, Y, un, rTE, u0)
        sens = -2.d0 * Yn(:,:,1:1) * sens(:,:,1:1)
        sens = sens(:,:,1:1) / (Yn(:,:,1:1) + Y(:,:,2:2))**2.d0
    else
        call M1_1(Yn, Y, un, omega, thk, par, nFreq, nC, nL1, rTE, u0, sens)
    endif

    deallocate(omega, stat=istat)
    if (istat /= 0) stop "Could not deallocate omega"
    deallocate(par, thk, chim, stat=istat)
    if (istat /= 0) stop "Could not deallocate par, thk, chim"
    deallocate(un, stat=istat)
    if (istat /= 0) stop "Could not deallocate un"
    deallocate(Yn, stat=istat)
    if (istat /= 0) stop "Could not deallocate Yn"
    deallocate(Y, stat=istat)
    if (istat /= 0) stop "Could not deallocate Y"
end subroutine
!====================================================================!
!====================================================================!
subroutine M1_1(Yn, Y, un, omega, thk, par, nFreq, nC, nL1, rTE, u0, sens)
!====================================================================!
    integer :: nFreq
    integer :: nC
    integer :: nL1
    complex(kind=8), intent(inout), target :: Yn(:,:,:)
    complex(kind=8), intent(inout), target :: Y(:,:,:)
    complex(kind=8), intent(in), target :: un(:,:,:)
    complex(kind=8), intent(inout), target :: omega(:)
    real(kind=8), intent(in) :: thk(:)
    real(kind=8), intent(in) :: par(:)
    complex(kind=8), intent(inout) :: rTE(:,:)
    complex(kind=8), intent(inout) :: u0(:,:)
    complex(kind=8), intent(inout) :: sens(:,:,:)

    real(kind=8) :: p
    real(kind=8) :: t

    complex(kind=8), dimension(nFreq,nC) :: b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10 ! On the stack, does this matter at large scale?
    complex(kind=8) :: oMu0
    complex(kind=8) :: t1, t2, t3, t4, t5, t6, tanuht
    complex(kind=8) :: un_, Y_, Yn_
    complex(kind=8) :: z0,z1,z2,z3,z4
    complex(kind=8) :: accumulate(nFreq, nC, nL1-1)

    integer :: i, i1, istat
    integer :: j
    integer :: k
    integer :: nLayers

    nLayers = nL1 - 1

    ! Initialize cumprod array
    accumulate = 1.d0

    omega = omega * mu0

    do j = 1, nC
        b10(:,j) = omega
    enddo

    do i = nLayers,2,-1
        i1 = i + 1
        p = par(i)
        t = thk(i)
        b1 = Yn(:,:,i)
        b2 = Y(:,:,i1)
        b3 = un(:,:,i)
        b4 = tanh(b3*t)
        b5 = b4**2.d0
        b8 = (2.d0 * b1 * b2 * b5)
        b6 = b2 + (b1 * b4)
        b7 = b1 + (b2 * b4)
        b2 = b2**2.d0

        Y(:,:,i) = b1 * (b6 / b7)

        b6 = b1**2.d0

        b9 = (p / (2.d0 * b3 * (b7**2.d0)))

        b7 = b7**-2.d0
        b8 = b8 + ((b2 - b6) * b4) + 2.d0 * b6
        b4 = b6 * (1.d0 - b5)
        b6 = b1**3.d0
        accumulate(:,:,i) = b4 * b7
        b4 = b2 * b1
        b7 = b4 - b6
        b1 = b10 * t * b7
        b8 = b8 + (b1 * b5) - b1

        sens(:,:,i-1) = b8 * b9
    enddo


    ! Cumulative product over third dimension
    do i = 2, nLayers
        accumulate(:,:,i) = accumulate(:,:,i)*accumulate(:,:,i-1)
    enddo

    u0 = un(:,:,1)
    rTE = Yn(:,:,1) - Y(:,:,2)
    rTE = rTE / (Yn(:,:,1) + Y(:,:,2))

    Y(:,:,2) = (Yn(:,:,1) + Y(:,:,2))**-2.d0
    Yn(:,:,1) = -2.d0*Yn(:,:,1) * Y(:,:,2)

    sens = sens * accumulate
    do i = 1, nLayers
        sens(:,:,i) = sens(:,:,i) * Yn(:,:,1)
    enddo

end subroutine
!====================================================================!
!====================================================================!
subroutine M1_0(Yn, Y, un, thk, nL1, rTE, u0)
!====================================================================!
    complex(kind=8), intent(in) :: Yn(:,:,:)
    complex(kind=8), intent(inout) :: Y(:,:,:)
    complex(kind=8), intent(in) :: un(:,:,:)
    real(kind=8), intent(in) :: thk(:)
    integer :: nL1
    complex(kind=8), intent(inout) :: rTE(:,:)
    complex(kind=8), intent(inout) :: u0(:,:)

    integer :: i, i1
    integer :: nLayers

    nLayers = nL1 - 1
    i=nLayers
    do i = nLayers,2,-1
      i1 = i + 1
      u0 = tanh(un(:,:,i) * thk(i)) ! Use existing memory to save space
      rTE = Y(:,:,i1) + (Yn(:,:,i)*u0) ! Numerator
      u0 = 1.d0 / (Yn(:,:,i) + (Y(:,:,i1)*u0)) ! Denominator
      Y(:,:,i) = Yn(:,:,i) * rTE * u0
    enddo
    call M0_0(Yn, Y, un, rTE, u0)
end subroutine
!====================================================================!
!====================================================================!
subroutine M0_0(Yn, Y, un, rTE, u0)
!====================================================================!
    complex(kind=8), intent(in) :: Yn(:,:,:)
    complex(kind=8), intent(in) :: Y(:,:,:)
    complex(kind=8), intent(in) :: un(:,:,:)
    complex(kind=8), intent(inout) :: rTE(:,:)
    complex(kind=8), intent(inout) :: u0(:,:)

    u0 = un(:,:,1)
    rTE = Yn(:,:,1) - Y(:,:,2)
    rTE = rTE / (Yn(:,:,1) + Y(:,:,2))
end subroutine
!====================================================================!
!====================================================================!
subroutine initCoefficients(freqs, fCoeffs, parIn, thkIn, chimIn, nFreq, nC, nLayers, omega, par, thk, chim, un, Y, Yn)
!====================================================================!
    integer :: nFreq ! Number of frequencies
    integer :: nC ! Number of filter coefficients
    integer :: nLayers ! Number of layers
    real(kind=8), intent(in) :: freqs(nFreq) ! Frequencies
    real(kind=8), intent(in) :: fCoeffs(nFreq, nC) ! Hankel Filter Coefficients
    real(kind=8), intent(in) :: parIn(nLayers) ! Parameter values
    real(kind=8), intent(in) :: thkIn(nLayers) ! Thicknesses
    real(kind=8), intent(in) :: chimIn(nLayers) ! Magnetic susceptibility
    complex(kind=8), intent(inout) :: omega(nFreq)
    real(kind=8), intent(inout) :: par(nLayers+1)
    real(kind=8), intent(inout) :: thk(nLayers+1)
    real(kind=8), intent(inout) :: chim(nLayers+1)
    complex(kind=8), intent(inout) :: un(:,:,:)
    complex(kind=8), intent(inout) :: Y(:,:,:)
    complex(kind=8), intent(inout) :: Yn(:,:,:)

    complex(kind=8), dimension(nFreq,nC) :: b1,b2
    complex(kind=8) :: unTmp
    real(kind=8) :: p

    integer :: nL1
    integer :: i
    integer :: j
    integer :: k

    nL1 = nLayers + 1

    par(1) = 0.d0 ; par(2:nL1) = parIn
    chim(1) = 0.d0 ; chim(2:nL1) = chimIn
    thk(1) = NaN ; thk(2:nL1) = thkIn

    ! Compute the angular frequencies from the system frequencies
    do i = 1, nFreq
        omega(i) = complex(0.d0,2.d0 * pi * freqs(i))
    enddo

    ! Compute the Admitivity, yn=j*omega*eps+sigma
    do j = 1, nC
        b1(:,j) = omega * mu0
    enddo
    Y(:,:,1) = 1.d0 / b1

    b2 = fCoeffs**2.d0

    do i = 1, nL1
      p = par(i)
      un(:,:,i) = sqrt(((b1*eps0 + p)*b1) + b2)
      Yn(:,:,i) = un(:,:,i) * Y(:,:,1)
    enddo

    Y(:,:,1) = cNaN
    Y(:,:,2:nLayers) = 0.d0
    Y(:,:,nL1) = Yn(:,:,nL1)

end subroutine
!====================================================================!
end module

















