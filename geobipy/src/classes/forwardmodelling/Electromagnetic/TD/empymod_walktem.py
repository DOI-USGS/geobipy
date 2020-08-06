import numpy as np
from empymod.model import (bipole, tem)
from scipy.integrate._quadrature import _cached_roots_legendre
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline

def empymod_walktem(system, model1d):
    """Custom wrapper of empymod.model.bipole.

    Here, we calculate WalkTEM data using the ``empymod.model.bipole`` routine
    as an example. We could achieve the same using ``empymod.model.dipole`` or
    ``empymod.model.loop``.

    We model the big source square loop by calculating only half of one side of
    the electric square loop and approximating the finite length dipole with 3
    point dipole sources. The result is then multiplied by 8, to account for
    all eight half-sides of the square loop.

    The implementation here assumes a central loop configuration, where the
    receiver (1 m2 area) is at the origin, and the source is a 40x40 m electric
    loop, centered around the origin.

    Note: This approximation of only using half of one of the four sides
          obviously only works for central, horizontal square loops. If your
          loop is arbitrary rotated, then you have to model all four sides of
          the loop and sum it up.


    Parameters
    ----------


    Returns
    -------
    WalkTEM : EMArray
        WalkTEM response (dB/dt).

    """

    depth = np.r_[0.0, model1d.depth[:-1]]
    res = np.r_[2e14, model1d.par]
    # === CALCULATE FREQUENCY-DOMAIN RESPONSE ===
    # We only define a few parameters here. You could extend this for any
    # parameter possible to provide to empymod.model.bipole.
    length = 0.5 * system.transmitterLoop.sideLength

    EM = bipole(
        src=[0.0, length,  length, length, 0.0, 0.0],  # El. bipole source; half of one side.
        rec=[0, 0, 0, 0, 90],         # Receiver at the origin, vertical.
        depth = depth,        # Depth-model, adding air-interface.
        res = res,         # Provided resistivity model, adding air.
        # aniso=aniso,                # Here you could implement anisotropy...
        #                             # ...or any parameter accepted by bipole.
        freqtime = system.modellingFrequencies,                # Required frequencies.
        mrec = True,                    # It is an el. source, but a magn. rec.
        strength = 8,                   # To account for 4 sides of square loop.
        srcpts = 5,                     # Approx. the finite dip. with 3 points.
        htarg = {'fhtfilt': 'key_101_2009'},
        verb=0,# Short filter, so fast.
    )

    # Multiply the frequecny-domain result with
    # \mu for H->B, and i\omega for B->dB/dt.
    EM *= 2j * np.pi * system.modellingFrequencies * 4e-7 * np.pi

    # Apply filters the data for the given system
    for filt in system.offTimeFilters:
        EM *= filt.frequencyResponse(system.modellingFrequencies)


    # === CONVERT TO TIME DOMAIN ===
    EM, _ = np.squeeze(tem(EM[:, None],
                       np.array([1]),
                       system.modellingFrequencies,
                       system.modellingTimes,
                       -1,
                       system.ft,
                       system.ftarg))

    # === APPLY WAVEFORM ===
    return waveform(system.modellingTimes, EM, system.times, system.waveform.time-system.delayTime, system.waveform.amplitude)


def waveform(times, resp, times_wanted, wave_time, wave_amp, nquad=3):
    """Apply a source waveform to the signal.

    Parameters
    ----------
    times : ndarray
        Times of calculated input response; should start before and
        end after `times_wanted`.

    resp : ndarray
        EM-response corresponding to `times`.

    times_wanted : ndarray
        Wanted times.

    wave_time : ndarray
        Time steps of the wave.

    wave_amp : ndarray
        Amplitudes of the wave corresponding to `wave_time`, usually
        in the range of [0, 1].

    nquad : int
        Number of Gauss-Legendre points for the integration. Default is 3.

    Returns
    -------
    resp_wanted : ndarray
        EM field for `times_wanted`.

    """

    # Interpolate on log.
    PP = iuSpline(np.log10(times), resp)

    # Wave time steps.
    dt = np.diff(wave_time)
    dI = np.diff(wave_amp)
    dIdt = dI / dt

    # Gauss-Legendre Quadrature; 3 is generally good enough.
    g_x, g_w = _cached_roots_legendre(nquad)

    # Pre-allocate output.
    resp_wanted = np.zeros_like(times_wanted)

    # Loop over wave segments.
    for i, cdIdt in enumerate(dIdt):

        # We only have to consider segments with a change of current.
        if cdIdt == 0.0:
            continue

        # If wanted time is before a wave element, ignore it.
        ind_a = wave_time[i] < times_wanted
        if ind_a.sum() == 0:
            continue

        # If wanted time is within a wave element, we cut the element.
        ind_b = wave_time[i+1] > times_wanted[ind_a]

        # Start and end for this wave-segment for all times.
        ta = times_wanted[ind_a]-wave_time[i]
        tb = times_wanted[ind_a]-wave_time[i+1]
        tb[ind_b] = 0.0  # Cut elements

        # Gauss-Legendre for this wave segment. See
        # https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval
        # for the change of interval, which makes this a bit more complex.
        logt = np.log10(np.outer(0.5*(tb - ta), g_x) + 0.5*(ta + tb)[:, None])
        fact = 0.5*(tb - ta) * cdIdt
        resp_wanted[ind_a] += fact * np.sum(np.array(PP(logt) * g_w), axis=1)

    return resp_wanted