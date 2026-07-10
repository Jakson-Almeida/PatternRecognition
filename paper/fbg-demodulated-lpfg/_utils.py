import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import logging
from copy import copy
logger = logging.getLogger(__name__)


def find_wlres(wl, T, lims=[1.5e-6, 1.6e-6], dwl=3e-9, prominence=0.5):
    """
    Resonant wavelength estimation

    Find the spectrum dip and fit it to a loretzian

    Parameters
    ----------
    wl: np.array
        Wavelength

    T: np.array
        Spectrum
    
    dwl: float
        Wavelength range to fit loretzian

    lims: list
        Resonant wavelength bounds

    prominence: float
        Resonant dip prominence

    Returns
    -------
    wlres: float
        Resonant wavelength
    """
    info = {}

    wl_range = [min(lims)-10e-9, max(lims)+10e-9]
    resolution = np.mean(np.diff(wl))

    mask = ( wl > min(wl_range) ) & (  wl < max(wl_range))
    wl = wl[mask]
    T = T[mask]
    dwl = 3e-9
    resolution_proximity = 3

    peaks, peak_info = find_peaks(-T, prominence=prominence, 
                                  plateau_size=0, wlen=None)
    info = {}
    
    for i in range(len(peaks)):
        wl0 = wl[peaks[i]]
        mask = (wl> wl0 - dwl/2) & (wl < wl0 + dwl/2)

        try:
            popt, _ = curve_fit(transmission_spectra, wl[mask], T[mask],
                                p0=None, max_nfev=10000,
                                bounds=((-np.inf, wl0-resolution_proximity*resolution, 1e-10, -np.inf),
                                        (+np.inf, wl0+resolution_proximity*resolution, 100, np.inf)))

            resonant_wl = popt[1]
            resonant_power = transmission_spectra(popt[1], *popt)

        except RuntimeError:
            resonant_wl = wl[peaks[i]]
            resonant_power = T[peaks[i]]

        if len(peaks) == 1:
            info['resonant_wl'] = resonant_wl
            info['resonant_wl_power'] = resonant_power
        else:
            info[f'resonant_wl_{i}'] = resonant_wl
            info[f'resonant_wl_power_{i}'] = resonant_power

    best_index = np.argmax(peak_info['prominences'])
    info['best_index'] = best_index
    
    try:
        wlres = 1e9*info['resonant_wl']
    except KeyError:
        best = info['best_index']
        wlres = 1e9*info[f'resonant_wl_{best}']
    return wlres


def transmission_spectra(x, a, x0, w, bias):
    """
    Approximates a LPFG spectrum by a loretzian

    Parameters
    ----------
    x: np.array
        Wavelength for simulation

    a: float
        Attenuation intensity

    x0: float
        Resonant wavelength

    w: float
        FWHM

    bias: float
        Insertion loss

    Returns
    -------
    spectrum: np.array
        LPFG array

    """
    return -a*(1 + ((x - x0)/(w/(2*abs(a/3 - 1)**0.5)))**2)**(-1) - bias


def lin_interp(x, y, i, half):
    """
    Linear interpolation
    """
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))


def fwhm(x, y):
    """
    Estimate FWHM

    Parameters
    ----------
    x: np.array
        x-var

    y: np.array
        y-var

    Returns
    -------
    fwhm: float
    """
    half = max(y) - 3.010299956639812
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    crossings = [lin_interp(x, y, zero_crossings_i[0], half),
                 lin_interp(x, y, zero_crossings_i[1], half)]
    return max(crossings) - min(crossings)


def lorentz(x, a, x0, w, b):
    """
    Loretzian function
    """
    return a*(1 + ((x - x0)/(w/2))**2)**(-1) + b


def gaussian(x, a, x0, s, b=0):
    """
    Gaussian function
    """
    arg = -(x - x0)**2 / (2*s**2)
    return a * np.exp(arg) + b


def fbg_reflection(wl_bragg, fwhm, wl, unit='dB'):
    """
    FBG simulation
    Reference on equation:
    @article{peternella2017,
      title={Interrogation of a ring-resonator ultrasound sensor using a fiber Mach-Zehnder interferometer},
      author={Peternella, Fellipe Grillo and Ouyang, Boling and Horsten, Roland and Haverdings, Michael and Kat, Pim and Caro, Jacob},
      journal={Optics express},
      volume={25},
      number={25},
      pages={31622--31639},
      year={2017},
      publisher={Optical Society of America}
    }

    Parameters
    ----------
    wl_bragg: float
        Bragg wavelength
    fwhm: float
        FWHM
    wl: np.array
        Simulation wavelengths
    unit: string
        Unit

    Returns
    -------
    reflection_tf: np.array
        Reflection transfer function
    """
    R = (1 + ((wl - wl_bragg) / (fwhm / 2)) ** 8) ** (-1)
    if unit == 'dB':
        return 10*np.log10(R)
    elif unit == 'linear' or unit == 'lin':
        return R
    else:
        print('Invalid unit')
        return -1


def find_bragg(wl, sp, dwl=6, prominence=9):
    """
    Find Bragg wavelengths

    Parameters
    ----------
    wl: np.array
        Wavelength

    sp: np.array
        Spectrum

    dwl: float
        Wavelength range

    Returns
    -------
    wl_bragg: np.array
        Bragg wavelengths
    peaks: np.array
        Bragg wavelengths' intensities
    """
    loc, info = find_peaks(sp, prominence=prominence)
    wl_bragg = []
    peaks = []
    for b_i in loc:
        mask = (wl > wl[b_i]-dwl/2) & (wl < wl[b_i]+dwl/2)
        pars, cov = curve_fit(gaussian, wl[mask], sp[mask], 
                                p0=(-sp[b_i]+min(sp), wl[b_i], 1, -min(sp)), 
                                bounds=((-np.inf, wl[b_i]-dwl, 1e-17, -np.inf), 
                                        (np.inf, wl[b_i]+dwl, 1e+02, np.inf)))
        wl_bragg.append(pars[1])
        peaks.append(gaussian(pars[1], *pars))
    return np.array(wl_bragg), np.array(peaks)


def my_gauss(x, a, x0, w, bias):
    """
    Custom modified gaussian function for LPFG spectrum simulation

    Parameters
    ----------
    x: np.array
        Wavelength for simulation

    a: float
        Attenuation intensity

    x0: float
        Resonant wavelength

    w: float
        FWHM

    bias: float
        Insertion loss

    Returns
    -------
    spectrum: np.array
        LPFG array

    """
    s = 2*(abs(4*np.log(a/3.01)))**0.5
    s = w/s
    arg = -(x - x0)**2 / ((2*s)**2)
    return -a * np.exp(arg) - bias


def arbitrary_funcs(x, a, x0, w, bias, fcn):
    """
    Function to generate synthetic LPFG spectrum using a combination of functions

    Parameters
    ----------
    x: np.array
        Wavelength for simulation

    a: float
        Attenuation intensity

    x0: float
        Resonant wavelength

    w: float
        FWHM

    bias: float
        Insertion loss

    fcn: float
        Function selection parameter

    Returns
    -------
    spectrum: np.array
        LPFG array

    """
    if fcn < 0.2:
        y = transmission_spectra(x, a, x0, w, bias)
    elif fcn < 0.4:
        y = my_gauss(x, a, x0, w, bias)
    elif fcn < 0.6:
        y = 0.5*transmission_spectra(x, a, x0, w, bias) + 0.5*my_gauss(x, a, x0, w, bias)
    elif fcn < 0.8:
        y = -a*transmission_spectra(x, 1, x0, w, 0)*my_gauss(x, 1, x0, w, 0) - bias
    else:
        k = np.random.rand()
        y = k*transmission_spectra(x, a, x0, w, bias) + (1-k)*my_gauss(x, a, x0, w, bias)
    return y


def mapper(x, min_x, max_x, min_y, max_y):
    """
    Map values

    Parameters
    ----------
    x: float
        Input value

    min_x: float
        Minimum value of x

    max_x: float
        Maximum value of x

    min_y: float
        Minimum value of y

    max_y: float
        Maximum value of y

    Returns
    -------
    mapped_value: float
        Mapped value
    """
    dx = max_x-min_x
    dy = max_y-min_y
    return min_y + (dy/dx)*(x-min_x)


def noisy_arbitrary_funcs(x, a, x0, w, bias, fcn):
    """
    Generate a noisy LPFG transmission spectrum

    Parameters
    ----------
    x: np.array
        Wavelength for simulation

    a: float
        Attenuation intensity

    x0: float
        Resonant wavelength

    w: float
        FWHM

    bias: float
        Insertion loss

    fcn: float
        Function selection parameter

    Returns
    -------
    noisy_spectrum: np.array
        Noisy LPFG array

    clean_spectrum: np.array
        Clean LPFG array
    """
    y = arbitrary_funcs(x, copy(a), copy(x0), copy(w), copy(bias), copy(fcn))
    y_clean = copy(y)
    k = mapper(np.random.rand(), 0, 1, 1, 3)
    k = int(np.round(k))
    for i in range(k):
        if k==0:
            n_x0 = mapper(np.random.rand(), 0, 1, 1490e-9, 1610e-09)
            n_a = mapper(np.random.rand(), 0, 1, copy(a)/10, copy(a)/6)
            n_w = mapper(np.random.rand(), 0, 1, 60e-9, 100e-9)
        else:
            n_a = mapper(np.random.rand(), 0, 1, copy(a)/10, copy(a)/4)
            n_w = mapper(np.random.rand(), 0, 1, copy(w)*1.5, 70e-9)
            if np.random.rand() > 0.5:
                if copy(x0)-copy(w)/2 < 1510e-9:
                    n_x0 = mapper(np.random.rand(), 0, 1, copy(x0)+copy(w), 1590e-09)
                else:
                    n_x0 = mapper(np.random.rand(), 0, 1, 1510e-9, copy(x0)-copy(w)/2)
            else:
                if copy(x0)+copy(w)/2 > 1590e-09:
                    n_x0 = mapper(np.random.rand(), 0, 1, 1510e-9, copy(x0)-copy(w))
                else:
                    n_x0 = mapper(np.random.rand(), 0, 1, copy(x0)+copy(w)/2, 1590e-09)
        noisy_peak = arbitrary_funcs(x, n_a, n_x0, n_w, 0, np.random.rand())
        y = y + noisy_peak
        
        if np.random.rand() > 0.6:
            if np.random.rand() > 0.5:
                n_x0 = mapper(np.random.rand(), 0, 1, 1450e-9, 1500e-09)
            else:
                n_x0 = mapper(np.random.rand(), 0, 1, 1600e-9, 1650e-09)
            n_a = mapper(np.random.rand(), 0, 1, copy(a)*0.6, copy(a)*2)
            n_w = mapper(np.random.rand(), 0, 1, copy(w)*0.6, copy(w)*2)
            y = y + arbitrary_funcs(x, n_a, n_x0, n_w, 0, np.random.rand())
    return y, y_clean