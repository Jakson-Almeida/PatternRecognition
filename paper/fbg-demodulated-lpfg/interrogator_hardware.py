# -*- coding: utf-8 -*-
"""
@author: felip
"""

import numpy as np
from scipy import interpolate
from copy import copy
import logging
logger = logging.getLogger(__name__)


class InterrogatorHardware:
    """
    Simulate the LPFG interrogator hardware

        FBG is simulated using numerical approx,
        power meter by numerical integration, and
        devices concatenation by sum of the dB transfer
        function
    """
    def __init__(self, n_fbg=13, fwhm=100e-12, detection_range=None, fbg_array=None):
        """
        InterrogatorHardware constructor

        Parameters
        ----------
        n_fbg: int, optional
            Number of FBGs - equally spaced
        fwhm: float
            FBG FWHM in m
        detection_range: list
            min and max Bragg wavelengths
        fbg_array: np.array
            FBG positions - for custom setup
            overwrites n_fbga and detection_range
        """
        if detection_range is None:
            detection_range = [1512e-9, 1584e-9]
       
        if fbg_array is None:
            self.wl_bragg_array = np.linspace(detection_range[0], detection_range[1], n_fbg)
            self.detection_range = np.array(detection_range)
        else:
            self.wl_bragg_array = fbg_array
            self.detection_range = np.array([min(fbg_array), 
                                             max(fbg_array)])
            
        self.FWHM = np.array([fwhm]*len(self.wl_bragg_array))
        

        self.sim_wl = np.arange(detection_range[0] - np.diff(self.wl_bragg_array)[0]/2,
                                detection_range[1] + np.diff(self.wl_bragg_array)[0]/2,
                                fwhm/2)
        self.fbg_array = self.filter_bank()

        logger.debug(f'Interrogator constructed to detect an LPFG resonant' +
                     f' wavelength from {1e9*detection_range[0]} nm to {1e9*detection_range[1]} nm' +
                     f' with {n_fbg} equally spaced FBGs (FWHM: {1e12*fwhm} pm)\n' +
                     f' Bragg wavelengths: {np.round(1e9*self.wl_bragg_array, decimals=2)} nm')

    def set_custom_array(self, wl_bragg, fwhm):
        """
        Define a generic configuration for the FBGs

        Parameters
        ----------
        wl_bragg: list
            each fbg Bragg wavelength as a list (in nm)
        fwhm: list
            fwhm: each fbg fwhm as a list (in nm)

        Raises
        ----------
        ValueError
            If len(wl_bragg) != len(fwhm)
            If wavelengths not in nm

        """
        if len(wl_bragg) != len(fwhm):
            raise ValueError("Arrays describing the FBGs parameters Bragg wavelengths and FWHM must have the same length")
        if min(wl_bragg) < 100:
            raise ValueError("wl_bragg should be in nm")
        if min(fwhm) < 1e-6:
            raise ValueError("fwhm should be in nm")

        self.wl_bragg_array = np.array(wl_bragg)*1e-9
        self.FWHM = np.array(fwhm)*1e-9

        logger.info(f'Interrogator configured to detect an LPFG resonant' +
                    f' wavelength from {min(self.wl_bragg_array)} nm to {max(self.wl_bragg_array)} nm' +
                    f' using FBGs @ {1e9*self.wl_bragg_array} nm with {1e9*self.FWHM} nm FWHM')

    def get_fbg_position(self):
        """
        Returns the bragg wavelength of each fbg

        Returns
        ----------
        fbg_positios: np.array
        """
        return copy(self.wl_bragg_array)

    def get_detection_range(self):
        """
        Returns detection range

        Returns
        -------
        detection_range: np.array
        """
        return copy(self.detection_range)

    def get_filtered_power(self, wl_input, spec_input):
        """
        Returns the power filtered by the fbg array

        Parameters
        ----------
        wl_input: np.array
            Simulation wavlenegth array
        spec_input: np.array
            Simulation spectrum array

        Returns
        -------
        power_array: np.array
            FBG array filtered power
        """
        power_array = list()
        for fbg in self.fbg_array:
            interp = interpolate.interp1d(wl_input, spec_input, kind='cubic')
            filtered_lpg = interp(self.sim_wl) + fbg
            power_array.append(self.power_meter(wl=self.sim_wl, spec=filtered_lpg, unit='dBm'))
        return np.array(power_array)

    def filter_bank(self):
        """
        FBG reflection filter array

        Returns
        -------
        filter_bank: np.array
            Each fbg reflection

        """
        filters = list()
        for i in range(0, len(self.wl_bragg_array)):
            fbg_r = self.fbg_reflection(self.wl_bragg_array[i], self.FWHM[i], self.sim_wl)
            filters.append(fbg_r)
        return np.array(filters)
    
    @staticmethod
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
    
    @staticmethod
    def power_meter(wl=0, spec=0, unit='dBm', osaRes=0.5e-9):
        """
        Simulate a power meter by trapezoidal integration

        Parameters
        ----------
        wl: np.array
            Simulation wavelengths
        spec: np.array
            Simulation optical power density
        unit: string
            Unit
        osaRes: float
            Spectrum resolution

        Returns
        -------
        optical_power: float
            Optical power meter simulated reading
        """
        psdW = 1e-3 * 10**(spec/10)
        pw = np.abs(np.trapz(psdW, x=wl))/osaRes
        if unit == 'dBm':
            return 10*np.log10(pw*1e3)
        elif unit == 'dBW':
            return 10*np.log10(pw)
        elif unit == 'mW':
            return pw*1e3
        elif unit == 'W':
            return pw
        else:
            print('Invalid unit')
            return -1
