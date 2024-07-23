# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# KIDs Lab. Homodyne system
# fit_psd.py
# Set of functions to fit the PSD
#
# Marcial Becerril, @ 22 July 2024
# Latest Revision: 22 Jul 2024, 11:10 GMT-6
#
# TODO list:
# + Improve the fit
# + Apply log-binning
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# mbecerrilt@inaoep.mx
#
# --------------------------------------------------------------------------------- #


import sys

import numpy as np

import scipy.signal
from scipy.signal import butter, lfilter
from scipy.optimize import curve_fit

from matplotlib.pyplot import *

sys.path.append('../')
from misc.misc_funcs import *


class fit_psd(object):
    """
    PSD fitting object.
    Handle the PSD fitting
    Parameters
    ----------
    plot_name(opt) : string
        Plot name.
    ----------
    """
    def __init__(self, *args, **kwargs):
        # Key arguments
        # ----------------------------------------------
        # Linear binning size
        self.lin_bin_size = kwargs.pop('lin_bin_size', 5)
        # Lower limit amplifier noise
        self.freq_amp = kwargs.pop('freq_amp', 7.5e4)
        # Project name
        self.plot_name = kwargs.pop('plot_name', '')
        # ----------------------------------------------

    def lin_binning(self, freq_psd, psd, w=10):
        """
        Linear binning PSD.
        Parameters
        -----------
        freq_psd : array
            Frequency [Hz].
        psd : array
            Power Spectral Density [Hz²/Hz].
        w : int
            Size binning.
        -----------
        """
        n_psd = []
        n_freq = []
        psd_accum = 0
        freq_accum = 0
        for i, p in enumerate(psd):
            if i%w == 0 and i != 0:
                n_psd.append(psd_accum/w)
                n_freq.append(freq_accum/w)
                psd_accum = 0
                freq_accum = 0
            psd_accum += p
            freq_accum += freq_psd[i]
        
        n_freq = np.array(n_freq)
        n_psd = np.array(n_psd)

        return n_freq, n_psd
    
    def log_binning(self, freq_psd, psd, n_pts=500):
        """
        Logarithmic binning for PSD.
        Parameters
        -----------
        freq_psd : array
            Frequency [Hz].
        psd : array
            Power Spectral Density [Hz²/Hz].
        n_pts : int
            Number of points.
        -----------
        """

        start = freq_psd[0]
        stop = freq_psd[-1]
        """
        n = 0
        central_pts = []
        for i in range(n_pts+1):
            if int(n**2) > len(freq_psd):
                break
            else:
                central_pts.append(freq_psd[int(n**2)])
            n += 1
        central_pts = np.array(central_pts)
        """
        """
        if n_pts > len(central_pts):
            tot_pts = len(central_pts)
        else:
            tot_pts = n_pts
        """
        central_pts = np.logspace(np.log10(start), np.log10(stop), n_pts+1)
       
        n_freq = []
        n_psd = []
        for i in range(n_pts):
            idx_start = np.where(freq_psd > central_pts[i])[0][0]
            idx_stop = np.where(freq_psd <= central_pts[i+1])[0][-1] + 1

            if not np.isnan(np.mean(freq_psd[idx_start:idx_stop])):
                n_freq.append(np.mean(freq_psd[idx_start:idx_stop]))
                n_psd.append(np.median(psd[idx_start:idx_stop]))

        n_freq = np.array(n_freq)
        n_psd = np.array(n_psd)

        return n_freq, n_psd


    def get_psd_fit(self, freq_psd, psd, f0_fits, Q, amp_noise):
        """
        Get PSD fit
        Parameters
        -----------
        freq_psd : array
            Frequency [Hz].
        psd : array
            Power Spectral Density [Hz²/Hz].
        f0_fits : float
            Resonance frequency [Hz].
        Q : float
            Total quality factor.
        amp_noise : float
            Amplifier noise.
            Zero if PSD = PSD_ON - PSD_OFF.
        -----------
        """

        # Apply linear binning
        #n_freq, n_psd = self.lin_binning(freq_psd, psd, w=self.lin_bin_size)
        n_freq, n_psd = self.log_binning(freq_psd, psd)

        # Fit PSD curve
        gr_noise_s, amp_noise_s, tau_s, tls_a_s, tls_b_s, fit_PSD_s = self.fit_spectra_noise(freq_psd, n_freq, 
                n_psd, f0_fits, Q, amp_noise)

        return gr_noise_s, tau_s, amp_noise_s, tls_a_s, tls_b_s, fit_PSD_s
    
    def guess_params(self, freq, psd, gr_lims=[20, 50]):
        """
        Guess parameters to fit PSD.
        Parameters
        -----------
        freq_psd : array
            Frequency [Hz].
        psd : array
            Power Spectral Density [Hz²/Hz].
        gr_lims : list
            Generation-Recombination noise [Hz²/Hz].
        -----------
        """

        # G E N E R A T I O N - R E C O M B I N A T I O N   N O I S E
        # -------------------------------------
        gr_low = gr_lims[0]
        gr_up = gr_lims[1]

        idx_from = np.where(freq > gr_low)[0][0]
        idx_to = np.where(freq < gr_up)[0][-1]
        gr_guess = np.median(psd[idx_from:idx_to])

        # Define bounds
        gr_min = np.min(psd[idx_from:idx_to])
        gr_max = np.max(psd[idx_from:idx_to])

        # Q U A S I P A R T I C L E   L I F E T I M E
        # -------------------------------------
        # Quasipartile lifetime
        tauqp_guess = 1./(np.max(freq)-np.min(freq))

        # Define bounds
        tauqp_min = 0
        tauqp_max = 1./np.min(freq)

        # T L S   N O I S E 
        # -------------------------------------
        idx_one = np.where(freq > 1)[0][0]
        tlsa_guess = psd[idx_one]
        tlsb_guess = -0.5

        tlsa_min = gr_min
        tlsa_max = np.max(psd)

        tlsb_min = -2.0
        tlsb_max = 0.0

        guess = np.array([gr_guess, tauqp_guess, tlsa_guess, tlsb_guess])

        bounds = np.array([ [gr_min, tauqp_min, tlsa_min,  tlsb_min ],
                            [gr_max, tauqp_max, tlsa_max,  tlsb_max ]])

        return guess, bounds

    def fit_spectra_noise(self, ofreqs, bfreqs, psd, f0_fits, Q, amp_noise):
        """
        Perform the fitting on the PSD.
        Parameters
        -----------
        ofreqs : array
            Frequency original array [Hz].
        bfreqs : array
            Post-binned frequency array [Hz].
        psd : array
            PSD array [Hz²/Hz].
        -----------
        """

        guess, bounds = self.guess_params(bfreqs, psd)

        sigma = (1 / abs(np.gradient(bfreqs)))

        popt, pcov = curve_fit(lambda freqs, gr_level, tau_qp, tls_a, tls_b: spectra_noise_model(freqs,
                    gr_level, tau_qp, tls_a, tls_b, Q, f0_fits, amp_noise), bfreqs, psd, bounds=bounds, p0=guess, sigma=sigma)
        (gr_noise, tau_qp, tls_a, tls_b) = popt

        fit_PSD = spectra_noise_model(ofreqs, gr_noise, tau_qp, tls_a, tls_b, Q, f0_fits, amp_noise)

        return gr_noise, amp_noise, tau_qp, tls_a, tls_b, fit_PSD


    # P L O T   F U N C T I O N S 
    # ---------------------------
    def update_plot(self, freq_psd, psd, psd_fit):
        """
        Update plot
        Parameters
        ----------
        freq_psd : array
            Frequency [Hz].
        psd : array
            Power Spectral Density [Hz²/Hz].
        psd_fit : array
            Fitted PSD.
        ----------
        """

        # Plot raw PSD
        self._ax.semilogx(freq_psd, 10*np.log10(psd), 'c')
        # Plot binned data
        n_freq, n_psd = self.log_binning(freq_psd, psd)
        self._ax.semilogx(n_freq, 10*np.log10(n_psd), 'r', lw=1)
        # Plot fitted data
        self._ax.semilogx(freq_psd, 10*np.log10(psd_fit), 'k')

        self._ax.set_title(self.plot_name)
        self._ax.set_xlabel(r'Frequency [Hz]', fontsize=18, weight='bold')
        self._ax.set_ylabel(r'PSD [Hz$^2$/Hz] ', fontsize=18, weight='bold')

    def apply_psd_fit(self, freq_psd, psd, f0_fits, Q, amp_noise, inter=False):
        """
        Apply PSD fit.
        Parameters
        -----------
        freq_psd : array
            Frequency [Hz].
        psd : array
            Power Spectral Density [Hz²/Hz].
        f0_fits : float
            Resonance frequency [Hz].
        Q : float
            Total quality factor.
        amp_noise : float
            Amplifier noise.
            Zero if PSD = PSD_ON - PSD_OFF.
        inter : bool
            Activate interactive mode.
        -----------
        """

        # I N I T I A L   F I T
        # ----------------------
        # Global parameters
        self.Q = Q
        self.f0_fits = f0_fits
        self.amp_noise = amp_noise
        # Data
        self.freq_psd = freq_psd
        self.psd = psd
        # Apply fit
        gr_noise_s, tau_s, amp_noise_s, tls_a_s, tls_b_s, fit_PSD_s = self.get_psd_fit(freq_psd, psd, f0_fits, Q, amp_noise)

        # F I T   R E S U L T S
        # ----------------------
        self.fit_psd = fit_PSD_s
        # Update parameters
        self.gr_noise = gr_noise_s
        self.tau = tau_s
        self.amp_noise = amp_noise_s
        self.tls_a = tls_a_s
        self.tls_b = tls_b_s

        # Interactive mode
        if inter:
            self.interactive_mode(freq_psd, psd, fit_PSD_s)

    def interactive_mode(self, freq_psd, psd, fit_PSD_s):
        """
        Interactive mode to clean psd data.
        Parameters
        ----------
        freq_psd : array
            Frequency [Hz].
        psd : array
            Power Spectral Density [Hz²/Hz].
        psd_fit : array
            Fitted PSD.
        ----------
        """

        # Create figures
        self._fig = figure()
        self._ax = self._fig.add_subplot(111)

        self._freq_psd = freq_psd
        self._psd = psd
        self._fit_psd = fit_PSD_s

        self._cnt = 0
        self._idx = 0
        self.x_range = np.zeros(2, dtype=int)

        self.update_plot(self._freq_psd, self._psd, self._fit_psd)

        self._onclick_xy = self._fig.canvas.mpl_connect('button_press_event', self._onclick_ipsd)
        self._keyboard = self._fig.canvas.mpl_connect('key_press_event', self._key_pressed_ipsd)

        show()

    def _key_pressed_ipsd(self, event):
        """
        Keyboard event to save/discard line fitting changes.
        Paramters
        ---------
        event : event
            Key pressed event.
        ---------
        """

        sys.stdout.flush()

        if event.key in ['x', 'q', 'd', 'w']:

            # Save changes and close interactive mode.
            if event.key == 'x':
                print('Changes saved')
                self._fig.canvas.mpl_disconnect(self._onclick_ipsd)
                self._fig.canvas.mpl_disconnect(self._key_pressed_ipsd)
                close(self._fig)
                try:
                    # S A V E   P A R A M S
                    # --------------------------
                    self.fit_psd = self._fit_psd

                    self.gr_noise = self._gr_noise
                    self.tau = self._tau
                    self.amp_noise = self._amp_noise
                    self.tls_a = self._tls_a
                    self.tls_b = self._tls_b
                except:
                    print('No changes')

            # Discard changes and close interactive mode
            elif event.key == 'q':
                print('No changes to the fitting')
                self._fig.canvas.mpl_disconnect(self._onclick_ipsd)
                self._fig.canvas.mpl_disconnect(self._key_pressed_ipsd)
                close(self._fig)

            # Apply fit
            elif event.key == 'w':
                x_sort = np.sort(self.x_range)
                _freq_psd = np.concatenate((self._freq_psd[:x_sort[0]],self._freq_psd[x_sort[1]:]))
                # Get amplifier noise
                self.amp_noise = np.mean(self._psd[np.where(_freq_psd>self.freq_amp)[0]])
                if not np.isnan(self.amp_noise):

                    # R E P E A T   F I T
                    # --------------------------
                    # Redefine freq and psd
                    self._freq_psd = _freq_psd
                    self._psd = np.concatenate((self._psd[:x_sort[0]],self._psd[x_sort[1]:]))
                    # Fit new curve
                    gr_noise_s, tau_s, amp_noise_s, tls_a_s, tls_b_s, fit_PSD_s = self.get_psd_fit(self._freq_psd, \
                                                                self._psd, self.f0_fits, self.Q, self.amp_noise)

                    # S A V E   P A R A M S
                    # --------------------------
                    self._fit_psd = fit_PSD_s
                    
                    self._gr_noise = gr_noise_s
                    self._tau = tau_s
                    self._amp_noise = amp_noise_s
                    self._tls_a = tls_a_s
                    self._tls_b = tls_b_s
                    
                    # U P D A T E   P L O T
                    # --------------------------
                    cla()
                    self.update_plot(self._freq_psd, self._psd, fit_PSD_s)
                    self._fig.canvas.draw_idle()
                    print('Fringe removed')

                else:
                    print('Delete all data > ' + str(self.freq_amp) + 'Hz is not allowed')

            # Discard changes and load initial values
            elif event.key == 'd':
                # Reset values
                self._cnt = 0
                self._idx = 0
                self.x_range = np.zeros(2, dtype=int)

                # U P D A T E   P L O T
                # --------------------------
                cla()
                self.update_plot(self.freq_psd, self.psd, self.fit_psd)
                self._freq_psd = self.freq_psd
                self._psd = self.psd
                self._fig.canvas.draw_idle()
                print('Clear filters')

    def _onclick_ipsd(self, event):
        """
        On-click event to select lines
        Paramters
        ---------
        event : event
            Click event.
        ---------
        """

        if event.inaxes == self._ax:

            # Left-click select the data regions to discard.
            if event.button == 3:
                ix, iy = event.xdata, event.ydata
                
                # Handle data out of range
                if ix>self._freq_psd[-1]:
                    xarg = len(self._freq_psd)
                else:
                    xarg = np.where(self._freq_psd>ix)[0][0]
                
                self._ax.axvline(ix)

                self._cnt += 1
                self._idx = self._cnt%2
                self.x_range[self._idx] = xarg

                self._fig.canvas.draw_idle()