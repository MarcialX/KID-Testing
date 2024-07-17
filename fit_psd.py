# -*- coding: utf-8 -*-
"""
===================================================================
Fit power spectral density
===================================================================
"""

import numpy as np
# Scipy
import scipy.signal
from scipy.signal import butter, lfilter
from scipy.optimize import curve_fit
# Matplotlib
from matplotlib.pyplot import *
# Miscellaneous libraries
#sys.path.append("./misc/")
#from print_msg import *
#from main_cts import *

# FIT Spectral noise
class fit_spectral_noise():
    """
    Get the model which match the PSD noise.
    A hundred percent based in PhD Sam Rowe code
    """
    def fit_kid_psd(self, ori_freqs, psd_freqs, psd_df, kid_f0, kid_qr, amp_noise,
                    gr_guess=None, tauqp_guess=None, tlsa_guess=None,  tlsb_guess=-0.5,
                    gr_min=0,      tauqp_min=0,      tlsa_min=-np.inf, tlsb_min=-2.0,
                    gr_max=np.inf, tauqp_max=np.inf, tlsa_max=np.inf,  tlsb_max=-0.01,
                    sigma = None):

        def combined_model(freqs, gr_noise, tau_qp, tls_a, tls_b):
            # Ruido Generación-Recombinación
            gr = gr_noise/(1.+(2*np.pi*freqs*tau_qp)**2) / (1.+(2*np.pi*freqs*kid_qr/np.pi/kid_f0)**2)
            #gr = np.log10(gr)
            # Ruido TLS
            tls = tls_a*freqs**(tls_b) / (1.+(2*np.pi*freqs*kid_qr/np.pi/kid_f0)**2)
            #tls = np.log10(tls)
            # Ruido del amplificador
            amp = amp_noise
            #amp = np.log10(amp)
            # Ruido Total
            return gr + tls + amp

        if gr_guess is None:
            idx_from = np.where(psd_freqs>20)[0][0]
            idx_to = np.where(psd_freqs<50)[0][-1]
            gr_guess = np.median(psd_df[idx_from:idx_to])
        if tauqp_guess is None:
            tauqp_guess = 1./(psd_freqs.max()-psd_freqs.min())
        if tlsa_guess is None:
            idx_one = np.where(psd_freqs>1)[0][0]
            tlsa_guess = psd_df[idx_one]
        if tlsb_guess is None:
            tlsb_guess = -0.5

        guess = np.array([gr_guess, tauqp_guess, tlsa_guess,  tlsb_guess ])
        bounds = np.array([ [gr_min, tauqp_min, tlsa_min,  tlsb_min ],
                            [gr_max, tauqp_max, tlsa_max,  tlsb_max ]])

        if sigma is None:
            sigma = (1 / abs(np.gradient(psd_freqs)))

        pval, pcov = curve_fit(combined_model, psd_freqs, psd_df, p0=guess, bounds=bounds, sigma=sigma)
        (gr_noise,tau_qp,tls_a,tls_b) = pval

        fit_PSD = combined_model(ori_freqs, gr_noise, tau_qp, tls_a, tls_b)
        print(tls_a, tls_b)

        return gr_noise, tau_qp, amp_noise, tls_a, tls_b, fit_PSD

    def fit_kid_psd_log(self, psd_freqs, psd_df, kid_f0, kid_qr, amp_noise,
                    gr_guess=None, tauqp_guess=None, tlsa_guess=None,  tlsb_guess=-0.5,
                    gr_min=0,      tauqp_min=0,      tlsa_min=-np.inf, tlsb_min=-2.0,
                    gr_max=np.inf, tauqp_max=np.inf, tlsa_max=np.inf,  tlsb_max=-0.01,
                    sigma = None):

        def combined_model_log(freqs, gr_noise, tau_qp, tls_a, tls_b):
            # Ruido Generación-Recombinación
            gr = gr_noise/(1.+(2*np.pi*freqs*tau_qp)**2) / (1.+(2*np.pi*freqs*kid_qr/np.pi/kid_f0)**2)
            #gr = np.log10(gr)
            # Ruido TLS
            tls = tls_a*freqs**tls_b / (1.+(2*np.pi*freqs*kid_qr/np.pi/kid_f0)**2)
            #tls = np.log10(tls)
            # Ruido del amplificador
            amp = amp_noise
            #amp = np.log10(amp)
            # Ruido Total
            return np.log10(gr + tls + amp)
        
        if gr_guess is None:
            idx_from = np.where(psd_freqs>20)[0][0]
            idx_to = np.where(psd_freqs<50)[0][-1]
            gr_guess = np.median(psd_df[idx_from:idx_to])
        if tauqp_guess is None:
            tauqp_guess = 1./(psd_freqs.max()-psd_freqs.min())
        if tlsa_guess is None:
            idx_one = np.where(psd_freqs>1)[0][0]
            tlsa_guess = psd_df[idx_one]
        if tlsb_guess is None:
            tlsb_guess = -0.5

        guess = np.array([gr_guess, tauqp_guess, tlsa_guess,  tlsb_guess ])
        bounds = np.array([ [gr_min, tauqp_min, tlsa_min,  tlsb_min ],
                            [gr_max, tauqp_max, tlsa_max,  tlsb_max ]])

        if sigma is None:
            sigma = (1 / abs(np.gradient(psd_freqs)))

        psd_df = np.log10(psd_df)

        pval, pcov = curve_fit(combined_model_log, psd_freqs, psd_df, p0=guess, bounds=bounds, sigma=sigma)
        (gr_noise,tau_qp,tls_a,tls_b) = pval

        fit_PSD = combined_model_log(psd_freqs,gr_noise,tau_qp,tls_a,tls_b)

        return gr_noise,tau_qp,amp_noise,tls_a,tls_b,fit_PSD

    def fit_kid_psd_q(self,psd_freqs, psd_df, kid_f0,amp_noise,
                    gr_guess=None, tauqp_guess=None, tlsa_guess=None,  tlsb_guess=-1.5, Q_r=5000,
                    gr_min=0,      tauqp_min=0,      tlsa_min=-np.inf, tlsb_min=-1.501, Q_min=0,
                    gr_max=np.inf, tauqp_max=np.inf, tlsa_max=np.inf,  tlsb_max=-1.499, Q_max=np.inf,
                    sigma = None):
        """
        Get the model which match the PSD noise.
        In this function the Quality Factor is calculated as well
        """

        def combined_model_q(freqs,gr_noise,tau_qp,tls_a,tls_b,Qr):
            # Ruido Generación-Recombinación
            gr = gr_noise/(1.+(2*np.pi*freqs*tau_qp)**2) / (1.+(2*np.pi*freqs*Qr/np.pi/kid_f0)**2)
            # Ruido TLS
            tls = tls_a*freqs**tls_b / (1.+(2*np.pi*freqs*Qr/np.pi/kid_f0)**2)
            # Ruido del amplificador
            amp = amp_noise
            # Ruido Total
            return gr + tls + amp

        if gr_guess is None:
            gr_guess = 0.01
        if tauqp_guess is None:
            tauqp_guess = 1./(psd_freqs.max()-psd_freqs.min())
        if tlsa_guess is None:
            tlsa_guess = psd_df[-1]
        if tlsb_guess is None:
            tlsb_guess = 1.5

        guess = np.array([gr_guess, tauqp_guess, tlsa_guess,  tlsb_guess, Q_r ])
        bounds = np.array([ [gr_min, tauqp_min, tlsa_min,  tlsb_min, Q_min ],
                            [gr_max, tauqp_max, tlsa_max,  tlsb_max, Q_max ]])

        if sigma is None:
            sigma = (1 / abs(np.gradient(psd_freqs)))

        #print sigma

        pval, pcov = curve_fit(combined_model_q, psd_freqs, psd_df, guess, bounds=bounds, sigma=sigma)
        (gr_noise,tau_qp,tls_a,tls_b,Q_r) = pval

        fit_PSD = combined_model_q(psd_freqs,gr_noise,tau_qp,tls_a,tls_b,Q_r)

        return gr_noise,tau_qp,amp_noise,tls_a,tls_b,Q_r,fit_PSD

    def spectral_noise(self, freq, Nqp, tqp):
        """
        Expression for the calculus of Spectral Noise

        Parameters
        ----------
        Nqp : float number
            Number of quasiparticles
        tqp : float number
            Lifetime of quasiparticles
        freq : array
            Frequency of the spectral plot
        Returns
        -------
        Sn : array
            Spectral noise

        References
        -------
        [1] P. J. de Visser et al. J. Low Temperature Physics 2012
        """
        w = 2*np.pi*freq
        return (4*Nqp*tqp)/(1 + (w*tqp)**2)

    def fitPSD(self, freq, psd, bounds=None):
        """Function to fit a skewed Lorentzian to a resonator sweep. Uses scipy.optimize.curve_fit
        to perform least squares minimization.

        Parameters
        ----------
        freq : array_like
            Array containing frequency values of spectral noise
        psd : array_like
            Array containing spectral noise values

        Returns
        -------
        fitoutput : tuple
            A tuple containing three arrays;
                - popt; parameters found from least-squares fit. [Nqp, tqp]
                - perr; an estimate of the errors associated with the fit parameters
                - fit; the fit function evaluated using the fitting parameters
        """

        weights = np.ones_like(freq)

        init = np.array([0, 0])

        if bounds is not None:
            param_bounds = bounds
        else:
            param_bounds = np.array([np.array([0,0]), np.array([np.inf, 10])])

        PSDfitp, PSDcov = curve_fit(self.spectral_noise, freq, psd, p0=init, bounds = param_bounds,sigma=weights)

        errs = np.sqrt(np.diag(PSDcov))
        fitPSD = self.spectral_noise(freq, *PSDfitp)

        Nqp = PSDfitp[0]
        tqp = PSDfitp[1]

        errNqp = errs[0]
        errtqp = errs[1]

        return np.array([Nqp, tqp]), np.array([errNqp, errtqp]), fitPSD


# FIT psd with interactive mode
class fit_psd(object):
    def __init__(self, name="", sm_degree=9):
        # Class to fit the PSD noise
        self.fit_PSD = fit_spectral_noise()
        self.sm_degree = sm_degree
        self.plot_name = name

    def update_psd_fit(self, freq_psd, psd, f0_fits, Q, amp_noise):
        # SMooth applied
        #n_psd = scipy.signal.savgol_filter(psd, self.sm_degree, 3)
        n_freq, n_psd = self.decimate_psd(freq_psd, psd, w=5)

        # Fit PSD curve
        gr_noise_s,tau_s,amp_noise_s,tls_a_s,tls_b_s,fit_PSD_s = self.fit_PSD.fit_kid_psd(freq_psd, n_freq, n_psd, f0_fits, Q, amp_noise,
                gr_guess=None,  tauqp_guess=None,   tlsa_guess=None,  tlsb_guess=-0.5,
                gr_min=0,       tauqp_min=0,        tlsa_min=-np.inf, tlsb_min=-2.0,
                gr_max=np.inf,  tauqp_max=np.inf,   tlsa_max=np.inf,  tlsb_max=-0.01,
                sigma = None)

        return gr_noise_s,tau_s,amp_noise_s,tls_a_s,tls_b_s,fit_PSD_s

    def decimate_psd(self, freq_psd, psd, w=10):
        """
        Decimate PSD.
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

    # Update plot
    def update_plot(self, freq_psd, psd, psd_fit):
        # Plot original and fit
        self._ax.semilogx(freq_psd, 10*np.log10(psd), 'c')
    
        #psd = scipy.signal.savgol_filter(psd, self.sm_degree, 3)
        #self._ax.semilogx(freq_psd, 10*np.log10(psd), 'r', lw=1)
        
        n_freq, n_psd = self.decimate_psd(freq_psd, psd, w=5)
        #m_freq, m_psd = self.decimate_psd(freq_psd, psd, w=10)
        self._ax.semilogx(n_freq, 10*np.log10(n_psd), 'r', lw=1)
        #self._ax.semilogx(m_freq, 10*np.log10(m_psd), 'g', lw=1)
        
        self._ax.semilogx(freq_psd, 10*np.log10(psd_fit), 'k')

        self._ax.set_title(self.plot_name)
        self._ax.set_xlabel(r'Frequency [Hz]', fontsize=18, weight='bold')
        self._ax.set_ylabel(r'PSD [Hz$^2$/Hz] ', fontsize=18, weight='bold')

    # Update PSD
    def fit_psd(self, freq_psd, psd, f0_fits, Q, amp_noise, inter=False):
        # Global parameters
        self.Q = Q
        self.f0_fits = f0_fits
        self.amp_noise = amp_noise
        # Data
        self.freq_psd = freq_psd
        self.psd = psd
        # Initial fit
        gr_noise_s,tau_s,amp_noise_s,tls_a_s,tls_b_s,fit_PSD_s = self.update_psd_fit(freq_psd, psd, f0_fits, Q, amp_noise)
        # ----- PARAMETERS -----
        self.fit_psd = fit_PSD_s
        # Store parameters
        self.gr_noise = gr_noise_s
        self.tau = tau_s
        self.amp_noise = amp_noise_s
        self.tls_a = tls_a_s
        self.tls_b = tls_b_s
        # Interactive mode
        if inter:
            self.interactive_mode(freq_psd, psd, fit_PSD_s)

    # Interactive mode to clean psd data
    def interactive_mode(self, freq_psd, psd, fit_PSD_s):
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

        self._onclick_xy = self._fig.canvas.mpl_connect('button_press_event', self._onclick)
        self._keyboard = self._fig.canvas.mpl_connect('key_press_event', self._key_pressed)

        show()

    # Key pressed
    def _key_pressed(self, event):
        """
            Keyboard event to save/discard line fitting changes
        """
        sys.stdout.flush()
        if event.key in ['x', 'q', 'd', 'w']:

            freq_amp = 7.5e4

            if event.key == 'x':
                print('Changes saved')#, 'ok')
                self._fig.canvas.mpl_disconnect(self._onclick_xy)
                self._fig.canvas.mpl_disconnect(self._key_pressed)
                close(self._fig)
                try:
                    # ----- PARAMETERS -----
                    self.fit_psd = self._fit_psd
                    # Store parameters
                    self.gr_noise = self._gr_noise
                    self.tau = self._tau
                    self.amp_noise = self._amp_noise
                    self.tls_a = self._tls_a
                    self.tls_b = self._tls_b
                except:
                    print('No changes')#, 'warn')

            elif event.key == 'q':
                print('No changes to the fitting')#, 'warn')
                self._fig.canvas.mpl_disconnect(self._onclick_xy)
                self._fig.canvas.mpl_disconnect(self._key_pressed)
                close(self._fig)

            elif event.key == 'w':

                x_sort = np.sort(self.x_range)
                # Si no se limpia una zona segura
                _freq_psd = np.concatenate((self._freq_psd[:x_sort[0]],self._freq_psd[x_sort[1]:]))
                self.amp_noise = np.mean(self._psd[np.where(_freq_psd>freq_amp)[0]])
                if not np.isnan(self.amp_noise):
                    # Redefine freq and psd
                    self._freq_psd = _freq_psd
                    self._psd = np.concatenate((self._psd[:x_sort[0]],self._psd[x_sort[1]:]))
                    # Fit new curve
                    gr_noise_s,tau_s,amp_noise_s,tls_a_s,tls_b_s,fit_PSD_s = self.update_psd_fit(self._freq_psd, self._psd, self.f0_fits, self.Q, self.amp_noise)
                    # ----- PARAMETERS -----
                    self._fit_psd = fit_PSD_s
                    # Store parameters
                    self._gr_noise = gr_noise_s
                    self._tau = tau_s
                    self._amp_noise = amp_noise_s
                    self._tls_a = tls_a_s
                    self._tls_b = tls_b_s
                    # Update plot
                    cla()
                    self.update_plot(self._freq_psd, self._psd, fit_PSD_s)
                    self._fig.canvas.draw_idle()
                    print('Fringe removed')#, 'ok')
                else:
                    print('Delete all data > '+str(freq_amp)+'Hz is not allowed')#, 'fail')

            elif event.key == 'd':
                self._cnt = 0
                self._idx = 0
                self.x_range = np.zeros(2, dtype=int)
                # Update plot
                cla()
                self.update_plot(self.freq_psd, self.psd, self.fit_psd)
                self._freq_psd = self.freq_psd
                self._psd = self.psd
                self._fig.canvas.draw_idle()
                print('Clear filters')#, 'ok')

    def _onclick(self, event):
        """
            On click event to select lines
        """
        if event.inaxes == self._ax:
            # Left-click
            if event.button == 3:
                ix, iy = event.xdata, event.ydata
                # Add detectors
                if ix>self._freq_psd[-1]:
                    xarg = len(self._freq_psd)
                else:
                    xarg = np.where(self._freq_psd>ix)[0][0]
                self._ax.axvline(ix)

                self._cnt += 1
                self._idx = self._cnt%2
                self.x_range[self._idx] = xarg

                self._fig.canvas.draw_idle()