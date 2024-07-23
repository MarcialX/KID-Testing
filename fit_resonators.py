# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# KIDs Lab. Fit resonators.
# fit_resonators.py
# Set of functions to fit superconductive resonators.
#
# Marcial Becerril, @ 17 April 2024
# Latest Revision: 17 Apr 2024, 08:55 GMT-6
#
# TODO list:
# Functions missing:
#   + Visualizer tool
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# mbecerrilt@inaoep.mx
#
# --------------------------------------------------------------------------------- #

import os
import sys
import lmfit

import numpy as np

from scipy.signal import savgol_filter

sys.path.append('../')
from misc.msg_custom import *
from misc.timeout import timeout
from data_processing import *

from matplotlib.pyplot import *
#ion()


# R E S O N A T O R   M O D E L   F U N C T I O N S
# ---------------------------------------------------------
def cable_delay(f, ar, ai, **kwargs):
    """
        Cable delay in the transmission.
        Parameters
        ----------
        f : array
            Frequency.
        ar, ai: array
            Complex amplitude.
        tau[opt] : float
            Delay constant.
        ----------
    """
    # Key arguments
    # ----------------------------------------------
    # Cable delay
    tau = kwargs.pop('tau', 50e-9)
    # ----------------------------------------------

    # Cable delay
    return (ar+1j*ai)*np.exp(-1j*2*np.pi*f*tau)


def resonator_model(f, fr, ar, ai, Qr, Qc, phi, non, **kwargs):
    """
        Resonators nonlinear model.
        Parameters
        ----------
        f : array
            Frequency.
        fr : float
            Resonance frequency.
        ar, ai: array
            Complex amplitude.
        tau : float
            Cable delay.
        Qr : float
            Total quality factor.
        Qc : float
            Coupling quality factor.
        phi : float
            Rotation circle.
        non : float
            Nonlinear parameter.
        tau[opt] : float
            Delay constant.
        ----------
    """

    # Key arguments
    # ----------------------------------------------
    # Cable delay
    tau = kwargs.pop('tau', 50e-9)
    # ----------------------------------------------

    #tau = 50e-9
    A = (ar + 1j*ai)*np.exp(-1j*2*np.pi*f*tau)

    # Fractional frequency shift of non-linear resonator
    y0s = Qr*(f - fr)/fr
    y = np.zeros_like(y0s)

    for i, y0 in enumerate(y0s):
        coeffs = [4.0, -4.0*y0, 1.0, -(y0+non)]
        y_roots = np.roots(coeffs)
        # Extract real roots only. From [ref Pete Code]
        # If all the roots are real, take the maximum
        y_single = y_roots[np.where(abs(y_roots.imag) < 1e-5)].real.max()
        y[i] = y_single

    B = 1 - (Qr/(Qc*np.exp(1j*phi))) / (1 + 2j*y )

    return A*B


def phase_angle(f, theta_0, fr, Qr):
    """
        Phase angle.
        Parameters
        ----------
        f : array
            Frequency.
        theta_0: array
            Circle rotation.
        fr : float
            Resonance frequency.
        Qr : float
            Total quality factor.
        ----------
    """
    theta = -theta_0 + 2*np.arctan( 2*Qr*(1 - (f/fr) ) )
    return theta


def coarse_fit(f, data, **kwargs):
    """
        Coarse fit resoantor fit.
        Based on Gao PhD thesis.
        Parameters
        ----------
        f : array
            Frequency.
        data : array
            Sweep data.
        ----------
    """

    # Key arguments
    # ----------------------------------------------
    # Cable delay
    tau = kwargs.pop('tau', 50e-9)
    # ----------------------------------------------

    # Get S21
    I = data.real
    Q = data.imag

    # 1. Remove the cable delay
    # ------------------------------------------
    cable_delay_model = CableDelay(tau=tau)

    f_cable = np.concatenate((f[:10], f[-10:]))
    I_cable = np.concatenate((I[:10], I[-10:]))
    Q_cable = np.concatenate((Q[:10], Q[-10:]))

    s21_cable = I_cable + 1j*Q_cable

    guess = cable_delay_model.guess(f_cable, s21_cable)
    cable_res = cable_delay_model.fit(s21_cable, params=guess, f=f_cable)

    fit_s21 = cable_delay_model.eval(params=cable_res.params, f=f)

    ar = cable_res.values['cd_ar']
    ai = cable_res.values['cd_ai']

    s21_no_cable = data/fit_s21

    I_no_cable = s21_no_cable.real
    Q_no_cable = s21_no_cable.imag

    # 2. Derotate
    # ------------------------------------------
    idx_f0 = np.argmin( np.abs(data) )

    f0n = f[idx_f0]
    I0 = I_no_cable[idx_f0]
    Q0 = Q_no_cable[idx_f0]

    sel = np.abs(f-f0n) < 10e4
    xc, yc, r = fitCirc(I_no_cable[sel], Q_no_cable[sel])

    theta = np.arctan2(Q0-yc, I0-xc)
    I_derot = (I_no_cable-xc)*np.cos(-theta)-(Q_no_cable-yc)*np.sin(-theta)
    Q_derot = (I_no_cable-xc)*np.sin(-theta)+(Q_no_cable-yc)*np.cos(-theta)

    # 3. Fit the phase angle 400e3
    # ------------------------------------------
    sel2 = np.abs(f-f0n) < 400e3
    phase_angle_model = PhaseAngle()

    theta = np.arctan2(Q_derot[sel2], I_derot[sel2])

    for i in range(len(theta)):
        if f[sel2][i] < f0n and theta[i] < 0:
            theta[i] = theta[i] + 2*np.pi
        elif f[sel2][i] > f0n and theta[i] > 0:
            theta[i] = theta[i] - 2*np.pi

    guess = phase_angle_model.guess(f[sel2], theta)
    phase_res = phase_angle_model.fit(theta, params=guess, f=f[sel2])

    fit_theta = phase_angle_model.eval(params=phase_res.params, f=f[sel2])

    Qr = phase_res.values['pa_Qr']
    fr = phase_res.values['pa_fr']
    theta_0 = phase_res.values['pa_theta_0']

    # 4. Get Qc
    # ------------------------------------------
    mag_zc = np.sqrt(xc**2 + yc**2)
    arg_zc = np.arctan2(yc, xc)

    Qc = Qr*(mag_zc + r)/(2*r)
    phi = theta_0 - arg_zc

    return ar, ai, Qr, fr, Qc, phi


@timeout(150)
def fit_resonator(f, s21, n=3.5, **kwargs):
    """
        Fit resonator based on Gao model.
        Parameters
        ----------
        f : array
            Frequency.
        s21 : array
            Sweep data.
        n[opt] : float
            Number of linewdiths to extract data.
        ----------
    """

    # Key arguments
    # ----------------------------------------------
    # Cable delay
    tau = kwargs.pop('tau', 50e-9)
    # ----------------------------------------------

    # Fit resonator
    res_model = ResGaoModel(tau=tau)

    # Choose the region close to the detector
    guess = res_model.guess(f, s21)

    lw = guess['rgm_fr'].value/guess['rgm_Qr'].value

    # Get middle point
    I = s21.real
    Q = s21.imag
    di = np.diff(I)
    dq = np.diff(Q)
    di_dq = np.sqrt(di**2 + dq**2)
    m_idx = np.argmin(np.abs(s21))

    from_idx = f[m_idx]-n*lw
    if from_idx < f[0]:
        from_idx = 0
    else:
        from_idx = np.where(f>=from_idx)[0][0]
    #from_idx = 0

    to_idx = f[m_idx]+n*lw
    if to_idx > f[-1]:
        to_idx = len(f)
    else:
        to_idx = np.where(f>=to_idx)[0][0]

    guess = res_model.guess(f[from_idx:to_idx], s21[from_idx:to_idx])

    guess_s21 = res_model.eval(params=guess, f=f)

    result = res_model.fit(s21[from_idx:to_idx], params=guess, f=f[from_idx:to_idx])
    fit_s21 = res_model.eval(params=result.params, f=f)

    ar = result.values['rgm_ar']
    ai = result.values['rgm_ai']
    fr = result.values['rgm_fr']
    Qr = result.values['rgm_Qr']
    Qc = result.values['rgm_Qc']
    phi = result.values['rgm_phi']
    non = result.values['rgm_non']

    Qi = Qr*Qc / (Qc - Qr)

    fit_kid = {}

    # F I T   R E S U L T S.
    # --------------------------------------
    fit_kid['ar'] = ar
    fit_kid['ai'] = ai
    fit_kid['fr'] = fr
    fit_kid['Qr'] = Qr
    fit_kid['Qc'] = Qc
    fit_kid['Qi'] = Qi
    fit_kid['phi'] = phi
    fit_kid['non'] = non
    # E R R O R
    # --------------------------------------
    try:
        fit_kid['ar_err'] = result.uvars['rgm_ar'].std_dev
        fit_kid['ai_err'] = result.uvars['rgm_ai'].std_dev
        fit_kid['fr_err'] = result.uvars['rgm_fr'].std_dev
        fit_kid['Qr_err'] = result.uvars['rgm_Qr'].std_dev
        fit_kid['Qc_err'] = result.uvars['rgm_Qc'].std_dev
        fit_kid['Qi_err'] = None
        fit_kid['phi_err'] = result.uvars['rgm_phi'].std_dev
        fit_kid['non_err'] = result.uvars['rgm_non'].std_dev
    except:
        pass

    # F I T   D A T A
    # --------------------------------------
    fit_kid['fit_data'] = fit_s21 #result.best_fit
    fit_kid['freq_data'] = f #f[from_idx:to_idx]

    return fit_kid


# G A O   R E S O N A T O R   M O D E L
# ---------------------------------------------------------
class ResGaoModel(lmfit.model.Model):
    __doc__ = "resonator Gao model" + lmfit.models.COMMON_INIT_DOC

    def __init__(self, prefix='rgm_', *args, **kwargs):
        """
            Resonator model  model.
            Parameters
            ----------
            prefix : string
                Model prefix. 'rgm_' by default.
            ----------
        """

        super().__init__(resonator_model, *args, **kwargs)

        # Key arguments
        # ----------------------------------------------
        # Cable delay
        self.tau = kwargs.pop('tau', 50e-9)
        # ----------------------------------------------

        self.prefix = prefix

    def guess(self, f, data, **kwargs):
        """
            Guessing resonator parameters.
            Parameters
            ----------
            f : array
                Frequency
            data : array
                S21 parameter.
            ----------
        """

        print('T A U : ', self.tau)
        ar, ai, Qr, fr, Qc, phi = coarse_fit(f, data,  tau=self.tau)

        # Show the initial parameters
        print('I N I T I A L   P A R A M E T E R S')
        msg(f'fr : {round(fr, 1):.0f} Hz', 'info')
        msg(f'Qr : {round(Qr, 1):.0f}', 'info')
        msg(f'Qc : {round(Qc, 1):.0f}', 'info')

        # Defining the boundaries
        self.set_param_hint('%sar' % self.prefix, value=ar)
        self.set_param_hint('%sai' % self.prefix, value=ai)

        self.set_param_hint('%sQr' % self.prefix, value=Qr, min=100)
        #self.set_param_hint('%stau' % self.prefix, value=50e-9, min=40e-9, max=60e-9)
        self.set_param_hint('%sfr' % self.prefix, value=fr, min=f[0], max=f[-1])
        #self.set_param_hint('%stheta_0' % self.prefix, value=theta_0, min=-20*np.pi, max=20*np.pi)
        self.set_param_hint('%sdelta' % self.prefix, value=50e3, min=0, vary=True)

        self.set_param_hint('%sQc' % self.prefix, expr='%sdelta' % self.prefix+'+'+'%sQr' % self.prefix)
        self.set_param_hint('%sphi' % self.prefix, value=phi, min=-20*np.pi, max=20*np.pi)
        self.set_param_hint('%snon' % self.prefix, value=0.1, min=0.0, max=3.0)

        # Load the parameters to the model
        params = self.make_params()

        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


class CableDelay(lmfit.model.Model):
    __doc__ = "cable delay" + lmfit.models.COMMON_INIT_DOC

    def __init__(self, prefix='cd_', *args, **kwargs):
        """
            Cable delay model.
            Parameters
            ----------
            prefix : string
                Model prefix. 'cd_' by default
            ----------
        """

        super().__init__(cable_delay, *args, **kwargs)

        self.prefix = prefix

    def guess(self, f, data, **kwargs):
        """
            Guessing resonator parameters.
            Parameters
            ----------
            f : array
                Frequency
            data : array
                S21 parameter.
            ----------
        """

        # Amplitude
        ar_guess = np.mean(data.real)
        ai_guess = np.mean(data.imag)

        # We define the boundaries
        self.set_param_hint('%sar' % self.prefix, value=ar_guess)#, min=np.min(data.real), max=np.max(data.real))
        self.set_param_hint('%sai' % self.prefix, value=ai_guess)#, min=np.min(data.imag), max=np.max(data.imag))

        # Load the parameters to the model
        params = self.make_params()

        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)


class PhaseAngle(lmfit.model.Model):
    __doc__ = "phase angle" + lmfit.models.COMMON_INIT_DOC

    def __init__(self, prefix='pa_', *args, **kwargs):
        """
            Phase Angle model
            Parameters
            ----------
            prefix : string
                Model prefix. 'pa_' by default
            ----------
        """
        super().__init__(phase_angle, *args, **kwargs)
        self.prefix = prefix

    def guess(self, f, data, *args, **kwargs):
        """
            Guessing resonator parameters.
            Parameters
            ----------
            f : array
                Frequency
            ----------
        """

        # Total Q
        Qr_guess = 2.5e4
        # Resonance frequency
        sm_d_data = np.abs(savgol_filter( np.diff(data)/np.diff(f), 31, 3))
        idx_f0 = np.argmax(sm_d_data)
        fr_guess = f[idx_f0]
        # Theta delay
        theta_0_guess = 0

        # We define the boundaries
        self.set_param_hint('%sQr' % self.prefix, value=Qr_guess, min=100, )
        self.set_param_hint('%sfr' % self.prefix, value=fr_guess, min=f[0], max=f[-1])
        self.set_param_hint('%stheta_0' % self.prefix, value=theta_0_guess, min=-20*np.pi, max=20*np.pi)

        # Load the parameters to the model
        params = self.make_params()

        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)
