# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# KIDs Lab. Homodyne system
# homodyne.py
# Class to read/process data taken by the homodyne system.
#
# Marcial Becerril, @ 26 March 2024
# Latest Revision: 26 Mar 2024, 21:10 GMT-6
#
# TODO list:
# Functions missing:
#   + Show dictionary structure
#   + Read Continuous VNA sweeps and find number of KIDs
#       + KID finder for the detdaq machine
#   + Visualizer tool
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# mbecerrilt@inaoep.mx
#
# --------------------------------------------------------------------------------- #

import time
import sys

import numpy as np
from tqdm import tqdm
from os import walk

import xlsxwriter

from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.signal import butter, lfilter, freqz
from scipy import signal
from scipy import optimize
from scipy.optimize import curve_fit

from astropy.io import fits

#import matplotlib
#matplotlib.use('Agg')
from matplotlib.pyplot import *
ion()

from homodyne_funcs import *
from fit_resonators import *

from multiprocessing import Process, Manager
fitRes = Manager().dict()

sys.path.append('../')
from misc.msg_custom import *
from misc.display_dicts import *
from misc.misc_funcs import *

#from kid_finder import *
from data_processing import *

from datetime import datetime


# G L O B A L   D E F I N I T I O N S
# ----------------------------------------------------------
TEMP_TYPE = {'mK':'D',  'K':'B'}
BACKUP_FILE = "proj_bkp"

lstyle = ['-', '-', '--', 'dotted', '.-']
lmarker = ['s', '^', 'o', '*']

class Homodyne:
    """
    Homodyne data object.
    Handle homodyne data adqusition: timestreams + sweep.
    Parameters
    ----------
    diry : string
        Data directory.
    ----------
    """
    def __init__(self, diry, *args, **kwargs):

        # Key arguments
        # ----------------------------------------------
        # Project name
        proj_name = kwargs.pop('proj_name', None)
        # Working directory
        work_dir = kwargs.pop('work_dir', './')
        # Load only VNA sweeps?
        only_vna = kwargs.pop('only_vna', False)
        # Overdriven power for each KID
        overdriven = kwargs.pop('overdriven', [])
        # Expected KIDs
        expected = kwargs.pop('expected', None)
        # Load saved project
        load_saved = kwargs.pop('load_saved', False)
        # Additional input attenuation
        add_in_att = kwargs.pop('add_in_att', 0)
        # Additional output attenuation
        add_out_att = kwargs.pop('add_out_att', 0)
        # ----------------------------------------------

        # Create a directory for the project
        self.date, self.data_type, self.meas, self.sample = '', 'unkwnown', 'unkwnown', 0

        self.work_dir = work_dir

        self.overdriven = overdriven

        if proj_name is None:
            name = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.project_name = 'P-'+name
        else:
            self.project_name = proj_name

        if not load_saved:
            # Loading data base
            msg('Loading directory...', 'info')
            self.data = self.load_data(diry, only_vna=only_vna, expected=expected)
        else:
            msg('Loading saved file', 'info')
            self.data = np.load(diry+'/backup_data/'+BACKUP_FILE+'.npy', allow_pickle=True).item()
            
            diry = np.load(diry+'/backup_data/data_diry.npy')

        self.data_diry = diry

        foldername = diry.split('/')[-2]
        self.date, self.data_type, self.meas, self.sample = self._get_meas_char_from_foldername(foldername)

        # Create working directory
        self._create_diry(self.work_dir+self.project_name)
        self._create_diry(self.work_dir+self.project_name+'/fit_res_results')
        self._create_diry(self.work_dir+self.project_name+'/fit_res_results/vna')
        self._create_diry(self.work_dir+self.project_name+'/fit_res_results/homodyne')
        self._create_diry(self.work_dir+self.project_name+'/fit_res_results/summary_plots')

        self._create_diry(self.work_dir+self.project_name+'/fit_res_dict')
        self._create_diry(self.work_dir+self.project_name+'/fit_res_dict/vna')
        self._create_diry(self.work_dir+self.project_name+'/fit_res_dict/homodyne')

        self._create_diry(self.work_dir+self.project_name+'/fit_psd_results')
        self._create_diry(self.work_dir+self.project_name+'/fit_psd_dict')

        self._create_diry(self.work_dir+self.project_name+'/backup_data')

        self.add_in_atten = add_in_att
        self.add_out_atten = add_out_att

        self.found_kids = []

        # Material properties
        self.Delta = 1

        msg('Done', 'ok')

    def load_data(self, path, w=0.25, only_vna=False, expected=None):
        """
        Get all the timestream directories + vna sweeps.
        Parameters
        ----------
        path : string
            Path folder from where the data will be loaded.
        w(opt) : float
            (1 - w) is the fraction of data to select from the original
            frequency range.
        ----------
        """
        # Get the directories
        directories = next(walk(path), (None, None, []))[1]

        if len(directories) == 0:
            print ('No directories found')
            return -1

        # Get the data and store it in a dictionary
        data = {}
        for diry in directories:
            if diry.startswith('KID_K') and diry[5:].isnumeric():
                if not only_vna:
                    msg(diry, 'ok')
                    if not 'ts' in data:
                        data['ts'] = {}
                    kid = diry[4:]
                    data['ts'][kid] = {}
                    # Now get the sub-diry
                    subdirs = next(walk(path+diry), (None, None, []))[1]
                    for subdiry in tqdm(subdirs, desc='Loading detectors'):
                        temp_units = subdiry.split('_')[3]
                        temp_prefix = TEMP_TYPE[temp_units]
                        if self.data_type == 'Dark':
                            nz = 4
                        else:
                            nz = 3
                        temp = temp_prefix + subdiry.split('_')[2].zfill(nz)

                        data['ts'][kid][temp] = {}
                        # Finally the subsubdiry
                        subsubdirs = next(walk(path+diry+'/'+subdiry), (None, None, []))[1]
                        for subsubdiry in subsubdirs:
                            atten = f'A{round(float(subsubdiry[16:-2]), 1):.1f}'

                            data['ts'][kid][temp][atten] = {}

                            full_path = path+diry+'/'+subdiry+'/'+subsubdiry
                            data['ts'][kid][temp][atten]['path'] = full_path

                            # Read the homodyne data
                            try:
                                (f_s21, s21, sweep_hr), (ts_low_on, I_low_on, Q_low_on, hr_low_on), (ts_high_on, I_high_on, Q_high_on, hr_high_on), \
                                (ts_low_off, I_low_off, Q_low_off, hr_low_off), (ts_high_off, I_high_off, Q_high_off, hr_high_off) = get_homodyne_data(full_path)

                                # Frequency sweeps from homodyne.
                                data['ts'][kid][temp][atten]['f'] = f_s21
                                data['ts'][kid][temp][atten]['s21'] = s21
                                # Timestream ON/OFF resonance.
                                data['ts'][kid][temp][atten]['ts_on'] = [ts_low_on, ts_high_on]
                                data['ts'][kid][temp][atten]['ts_off'] = [ts_low_off, ts_high_off]
                                # Timestream I/Q data.
                                data['ts'][kid][temp][atten]['I_on'] = [I_low_on, I_high_on]
                                data['ts'][kid][temp][atten]['I_off'] = [I_low_off, I_high_off]
                                data['ts'][kid][temp][atten]['Q_on'] = [Q_low_on, Q_high_on]
                                data['ts'][kid][temp][atten]['Q_off'] = [Q_low_off, Q_high_off]
                                # Timestream ON/OFF header
                                data['ts'][kid][temp][atten]['hdr_on'] = [hr_low_on, hr_high_on]
                                data['ts'][kid][temp][atten]['hdr_off'] = [hr_low_off, hr_high_off]
                                # Frequency HR-sweep.
                                data['ts'][kid][temp][atten]['s21_hr'] = sweep_hr

                            except:
                                msg(f'Error reading file: {full_path}', 'fail')

            elif diry == 'VNA_Sweeps':
                msg(diry, 'ok')
                data['vna'] = {}
                data['vna']['full'] = {}
                vna_files = next(walk(path+diry), (None, None, []))[2]

                for vna_file in tqdm(vna_files, desc='Loading VNA files'):
                    # Get a name
                    if vna_file.startswith('S21_'):
                        vna_type, temp, temp_units, atten, n_sample = self._extract_features_from_name(vna_file)
                        vna_name = vna_type+'-'+temp+'-'+atten

                        print('****************')
                        print(vna_name, n_sample)

                        temp_prefix = TEMP_TYPE[temp_units]
                        temp = temp_prefix+temp
                        atten = 'A'+atten

                        # Read the VNA sweep
                        vna_full_path = path+diry+'/'+vna_file
                        vna_data = fits.getdata(vna_full_path)
                        f = vna_data.field(0)
                        I = vna_data.field(1)
                        Q = vna_data.field(2)
                        s21 = I + 1j*Q

                        # Read the continuous sweeps
                        if vna_type == 'con':

                            print('++++++++++++++++++++++++')
                            print(vna_full_path)

                            if not temp in data['vna']['full']:
                                data['vna']['full'][temp] = {}
                            if not atten in data['vna']['full'][temp]:
                                data['vna']['full'][temp][atten] = {}
                            if not 'data' in data['vna']['full'][temp][atten]:
                                data['vna']['full'][temp][atten]['data'] = {}
                                data['vna']['full'][temp][atten]['header'] = {}
                                data['vna']['full'][temp][atten]['path'] = {}

                            data['vna']['full'][temp][atten]['data'][n_sample] = [f, s21]
                            data['vna']['full'][temp][atten]['path'][n_sample] = vna_full_path

                            hdul = fits.open(vna_full_path)
                            hdr = hdul[1].header
                            data['vna']['full'][temp][atten]['header'][n_sample] = hdr

                        elif vna_type == 'seg':
                            # Extract data by KID
                            # ------------------------------------------------
                            n_kids = self._find_kids_segmented_vna(f, thresh=5e4, exp=expected)

                            # Split data per KID
                            for kid in range(len(n_kids)-1):
                                # Get detector
                                from_idx = n_kids[kid]+1
                                to_idx = n_kids[kid+1]
                                # Get individual detector
                                f_kid = f[from_idx:to_idx]
                                s21_kid = s21[from_idx:to_idx]

                                f_kid = f_kid[int(w*len(f_kid)):-int(w*len(f_kid))]
                                s21_kid = s21_kid[int(w*len(s21_kid)):-int(w*len(s21_kid))]

                                str_kid = 'K'+str(kid).zfill(3)

                                if not str_kid in data['vna']:
                                    data['vna'][str_kid] = {}
                                if not temp in data['vna'][str_kid]:
                                    data['vna'][str_kid][temp] = {}
                                if not atten in data['vna'][str_kid][temp]:
                                    data['vna'][str_kid][temp][atten] = {}

                                #data['vna'][str_kid][temp][atten] = {}

                                if not 'data' in data['vna'][str_kid][temp][atten]:
                                    data['vna'][str_kid][temp][atten]['data'] = {}
                                    data['vna'][str_kid][temp][atten]['header'] = {}

                                #data['vna'][str_kid][temp][atten]['data'] = [f_kid, s21_kid]
                                data['vna'][str_kid][temp][atten]['data'][n_sample] = [f_kid, s21_kid]
                                hdul = fits.open(vna_full_path)
                                hdr = hdul[1].header
                                data['vna'][str_kid][temp][atten]['header'][n_sample] = hdr

        return data

    def save_data(self, filename=None):
        """
        Save analysis in a numpy file.
        Parameters
        ----------
        filename : string
            File name of saved data.
        ----------
        """
        # If the path is not defined
        if filename is None:
            filename = BACKUP_FILE

        np.save(self.work_dir+self.proj_name+'/backup_data/'+filename, self.data)
        np.save(self.work_dir+self.proj_name+'/backup_data/data_diry.npy', self.data_diry)

    def fit_vna_resonators(self, kid=None, temp=None, atten=None, sample=0, n=3.5, **kwargs):
        """
        Fit the VNA resonators.
        Define a set of data to fit. If none is defined, it will
        fit all the detectors.
        Parameters
        ----------
        kid : int, string
            KID ID. If 'None' it will take all the resonators.
        temp : int, string
            Temperature. If 'None' it will take all the temperatures, whether base
            temperatura or Blackbody temperature.
        atten : string
            Attenuation. If 'None' it will select all the attenuations.
        sample(opt) : int
            Sample number.
        n(opt) : float
            Window size to fit. By default it is 3.5.
        ----------
        """

        # Key arguments
        # ----------------------------------------------
        # Cable delay
        tau = kwargs.pop('tau', 50e-9)
        # Name
        prefix = kwargs.pop('prefix', 'A')
        # Plot results
        plot_res = kwargs.pop('plot_res', True)
        # Verbose
        verbose = kwargs.pop('verbose', False)
        # ----------------------------------------------

        jobs = []
        kids = self._get_kids_to_sweep(kid)
        for kid in kids:
            msg(kid, 'info')
            print('***************************************')

            temps = self._get_temps_to_sweep(temp, kid)
            for temp in temps:

                attens = self._get_atten_to_sweep(atten, temp, kid)
                for atten in attens:
                    msg('Fit '+kid, 'info')

                    try:
                        f_vna, s21_vna = self.data['vna'][kid][temp][atten]['data'][sample]

                        p = Process(target=self._fit_res_join, args=(kid, temp, atten, f_vna, s21_vna, n, tau ))
                        jobs.append(p)
                        p.start()

                    except KeyError as err:
                        print(err)
                        msg('Data not found', 'fail')

                    except Exception as e:
                        print(e)
                        msg('Fit is taking too much time...', 'fail')

        for proc in jobs:
        	proc.join()

        for k in fitRes.keys():

            parse_key = k.split(',')
            kid = parse_key[0]
            temp = parse_key[1]
            atten = parse_key[2]

            if not 'fit' in self.data['vna'][kid][temp][atten]:
                self.data['vna'][kid][temp][atten]['fit'] = {}

            self.data['vna'][kid][temp][atten]['fit'][sample] = fitRes[k]

            try:
                Qr = self.data['vna'][kid][temp][atten]['fit'][sample]['Qr']
                Qc = self.data['vna'][kid][temp][atten]['fit'][sample]['Qc']

                Qr_err = self.data['vna'][kid][temp][atten]['fit'][sample]['Qr_err']
                Qc_err = self.data['vna'][kid][temp][atten]['fit'][sample]['Qc_err']

                e1 = -Qr**2/(Qc-Qr)**2
                e2 = Qc**2/(Qc-Qr)**2
                print('Qi_error')
                print(np.sqrt( (e1*Qc_err)**2 + (e2*Qr_err)**2 ))
                self.data['vna'][kid][temp][atten]['fit'][sample]['Qi_err'] = np.sqrt( (e1*Qc_err)**2 + (e2*Qr_err)**2 )

            except:
                print('Error getting Qi error')

            # Save data
            np.save(self.work_dir+self.project_name+'/fit_res_dict/vna/fit-'+str(kid)+'-'+str(temp).zfill(3)+'-'+str(atten).zfill(2)+'-S'+str(sample), self.data['vna'][kid][temp][atten]['fit'][sample])
            msg(str(kid)+'-'+str(temp).zfill(3)+'-'+str(atten).zfill(2)+'-S'+str(sample)+' fit file saved', 'ok')

            if verbose:
                # Show the initial parameters
                print('R E S U L T S')
                msg(f"fr : {round(fitRes[k]['fr'], 1):.0f} Hz", 'info')
                msg(f"Qr : {round(fitRes[k]['Qr'], 1):.0f}", 'info')
                msg(f"Qc : {round(fitRes[k]['Qc'], 1):.0f}", 'info')
                msg(f"Qi : {round(fitRes[k]['Qi'], 1):.0f}", 'info')
                msg(f"phi : {fitRes[k]['phi']:.2f}", 'info')
                msg(f"non : {fitRes[k]['non']:.2f}", 'info')

            # P L O T   T H E   R E S U L T S
            # ------------------------------------------------------------------
            if plot_res:

                f, s21 = self.data['vna'][kid][temp][atten]['data'][sample]
                f_fit = self.data['vna'][kid][temp][atten]['fit'][sample]['freq_data']
                s21_fit = self.data['vna'][kid][temp][atten]['fit'][sample]['fit_data']
                err = 20*np.log10(np.abs(s21)) - 20*np.log10(np.abs(s21_fit))

                rc('font', family='serif', size='16')
                fig = figure(figsize=(20,10))
                gs = GridSpec(nrows=2, ncols=2, width_ratios=[1, 1], height_ratios=[6, 1])

                subplots_adjust(
                    top=0.98,
                    bottom=0.075,
                    left=0.05,
                    right=0.98,
                    hspace=0.0,
                    wspace=0.12
                )

                ax_sweep = fig.add_subplot(gs[0, 0])
                ax_sweep.plot(f, 20*np.log10(np.abs(s21)), 'b.-', label='Data')
                ax_sweep.plot(f_fit, 20*np.log10(np.abs(s21_fit)), 'k-', label='Fit')
                ax_sweep.set_ylabel('dB')
                ax_sweep.legend()
                fr = self.data['vna'][kid][temp][atten]['fit'][sample]['fr']
                Qr = self.data['vna'][kid][temp][atten]['fit'][sample]['Qr']
                Qc = self.data['vna'][kid][temp][atten]['fit'][sample]['Qc']
                Qi = self.data['vna'][kid][temp][atten]['fit'][sample]['Qi']

                if 'Qr_err' in self.data['vna'][kid][temp][atten]['fit'][sample]:
                    Qr_err = self.data['vna'][kid][temp][atten]['fit'][sample]['Qr_err']
                    Qc_err = self.data['vna'][kid][temp][atten]['fit'][sample]['Qc_err']
                    Qi_err = self.data['vna'][kid][temp][atten]['fit'][sample]['Qi_err']

                    ax_sweep.text(f[0], np.min(20*np.log10(np.abs(s21_fit))) + (np.max(20*np.log10(np.abs(s21_fit))) - \
                    np.min(20*np.log10(np.abs(s21_fit))) )/2, f'{kid}\nT: {temp[1:]} mK\nA: {atten[1:]}dB\nF0: {fr/1e6:.2f} MHz\nQr: {round(Qr, 0):,.0f}+/-{round(Qr_err, 0):,.0f} \nQc: {round(Qc, 0):,.0f}+/-{round(Qc_err, 0):,.0f} \nQi: {round(Qi, 0):,.0f}+/-{round(Qi_err, 0):,.0f}' )
                else:
                    ax_sweep.text(f[0], np.min(20*np.log10(np.abs(s21_fit))) + (np.max(20*np.log10(np.abs(s21_fit))) - \
                    np.min(20*np.log10(np.abs(s21_fit))) )/2, f'{kid}\nT: {temp[1:]} mK\nA: {atten[1:]}dB\nF0: {fr/1e6:.2f} MHz\nQr: {round(Qr, 0):,.0f} \nQc: {round(Qc, 0):,.0f} \nQi: {round(Qi, 0):,.0f}' )

                ax_sweep.grid()

                ax_err = fig.add_subplot(gs[1, 0])
                ax_err.plot(f, err, 'k.')
                ax_err.set_ylabel('Residual')
                ax_err.set_xlabel('Frequency[Hz]')
                ax_err.grid()

                ax_iq = fig.add_subplot(gs[:, 1])
                ax_iq.axis('equal')
                ax_iq.plot(s21.real, s21.imag, 'r.-', label='Data')
                ax_iq.plot(s21_fit.real, s21_fit.imag, 'k-', label='Fit')
                #plot(I[m_idx], Q[m_idx], 'ko')
                ax_iq.set_xlabel('I')
                ax_iq.set_ylabel('Q')
                ax_iq.legend()
                ax_iq.grid()

                for location in ['left', 'right', 'top', 'bottom']:
                    ax_sweep.spines[location].set_linewidth(2)
                    ax_err.spines[location].set_linewidth(2)
                    ax_iq.spines[location].set_linewidth(2)

                name = str(kid)+'-'+str(temp)+'-'+str(atten)+'-S'+str(sample)
                fig.savefig(self.work_dir+self.project_name+'/fit_res_results/vna/'+prefix+'-'+name+'.png')
                close(fig)

                ax_err, ax_iq, ax_sweep = None, None, None

            del fitRes[k]

    def merge_fit_res(self, kid=None, temp=None, atten=None, samples=None):
        """
        Merge the fit resonators results.
        Parameters
        ----------
        kid : int, string
            KID ID. If 'None' it will take all the resonators.
        temp : int, string
            Temperature. If 'None' it will take all the temperatures, whether base
            temperatura or Blackbody temperature.
        atten : string
            Attenuation. If 'None' it will select all the attenuations.
        sample(opt) : int
            Sample number. If 'None' take all the samples/repeats.
        ----------
        """

        kids = self._get_kids_to_sweep(kid)
        for kid in kids:
            msg('->'+str(kid), 'info')
            temps = self._get_temps_to_sweep(temp, kid)
            for tmp in temps:
                msg('  ->'+str(tmp), 'info')
                attens = self._get_atten_to_sweep(atten, tmp, kid)
                for att in attens:
                    msg('    -->'+str(att), 'info')
                    try:
                        samples = self.data['vna'][kid][tmp][att]['fit'].keys()
                        for item in ['ar', 'ai', 'fr', 'Qr', 'Qc', 'Qi', 'phi', 'non']:

                            #cnt = 0
                            #sum_r = 0
                            #sqr_err_r = 0
                            values = []
                            for sample in samples:
                                if isinstance(sample, int):

                                    value = self.data['vna'][kid][tmp][att]['fit'][sample][item]
                                    values.append(value)

                                    #err_val = self.data['vna'][kid][temp][atten]['fit'][sample][item+'_err']
                                    #sum_r += value
                                    #sqr_err_r += err_val**2
                                    #cnt += 1

                            """
                            print('******************************')
                            print(item)
                            print('mean:', np.mean(values))
                            print('std:', np.std(values))
                            """

                            self.data['vna'][kid][tmp][att]['fit'][item] = np.mean(values)
                            self.data['vna'][kid][tmp][att]['fit'][item+'_err'] = np.std(values)

                            #val_q = sum_r/cnt
                            #err_q = (1/cnt)*np.sqrt(sqr_err_r)

                    except:
                        print('No fit in: ', kid, tmp, att)

    def find_overdriven_atts(self, temp, sample=0, thresh=0.7, inter=False):
        """
        Find the pre-overdriven attenuations given the fit results + manual selection (opt).
        """
        # Select all the KIDs
        kids = self._get_kids_to_sweep(None)
        
        # Temporal overdriven atts
        self.temp_att = np.zeros_like(kids, dtype=float)

        cnt = 0
        fig_cnt = -1
        tot_cnt = 0
        self.n_fig_od = 0
        ioff()
        for k, kid in enumerate(kids):
            # Select all the attenuations
            tmp = self._get_temps_to_sweep(temp, kid)[0]
            attens = self._get_atten_to_sweep(None, tmp, kid)
            kid_non = np.zeros_like(attens, dtype=float)
            for a, att in enumerate(attens):
                # Check the non-linearity
                kid_non[a] = self.data['vna'][kid][tmp][att]['fit'][sample]['non']

            idx = len(kid_non) - np.where(kid_non[::-1]>thresh)[0]
            if len(idx) > 0:
                idx = idx[0]
            else:
                idx = 0

            if k%6 == 0:
                if fig_cnt >= 0:
                    self._create_od_fig(fig, ax)

                fig, ax = subplots(6, 5)
                subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.07, hspace=0.0, wspace=0.0)
                
                self.over_atts_mtx = np.zeros((6,5))
                self.over_atts_mask = np.zeros((6,5), dtype=bool)

                fig_cnt += 1
                self.n_fig_od = fig_cnt
                cnt = 0

            # Assign overdriven attenuations
            self.temp_att[k] = float(attens[idx][1:])
            
            for i in np.arange(5):
                idx_map = i + idx - 2
                
                ii = cnt%5
                jj = int(cnt/5)

                if idx_map >= 0:
                    f, s21 = self.data['vna'][kid][tmp][attens[idx_map]]['data'][sample]
                    f_fit = self.data['vna'][kid][tmp][attens[idx_map]]['fit'][sample]['freq_data']
                    fit_s21 = self.data['vna'][kid][tmp][attens[idx_map]]['fit'][sample]['fit_data']

                    ax[jj, ii].plot(s21.real, s21.imag, 'ro-')
                    ax[jj, ii].plot(fit_s21.real, fit_s21.imag, 'k')

                    ax[jj, ii].axis('equal')

                    if i == 2:
                        ax[jj, ii].patch.set_facecolor('green')
                    else:
                        ax[jj, ii].patch.set_facecolor('blue')

                    ax[jj, ii].patch.set_alpha(0.2)

                    #if idx_map == idx:
                    #    text_size = 18
                    #    box_color = 'cyan'
                    #    text_color = 'black'
                    #else:
                    text_size = 14
                    text_color = 'white'
                    box_color = 'purple'

                    ax[jj, ii].text(np.min(s21.real)-0.25*(np.max(s21.real)-np.min(s21.real)), np.min(s21.imag), attens[idx_map] + ' dB', {'fontsize': text_size, 'color':text_color}, bbox=dict(facecolor=box_color, alpha=0.95))
                    self.over_atts_mtx[jj, ii] = float(attens[idx_map][1:])
                    self.over_atts_mask[jj, ii] = True

                if jj == 5 or tot_cnt == len(kids)-1:
                    ax[jj, ii].set_xlabel("I [V]")
                if cnt%5 == 0:
                    ax[jj, ii].set_ylabel(kid+"\nQ [V]")

                cnt += 1
                tot_cnt += 1

        self._create_od_fig(fig, ax)

        ion()

    def _create_od_fig(self, fig, ax):
        """
        Create an overdriven plot.
        """
        self.ax_od = ax 
        self.fig_od = fig

        self._onclick_subplot = self.fig_od.canvas.mpl_connect("button_press_event", self._onclick_subplot_event)
        self._keyboard_subplot = self.fig_od.canvas.mpl_connect('key_press_event', self._key_subplot_event)
        show()

    def _onclick_subplot_event(self, event):
        """
        Subplot click event.
        """
        for i in range(6):
            for j in range(5):
                if event.inaxes == self.ax_od[i, j]:
                    #print ("event in ", i, j)
                    if self.over_atts_mask[i, j]:
                        self.update_overdriven_plot(i, j)
                        self.temp_att[self.n_fig_od*6+i] = self.over_atts_mtx[i, j]

    def update_overdriven_plot(self, i, j):
        """
        Update overdriven plot.
        """
        for m in range(5):
            if self.over_atts_mask[i, m]:
                if m == j:
                    self.ax_od[i, m].patch.set_facecolor('green')
                else:
                    self.ax_od[i, m].patch.set_facecolor('blue')
                self.ax_od[i, m].patch.set_alpha(0.2)
        self.fig_od.canvas.draw_idle()

    def _key_subplot_event(self, event):
        """
        Subplot keyboard event.
        """
        sys.stdout.flush()
        if event.key in ['x', 'd']:

            if event.key == 'x':
                self.fig_od.canvas.mpl_disconnect(self._onclick_subplot)
                self.fig_od.canvas.mpl_disconnect(self._keyboard_subplot)
                close(self.fig_od)

                self.overdriven = self.temp_att
                msg('Changes saved!', 'info')

            elif event.key == 'd':
                for i in range(6):
                    self.update_overdriven_plot(i, 2)
                    self.temp_att[self.n_fig_od*6+i] = self.over_atts_mtx[i, 2]

                #print(self.temp_att)
                msg('Undoing changes', 'info')

    def find_kids(self, f, s21, down_factor=35, baseline_params=(501, 5), Qr_lim=[1500, 150000], Qc_lim=[1000, 150000], inter=True):
        """
        Find resonators from the VNA sweep.
        """

        # 1. Invert the S21 sweep
        # ---------------------------------
        s21_log = 20*np.log10(np.abs(s21))
        s21_log_inv = -1*s21_log
        # 2. Downsampling
        # ---------------------------------
        s21_down = s21_log_inv[::down_factor]
        f_down = f[::down_factor]

        s21_down = np.append(s21_down, s21_log_inv[-1])
        f_down = np.append(f_down, f[-1])

        # 3. Extract the baseline
        # ---------------------------------
        npoints = baseline_params[0]
        order = baseline_params[1]
        baseline_down = savgol_filter(s21_down, npoints, order)
        inter_mdl = interpolate.interp1d(f_down, baseline_down)
        baseline = inter_mdl(f)

        s21_no_base = s21_log_inv - baseline

        sm_s21_no_base = savgol_filter(s21_no_base, 7, 2)
        peaks, _ = find_peaks(sm_s21_no_base, distance=50, height=0.5, prominence=0.5)

        df = np.mean(np.diff(f))
        nw_peaks = []
        flags = []
        for peak in peaks:
            size_sample = 6*f[peak]/Qr_lim[0]/2
            range_a = peak - int(size_sample/df)
            range_b = peak + int(size_sample/df)
            # Take a sample
            fsm = f[range_a:range_b]
            s21_sm = s21[range_a:range_b]

            ar, ai, Qr, fr, Qc, phi = coarse_fit(fsm, s21_sm, tau=50e-9)

            if (Qr > Qr_lim[0] and Qr < Qr_lim[1]) and (Qc > Qc_lim[0] and Qc < Qc_lim[1]):
                #axvline(f[peak], color='r')
                flags.append(True)
            else:
                #axvline(f[peak], color='k', lw=0.5)
                flags.append(False)

            nw_peaks.append(peak)

        #xlabel('Frequency[MHz]')
        #ylabel('S21[dB]')

        # Interactive mode
        if inter:
            self.interactive_mode(f, s21, nw_peaks, flags)
        else:
            sel_peaks = []
            for p, peak in enumerate(nw_peaks):
                if flags[p]:
                    sel_peaks.append(f[peak])
            self.found_kids = sel_peaks

    # Interactive mode to clean psd data
    def interactive_mode(self, f, s21, peaks, flags):
        # Create figures
        self._fig = figure()
        self._ax = self._fig.add_subplot(111)

        self._freq = f
        self._s21 = s21

        self._peaks = peaks
        self._flags = flags

        self._mode = True

        self._freq_backup = np.copy(f)
        self._s21_backup = np.copy(s21)
        self._peaks_backup = np.copy(peaks)
        self._flags_backup = np.copy(flags)

        self.update_plot(self._freq, self._s21, self._peaks, self._flags)

        self._onclick_xy = self._fig.canvas.mpl_connect('button_press_event', self._onclick)
        self._keyboard = self._fig.canvas.mpl_connect('key_press_event', self._key_pressed)

        show()

    # Update plot
    def update_plot(self, freq, s21, peaks, flags):
        # Plot original and fit
        self._ax.semilogx(freq, 20*np.log10(np.abs(s21)), 'b')

        for p, peak in enumerate(peaks):
            if flags[p]:
                axvline(freq[peak], color='r')
            else:
                axvline(freq[peak], color='k', lw=0.5)

        if self._mode:
            self._ax.patch.set_facecolor('green')
            self._ax.patch.set_alpha(0.2)
        else:
            self._ax.patch.set_facecolor('red')
            self._ax.patch.set_alpha(0.2)

        self._ax.set_xlabel(r'Frequency [Hz]')
        self._ax.set_ylabel(r'S21 [dB]')

    # Key pressed
    def _key_pressed(self, event):
        """
        Keyboard event to save/discard line fitting changes
        """
        sys.stdout.flush()
        if event.key in ['x', 'q', 'd', 'u', 'a', 'r']:

            if event.key == 'x':
                self._fig.canvas.mpl_disconnect(self._onclick_xy)
                self._fig.canvas.mpl_disconnect(self._key_pressed)
                close(self._fig)

                # Save data
                sel_peaks = []
                for p, peak in enumerate(self._peaks):
                    if self._flags[p]:
                        sel_peaks.append(self._freq[peak])
                self.found_kids = sel_peaks
                msg('Changes saved!', 'info')

            elif event.key == 'u':
                cla()
                self.update_plot(self._freq, self._s21, self._peaks, self._flags)
                self._fig.canvas.draw_idle()

            elif event.key == 'a':
                self._mode = True
                cla()
                self.update_plot(self._freq, self._s21, self._peaks, self._flags)
                self._fig.canvas.draw_idle()

            elif event.key == 'r':
                self._mode = False
                cla()
                self.update_plot(self._freq, self._s21, self._peaks, self._flags)
                self._fig.canvas.draw_idle()

            elif event.key == 'd':
                self._freq = self._freq_backup
                self._s21 = self._s21_backup
                self._peaks = self._peaks_backup
                self._flags = self._flags_backup
                cla()
                self.update_plot(self._freq, self._s21, self._peaks, self._flags)
                self._fig.canvas.draw_idle()
                msg('Undoing changes', 'info')

    def _onclick(self, event, thresh=5e4):
        """
        On click event to select lines
        """
        if event.inaxes == self._ax:
            # Left-click
            if event.button == 3:
                ix, iy = event.xdata, event.ydata
                # Add detectors
                if ix > self._freq[-1]:
                    xarg = len(self._freq)
                else:
                    xarg = np.where(self._freq>ix)[0][0]

                flag_done = True
                for p, peak in enumerate(self._peaks):
                    if np.abs(ix - self._freq[peak]) < thresh:
                        if self._mode:
                            self._flags[p] = True
                        else:
                            self._flags[p] = False
                        flag_done = False
                        break

                if flag_done and self._mode:
                    ix_idx = np.where(ix<self._freq)[0][0]
                    self._peaks.append(ix_idx)
                    self._flags.append(True)

                if self._mode:
                    self._ax.axvline(ix, color='g')
                    self._fig.canvas.draw_idle()
                else:
                    cla()
                    self.update_plot(self._freq, self._s21, self._peaks, self._flags)
                    self._fig.canvas.draw_idle()

    def split_continuous_by_kid(self, temp=None, atten=None, lws=6, Qr=1000):
        """
        Divide the VNA by the found detectors. Under construction...
        """

        temps = self._get_temps_to_sweep(temp, vna_type='con')
        print(temps)

        """
        df = np.mean(np.diff(f))
        for kid, freq in enumerate(self.found_kids):
            size_sample = lws*freq/Qr/2

            freq_idx = np.where(f>=freq)[0][0]

            range_a = freq_idx - int(size_sample/df)
            range_b = freq_idx + int(size_sample/df)

            f_sample = f[range_a:range_b]
            s21_sample = s21[range_a:range_b]

            plot(f_sample, 20*np.log10(np.abs(s21_sample)))
        """


    def get_responsivity(self, kid, atten, temp_conv='Nqp', material='Al', V=1, nu=35e9, var='fr', sample=0, flag_kid=[], custom=None, data_source='vna', diry_fts='/home/marcial/Homodyne-project/FFT-ANL-SLIM-SO-23', from_fit=False, method='min', plot_res=True):
        """
        Get responsivity
        Parameters
        ----------
        kid : int, string
            KID IDs. If 'None' it will take all the resonators.
        atten : string
            Attenuation. If 'None' it will select all the attenuations.
        temp_conv(opt) : string
            Convert temperature to a defined parameter.
            If 'None', use temperature as itself.
               'Nqp' convert to number of quasiparticles.
        var(opt) : string
            Parameter to evaluate the responsivity:
            'fr'(default): Resonance frequency
            'Qr': Total quality factor.
            'Qc': Coupling quality factor.
            'Qi': Intrinsec quality factor.
        sample(opt) : int
            Sample number. If 'None' take all the samples/repeats.
        material(opt) : string
            Defined material. Only necessary for Nqp calculation.
        V(opt) : float
            Device volume[um³].
        nu(opt) : float
            Bandwidth.
        flag_kid(opt) : list
            List of detectors to flag.
        custom(opt) : dict
            Customise the plots.
        data_source(opt) : string
            'vna' to use the sweeps from the VNA.
            'homo' uses the Homodyne sweeps.
        diry_fts(opt) : string
            Folder that contains the FTS measurements to convert BB data to power.
        from_fit(opt) : bool
            'True' uses the numbers from the fit.
            'False'. If var == 'fr', it gets the fr according to the 'method' defined.
        method(opt) : string
            Method to get the resonance frequency 'fr'.
        plot_res(opt) : bool
            Plot the responsivity as var vs power/BB/Nqp
        ----------
        """

        if from_fit == False and method == 'min':
            if var != 'fr':
                msg("Only var = 'fr' is valid under these conditions", "fail")
                return

        lstyle_pointer = 0

        if self.data_type.lower() == 'dark':
            temp_field = 'SAMPLETE'
        elif self.data_type.lower() == 'blackbody':
            temp_field = 'BLACKBOD'

        kids = self._get_kids_to_sweep(kid)

        xg = int(np.sqrt(len(kids)))
        yg = int(len(kids)/xg)

        S = []
        pwrs = []
        if from_fit:
            figure()

        for k, kid in enumerate(kids):
            msg(kid, 'info')

            i = k%yg
            j = int(k/yg)

            temps = self._get_temps_to_sweep(None, kid)
            xs = np.zeros_like(temps, dtype=float)
            real_temp = np.zeros_like(temps, dtype=float)
            for t, tm in enumerate(temps):
                att = self._get_atten_to_sweep(atten[k], tm, kid)[0]
                try:
                    if from_fit:
                        if data_source == 'vna':
                            if (not 'fit' in self.data['vna'][kid][tm][att]) or kid in flag_kid:
                                if var == 'fr':
                                    msg(kid+'-'+tm+'-'+att+'. Fit data not available, using the min criteria.', 'info')
                                    f, s21 = self.data['vna'][kid][tm][att]['data'][sample]
                                    x = f[np.argmin(20*np.log10(np.abs(s21)))]
                                else:
                                    x = None
                            else:
                                x = self.data['vna'][kid][tm][att]['fit'][var]
                        elif data_source == 'homo':
                            if var == 'fr':
                                x = self.data['ts'][kid][tm][att]['fit'][var]
                            else:
                                x = self.data['ts'][kid][tm][att]['fit_psd']['params'][var]
                    else:
                        print(kid, tm, att)
                        if method == 'min':
                            f, s21 = self.data['vna'][kid][tm][att]['data'][sample]
                            x = f[np.argmin(20*np.log10(np.abs(s21)))]

                    xs[t] = x

                except:
                    pass

                real_temp[t] = float(self.data['vna'][kid][tm][att]['header'][0][temp_field])

            try:
                power = []
                for rt in real_temp:
                    print('***********************************************')
                    msg('Temperature[K]: '+str(rt), 'info')
                    if temp_conv == 'power':
                        if method == 'fts':
                            p = get_power_from_FTS(diry_fts, k, rt)
                        elif method == 'bandwidth':
                            p = bb2pwr(rt, nu[k])

                        msg('Power[pW]: '+str(p*1e12), 'info')
                        power.append(p)

                    if temp_conv == 'Nqp':
                        Delta = get_Delta(Tcs[material])
                        self.Delta = Delta

                        nqp = get_nqp(N0s[material], rt, Delta)
                        Nqp = nqp * V
                        power.append(Nqp)

                        msg('Material: '+material, 'info')
                        msg('Volume [um³]: '+str(V), 'info')
                        msg('Energy gap: '+str(Delta), 'info')
                        msg('Nqp: '+str(Nqp), 'info')

                power = np.array(power)
                #print(power)

                # Get the responsivity
                # ------------------------------------------------------------------
                if temp_conv == 'power':
                    popt, pcov = curve_fit(f0_vs_pwr_model, power, xs, p0=[1e4, -0.5])
                    a, b = popt
                    dF0_dP = a*b*power**(b-1)

                    S.append(dF0_dP)

                elif temp_conv == 'Nqp':

                    #plot(power, xs, 'rs-')
                    dF0_dNqp, b = np.polyfit(power[2:], xs[2:], 1)
                    Nqps_fit = np.linspace(power[0], power[-1], 1000)
                    #plot(Nqps_fit, Nqps_fit*dF0_dNqp + b, 'k')

                    S.append(dF0_dNqp)

                pwrs.append(power)

            except Exception as e:
                S.append(None)
                pwrs.append(None)

            """
            # P O W E R
            # ---------------------------------------------
            #P0 = bb2pwr(real_temp, nu[k])
            P0 = get_power_from_FTS(diry_fts, k, real_temp)
            print(P0)
            #P0 = get_P_from_FTS('../FTS/', kid, real_bb_temps[bb_temps.index(temp)], nu=f0)

            dF0_dP = a*b*P0**(b-1)
            b0 = f0_vs_pwr_model(P0, a, b) - P0*dF0_dP

            P_fit = np.linspace(power[0]-5e-12, power[-2], 1000)
            plot(1e12*power, f0_vs_pwr_model(power, *popt), 'rs-', label='fit')
            plot(1e12*P_fit, (P_fit*dF0_dP + b0), 'k')
            return
            # ---------------------------------------------
            """

            if plot_res:
                if k%10 == 0:
                    lstyle_pointer += 1
                if var == 'fr':
                    xs_plot = (xs - xs[0])/xs[0]
                    ylabel('ffs')
                else:
                    xs_plot = xs
                    ylabel(var)

                if not np.sum([np.isnan(i) for i in xs_plot]) == len(xs_plot):

                    if not custom is None:
                        color = custom[0][k]
                        mk = custom[1][k]
                        lsty = custom[2][k]
                        plot(power, xs_plot, label=kid, linestyle=lsty, marker=mk, color=color)

                    else:
                        plot(power, xs_plot, label=kid, linestyle=lstyle[lstyle_pointer], marker=lmarker[lstyle_pointer])

                    if temp_conv == 'power':
                        xlabel('BB temp [K]')
                    elif temp_conv == 'Nqp':
                        xlabel('Nqp')
                    legend(ncol=2)

        return S, pwrs

    def calculate_df(self, I, Q, hdr):
        """
        Get the resonance frequency shift.
        Parameters
        ----------
        I/Q : array/list
            Quadrature signals.
        hdr : dictionary
            Sweep Header to get IF0, QF0, DIDF and DQDF
        ----------
        """

        # Get I and Q at the resonance frequency
        I0 = hdr['IF0']
        Q0 = hdr['QF0']
        # Get the gradients
        dqdf = hdr['DQDF']
        didf = hdr['DIDF']

        df, didq_mag = get_df(I, Q, didf, dqdf, I0, Q0)

        return df, didq_mag

    def get_psd_on_off(self, kid, temp, atten, ignore=[[0,1], [0]], fit=True, plot_fit=True):
        """
        Get the PSD on - off for a set of resonators.
        Parameters
        ----------
        kid : int/list/array
            KID IDs. If 'None' take all the resonators available.
        temp : int/list/array
            Define the temperature. If 'None' take all the temperatures available.
        atten : int/list/array
            Define the attenuations. If 'None' take all the temperatures available.
        ignore : list
            Define the timestreams samples not used.
            First item for the low-frequency band.
            Second item for the high-frequency band.
        fit : bool
            Fit raw PSD?
        plot_fit : bool
            Plot raw PSD(on-off) + the fit + save graph.
        ----------
        """

        f_off, psd_off, df_off = self.calculate_psd(kid, temp, atten, mode='off', ignore=ignore)
        self.data['ts'][kid][temp][atten]['psd'] = {}
        self.data['ts'][kid][temp][atten]['psd']['off'] = [f_off, psd_off]
        f_on, psd_on, df_on = self.calculate_psd(kid, temp, atten, mode='on', ignore=ignore)
        self.data['ts'][kid][temp][atten]['psd']['on'] = [f_on, psd_on]

        f0 = self.data['vna'][kid][temp][atten]['fit']['fr']
        Qr = self.data['vna'][kid][temp][atten]['fit']['Qr']

        if fit:
            f_nep, psd_nep, fit_psd_params, f, k_knee = fit_mix_psd(f_on, psd_on, psd_off, f0, Qr, amp_range=[7.5e4, 8.0e4], trim_range=[0, 9e4])
            self.data['ts'][kid][temp][atten]['fit_psd'] = {}
            self.data['ts'][kid][temp][atten]['fit_psd']['params'] = fit_psd_params
            self.data['ts'][kid][temp][atten]['fit_psd']['psd'] = [f_nep, psd_nep]
            self.data['ts'][kid][temp][atten]['fit_psd']['k_knee'] = k_knee

        if fit and plot_fit:

            fig, ax = subplots(1,1, figsize=(20,10))
            subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.07, hspace=0.0, wspace=0.0)

            ax.loglog(f_nep, psd_nep, lw=1.0)
            fit_PSD = combined_model(f_nep, fit_psd_params['gr'], fit_psd_params['tau'], fit_psd_params['tls_a'], fit_psd_params['tls_b'], f0, Qr, fit_psd_params['amp_noise'])
            ax.loglog(f_nep, fit_PSD, 'k', lw=2.5, label='fit')

            # Ruido Generación-Recombinación
            gr = fit_psd_params['gr']/(1.+(2*np.pi*f_nep*fit_psd_params['tau'])**2) / (1.+(2*np.pi*f_nep*Qr/np.pi/f0)**2)
            # Ruido TLS
            tls = fit_psd_params['tls_a']*f_nep**fit_psd_params['tls_b'] / (1.+(2*np.pi*f_nep*Qr/np.pi/f0)**2)

            ax.loglog(f_nep, gr, 'r-', lw=2.5, label='gr noise')
            ax.text(0.5, gr[0]*1.5, f'GR:{gr[0]:.3f} Hz^2/Hz')

            ax.loglog(f_nep, tls, 'b-', lw=2.5, label='1/f')
            ax.loglog(f_nep, fit_psd_params['amp_noise']*np.ones_like(f_nep), 'g-', label='amp', lw=2)

            tau = fit_psd_params['tau']
            ax.text(2000, 0.1*np.max(psd_nep), f'Qr  : {round(Qr,-1):,.0f}\nf0   : {round(f0/1e6,-1):,.0f} MHz\ntau : {tau*1e6:.1f} us')

            #amp = fit_psd_params['amp_noise']
            #ax.text(0.5, amp*1.5, f'{amp:.3f} Hz^2/Hz')
            ax.axhline(fit_psd_params['gr'], color='k', linestyle='dashed', lw=2)

            knee = self.data['ts'][kid][temp][atten]['fit_psd']['k_knee']
            ax.axvline(knee, color='m', lw=2.5)
            ax.text(knee, 1e-3, f'1/f knee: {knee:.1f} Hz')
            ax.set_ylim([1e-4, 1e3])

            ax.grid(True, which="both", ls="-")
            name = str(kid)+'-'+str(temp)+'-'+str(atten)
            base_temp = self.data['ts'][kid][temp][atten]['s21_hr']['SAMPLETE']
            ax.set_title('PSD-noise-'+name+'-'+str(base_temp)+' K')
            ax.set_xlabel('Frequency[Hz]')
            ax.set_ylabel(r'PSD [Hz$^2$/Hz]')

            ax.legend()

            fig.savefig(self.work_dir+self.project_name+'/fit_psd_results/'+name+'.png')
            close(fig)

            np.save(self.work_dir+self.project_name+'/fit_psd_dict/'+name, self.data['ts'][kid][temp][atten]['fit_psd'])

    def calculate_psd(self, kid, temp, atten, mode='on', ignore=[[0,1,2], [0]]):
        """
        Calculate the PSD.
        Parameters
        ----------
        kid : int/list/array
            KID IDs. If 'None' take all the resonators available.
        temp : int/list/array
            Define the temperature. If 'None' take all the temperatures available.
        atten : int/list/array
            Define the attenuations. If 'None' take all the temperatures available.
        mode : string
            'on' or 'off' timestreams data.
        ignore : list
            Define the timestreams samples not used.
            First item for the low-frequency band.
            Second item for the high-frequency band.
        ----------
        """
        try:
            # Low-frequency PSD
            I_low = self.data['ts'][kid][temp][atten]['I_'+mode][0]
            Q_low = self.data['ts'][kid][temp][atten]['Q_'+mode][0]
            I_low_f, Q_low_f = [], []
            for i in range(len(I_low)):
                if not i in ignore[0]:
                    I_low_f.append(I_low[i])
                    Q_low_f.append(Q_low[i])

            print('Low-freq samples: ', len(I_low_f))

            hdr_low = self.data['ts'][kid][temp][atten]['hdr_'+mode][0]
            df_low, didq_mag_low = self.calculate_df(I_low_f, Q_low_f, hdr_low)

            fs = hdr_low['SAMPLERA']
            print(fs)
            freq_low, psd_low = get_psd(df_low, fs)

            # High-frequency PSD
            I_high = self.data['ts'][kid][temp][atten]['I_'+mode][1]
            Q_high = self.data['ts'][kid][temp][atten]['Q_'+mode][1]
            I_high_f, Q_high_f = [], []
            for i in range(len(I_high)):
                if not i in ignore[1]:
                    I_high_f.append(I_high[i])
                    Q_high_f.append(Q_high[i])

            print('High-freq samples: ', len(I_high_f))

            hdr_high = self.data['ts'][kid][temp][atten]['hdr_'+mode][1]
            df_high, didq_mag_high = self.calculate_df(I_high_f, Q_high_f, hdr_high)

            fs = hdr_high['SAMPLERA']
            print(fs)
            freq_high, psd_high = get_psd(df_high, fs)

            f, psd = mix_psd([freq_low, freq_high], [psd_low, psd_high])

            return f, psd, (df_low, df_high)

        except Exception as e:
            msg('Data not available', 'fail')
            print(e)
            return -1

    def despike(self, kid=None, temp=None, atten=None, ignore=[[0,1], [0]], win_size=350, sigma_thresh=3.5, peak_pts=4, **kwargs):
        """
        Despike the timestreams.
        Parameters
        ----------
        kid : int/list/array
            KID IDs. If 'None' take all the resonators available.
        temp : int/list/array
            Define the temperature. If 'None' take all the temperatures available.
        atten : int/list/array
            Define the attenuations. If 'None' take all the attenuations available.
        ignore : list
            Define the timestreams samples not used.
            First item for the low-frequency band.
            Second item for the high-frequency band.
        win_size : int
            Window size to detect and extract the resonators.
        sigma_thresh : float
            Spike events detection threshold.
        peaks_pts : int
            Minimum amount of continuous points to consider the event as an spike.
        ----------
        """
        # Verbose
        verbose = kwargs.pop('verbose', False)

        kids = self._get_kids_to_sweep(kid, mode='ts')
        for kid in kids:
            print(kid)
            temps = self._get_temps_to_sweep(temp, kid, mode='ts')
            for tmp in temps:
                print(' ->', tmp)
                attens = self._get_atten_to_sweep(atten, tmp, kid, mode='ts')
                for att in attens:
                    print('     -->', att)
                    try:
                        # Low-res
                        n_lw_ts = len(self.data['ts'][kid][tmp][att]['I_on'][0])
                        for lw in range(n_lw_ts):
                            if not lw in ignore[0]:

                                I_on = self.data['ts'][kid][tmp][att]['I_on'][0][lw]
                                I_off = self.data['ts'][kid][tmp][att]['I_off'][0][lw]

                                i_on_t, c1 = cr_filter(I_on, win_size=win_size, sigma_thresh=sigma_thresh, peak_pts=peak_pts, verbose=verbose)
                                self.data['ts'][kid][tmp][att]['I_on'][0][lw] = i_on_t
                                i_off_t, c1 = cr_filter(I_off, win_size=win_size, sigma_thresh=sigma_thresh, peak_pts=peak_pts, verbose=verbose)
                                self.data['ts'][kid][tmp][att]['I_off'][0][lw] = i_off_t

                                Q_on = self.data['ts'][kid][tmp][att]['Q_on'][0][lw]
                                Q_off = self.data['ts'][kid][tmp][att]['Q_off'][0][lw]

                                q_on_t, c1 = cr_filter(Q_on, win_size=win_size, sigma_thresh=sigma_thresh, peak_pts=peak_pts, verbose=verbose)
                                self.data['ts'][kid][tmp][att]['Q_on'][0][lw] = q_on_t
                                q_off_t, c1 = cr_filter(Q_off, win_size=win_size, sigma_thresh=sigma_thresh, peak_pts=peak_pts, verbose=verbose)
                                self.data['ts'][kid][tmp][att]['Q_off'][0][lw] = q_off_t

                        # High-res
                        n_hg_ts = len(self.data['ts'][kid][tmp][att]['I_on'][1])
                        for hg in range(n_hg_ts):
                            if not hg in ignore[1]:

                                I_on = self.data['ts'][kid][tmp][att]['I_on'][1][hg]
                                I_off = self.data['ts'][kid][tmp][att]['I_off'][1][hg]

                                i_on_t, c1 = cr_filter(I_on, win_size=win_size, sigma_thresh=sigma_thresh, peak_pts=peak_pts, verbose=verbose)
                                self.data['ts'][kid][tmp][att]['I_on'][1][hg] = i_on_t
                                i_off_t, c1 = cr_filter(I_off, win_size=win_size, sigma_thresh=sigma_thresh, peak_pts=peak_pts, verbose=verbose)
                                self.data['ts'][kid][tmp][att]['I_off'][1][hg] = i_off_t

                                Q_on = self.data['ts'][kid][tmp][att]['Q_on'][1][hg]
                                Q_off = self.data['ts'][kid][tmp][att]['Q_off'][1][hg]

                                q_on_t, c1 = cr_filter(Q_on, win_size=win_size, sigma_thresh=sigma_thresh, peak_pts=peak_pts, verbose=verbose)
                                self.data['ts'][kid][tmp][att]['Q_on'][1][hg] = q_on_t
                                q_off_t, c1 = cr_filter(Q_off, win_size=win_size, sigma_thresh=sigma_thresh, peak_pts=peak_pts, verbose=verbose)
                                self.data['ts'][kid][tmp][att]['Q_off'][1][hg] = q_off_t

                    except Exception as e:
                        msg('Error loading file\n'+str(e), 'warn')

    def load_fit(self, folder, data_type='vna'):
        """
        Load fit files.
        Parameters
        ----------
        folder : string
            Folder name from where the sweeps fit are extracted.
        data_type : string
            Data type: vna or homodyne.
        ----------
        """

        files = next(walk(folder+'/fit_res_dict/'+data_type+'/'), (None, None, []))[2]

        for f in files:
            kid = f.split('-')[1]
            temp = f.split('-')[2]
            atten = f.split('-')[3]
            ns = int((f.split('-')[-1][1:]).split('.')[0])

            try:
                sample = np.load(folder+'/fit_res_dict/'+data_type+'/'+f, allow_pickle=True).item()

                if not 'fit' in self.data['vna'][kid][temp][atten]:
                    self.data['vna'][kid][temp][atten]['fit'] = {}
                self.data['vna'][kid][temp][atten]['fit'][ns] = sample

            except Exception as e:
                print('Fail loading '+f)
                print(str(e))

    # S O M E   U S E F U L   P L O T S
    # --------------------------------------------------------------------------
    def plot_qs_vs_drive_power(self, kid=None, temp=None, atten=None, cmap='jet'):
        """
        Generate the plot qi vs drive power.
        Parameters
        ----------
        kid : int/list/array
            KID IDs. If 'None' take all the resonators available.
        temp : int/list/array
            Define the temperature. If 'None' take all the temperatures available.
        atten : int/list/array
            Define the attenuations. If 'None' take all the attenuations available.
        cmap : string
            Colormap.
        ----------
        """

        cmap = matplotlib.cm.get_cmap(cmap)

        kids = self._get_kids_to_sweep(kid, mode='vna')

        temporal_temps = []
        join_temps = []
        for kid in kids:
            join_temps.append(self._get_temps_to_sweep(temp, kid, mode='vna'))
            temporal_temps.append(len(self._get_temps_to_sweep(temp, kid, mode='vna')))

        n_temps = join_temps[np.argmax(temporal_temps)]

        x = int(np.sqrt(len(n_temps)))
        y = int(len(n_temps)/x)

        fig_qi, ax_qi = subplots(x, y, sharey=True, sharex=True)
        subplots_adjust(left=0.07, right=0.99, top=0.99, bottom=0.07, hspace=0.0, wspace=0.0)
        fig_qc, ax_qc = subplots(x, y, sharey=True, sharex=True)
        subplots_adjust(left=0.07, right=0.99, top=0.99, bottom=0.07, hspace=0.0, wspace=0.0)
        fig_qr, ax_qr = subplots(x, y, sharey=True, sharex=True)
        subplots_adjust(left=0.07, right=0.99, top=0.99, bottom=0.07, hspace=0.0, wspace=0.0)

        for t, tmp in enumerate(n_temps):

            i = t%y
            j = int(t/y)

            lstyle_pointer = 0
            norm_color = matplotlib.colors.Normalize(vmin=0, vmax=len(kids))
            for k, kid in enumerate(kids):
                print(kid)
                k2 = int(kid[1:])
                attens = self._get_atten_to_sweep(atten, tmp, kid, mode='vna')

                atts_num = []
                qi, qc, qr = [], [], []
                qi_err, qc_err, qr_err = [], [], []
                for att in attens:
                    try:
                        if float(att[1:]) >= self.overdriven[k2]:
                            if not self.data['vna'][kid][tmp][att]['fit']['Qi_err'] is None:
                                print(' ->'+att)

                                #if (self.data['vna'][kid][tmp][att]['fit']['Qi_err']/self.data['vna'][kid][tmp][att]['fit']['Qi']) < 0.1:

                                # Get the attenuations
                                extra_att = self.data['vna'][kid][tmp][att]['header'][0]['ATT_UC'] + \
                                            self.data['vna'][kid][tmp][att]['header'][0]['ATT_C'] + \
                                            self.data['vna'][kid][tmp][att]['header'][0]['ATT_RT']

                                #print(self.data['vna'][kid][tmp][att]['header'][0]['ATT_UC'])
                                #print(self.data['vna'][kid][tmp][att]['header'][0]['ATT_C'])
                                #print(self.data['vna'][kid][tmp][att]['header'][0]['ATT_RT'])

                                atts_num.append(-1*(float(att[1:])+extra_att+self.add_in_atten) )

                                # Get Qs errors
                                qi_err.append(self.data['vna'][kid][tmp][att]['fit']['Qi_err'])
                                qc_err.append(self.data['vna'][kid][tmp][att]['fit']['Qc_err'])
                                qr_err.append(self.data['vna'][kid][tmp][att]['fit']['Qr_err'])

                                # Get Qs
                                qi.append(self.data['vna'][kid][tmp][att]['fit']['Qi'])
                                qc.append(self.data['vna'][kid][tmp][att]['fit']['Qc'])
                                qr.append(self.data['vna'][kid][tmp][att]['fit']['Qr'])

                    except Exception as e:
                        print(att+' not available\n'+str(e))

                if k%10 == 0:
                    lstyle_pointer += 1

                if len(n_temps) == 1:
                    """
                    ax_qi.errorbar(atts_num, qi, yerr=qi_err, color=cmap(norm_color(k)), marker='s', ecolor='k', capsize=2, linestyle=lstyle[lstyle_pointer])
                    ax_qi.plot(atts_num, qi, 's', label=kid, color=cmap(norm_color(k)), linestyle=lstyle[lstyle_pointer])

                    ax_qc.errorbar(atts_num, qc, yerr=qc_err, color=cmap(norm_color(k)), marker='s', ecolor='k', capsize=2, linestyle=lstyle[lstyle_pointer])
                    ax_qc.plot(atts_num, qc, '^', label=kid, color=cmap(norm_color(k)), linestyle=lstyle[lstyle_pointer])

                    ax_qr.errorbar(atts_num, qr, yerr=qr_err, color=cmap(norm_color(k)), marker='s', ecolor='k', capsize=2, linestyle=lstyle[lstyle_pointer])
                    ax_qr.plot(atts_num, qr, 'o', label=kid, color=cmap(norm_color(k)), linestyle=lstyle[lstyle_pointer])
                    """
                    #ax_qi.errorbar(atts_num, qi, yerr=qi_err, marker='s', ecolor='k', capsize=2, linestyle=lstyle[lstyle_pointer])
                    ax_qi.plot(atts_num, qi, 's', label=kid,  linestyle=lstyle[lstyle_pointer])

                    #ax_qc.errorbar(atts_num, qc, yerr=qc_err, marker='s', ecolor='k', capsize=2, linestyle=lstyle[lstyle_pointer])
                    ax_qc.plot(atts_num, qc, '^', label=kid,  linestyle=lstyle[lstyle_pointer])

                    #ax_qr.errorbar(atts_num, qr, yerr=qr_err, marker='s', ecolor='k', capsize=2, linestyle=lstyle[lstyle_pointer])
                    ax_qr.plot(atts_num, qr, 'o', label=kid,  linestyle=lstyle[lstyle_pointer])

                else:
                    ax_qi[j, i].errorbar(atts_num, qi, yerr=qi_err, color=cmap(norm_color(k)), marker='s', ecolor='k', capsize=2, linestyle=lstyle[lstyle_pointer])
                    ax_qi[j, i].plot(atts_num, qi, 's', label=kid, color=cmap(norm_color(k)), linestyle=lstyle[lstyle_pointer])

                    ax_qc[j, i].errorbar(atts_num, qc, yerr=qc_err, color=cmap(norm_color(k)), marker='s', ecolor='k', capsize=2, linestyle=lstyle[lstyle_pointer])
                    ax_qc[j, i].plot(atts_num, qc, '^', label=kid, color=cmap(norm_color(k)), linestyle=lstyle[lstyle_pointer])

                    ax_qr[j, i].errorbar(atts_num, qr, yerr=qr_err, color=cmap(norm_color(k)), marker='s', ecolor='k', capsize=2, linestyle=lstyle[lstyle_pointer])
                    ax_qr[j, i].plot(atts_num, qr, 'o', label=kid, color=cmap(norm_color(k)), linestyle=lstyle[lstyle_pointer])

            if i == 0:
                if len(n_temps) == 1:
                    ax_qi.set_ylabel('Qi')
                    ax_qc.set_ylabel('Qc')
                    ax_qr.set_ylabel('Qr')

                    ax_qi.grid(True, which="both", ls="-")
                    ax_qc.grid(True, which="both", ls="-")
                    ax_qr.grid(True, which="both", ls="-")

                else:
                    ax_qi[j, i].set_ylabel('Qi')
                    ax_qc[j, i].set_ylabel('Qc')
                    ax_qr[j, i].set_ylabel('Qr')

            if j == y-1:
                if len(n_temps) == 1:
                    ax_qi.set_xlabel('Drive power [dB]')
                    ax_qc.set_xlabel('Drive power [dB]')
                    ax_qr.set_xlabel('Drive power [dB]')

                else:
                    ax_qi[j, i].set_xlabel('Drive power [dB]')
                    ax_qc[j, i].set_xlabel('Drive power [dB]')
                    ax_qr[j, i].set_xlabel('Drive power [dB]')

                if i == x-1:
                    if len(n_temps) == 1:
                        ax_qi.legend(ncol=2)
                        ax_qc.legend(ncol=2)
                        ax_qr.legend(ncol=2)
                    else:
                        ax_qi[j, i].legend(ncol=2)
                        ax_qc[j, i].legend(ncol=2)
                        ax_qr[j, i].legend(ncol=2)

            t += 1

    def vna_xls_report(self, name=None):
        """
        Create the report of results from the VNA measurements.
        Parameters
        ----------
        name  : string
            Folder nm
        ----------
        """

        if name == None:
            name = self.project_name+'-VNA_report.xlsx'

        workbook = xlsxwriter.Workbook(self.work_dir+self.project_name+'/'+name)

        bold = workbook.add_format({'bold': True})

        kids = self._get_kids_to_sweep(None, mode='vna')
        temps = self._get_temps_to_sweep(None, kids[0], mode='vna')

        for t, temp in enumerate(temps):
            worksheet = workbook.add_worksheet(temp)
            block = 0

            for k, kid in enumerate(kids):
                col = 0
                worksheet.write(1, 3*block+1, 'Qi', bold)
                #worksheet.write(0, 3*block+1, kid)
                worksheet.write(1, 3*block+2, 'Qc', bold)
                worksheet.write(1, 3*block+3, 'Qr', bold)
                worksheet.merge_range(0, 3*block+1, 0, 3*block+3, kid, bold)
                attens = self._get_atten_to_sweep(None, temp, kid, mode='vna')

                if k == 0:
                    worksheet.write(len(attens)+3, 0, 'Min: ', bold)
                    #worksheet.write(len(attens)+3, 0, '=MIN()', bold)
                    worksheet.write(len(attens)+4, 0, 'Max: ', bold)
                    worksheet.write(len(attens)+5, 0, 'Mean: ', bold)
                    worksheet.write(len(attens)+6, 0, 'Median: ', bold)
                    worksheet.write(len(attens)+7, 0, 'Stdev: ', bold)

                for atten in attens:
                    if k == 0:
                        worksheet.write(1, 0, 'Att[dB]', bold)
                        worksheet.write(col+2, 0, atten, bold)

                    try:
                        qi = self.data['vna'][kid][temp][atten]['fit']['Qi']
                        worksheet.write(col+2, 3*block+1, qi)
                        qc = self.data['vna'][kid][temp][atten]['fit']['Qc']
                        worksheet.write(col+2, 3*block+2, qc)
                        qr = self.data['vna'][kid][temp][atten]['fit']['Qr']
                        worksheet.write(col+2, 3*block+3, qr)
                    except:
                        pass

                    col += 1

                att_num = [float(a[1:]) for a in attens]
                from_cal = np.where(np.array(att_num)>=self.overdriven[k])[0][0]

                qi_col = xlsxwriter.utility.xl_col_to_name(3*block+1)
                worksheet.write_formula(len(attens)+3, 3*block+1, '=MIN('+qi_col+str(3+from_cal)+':'+qi_col+str(3+len(attens)-1)+')')
                qc_col = xlsxwriter.utility.xl_col_to_name(3*block+2)
                worksheet.write_formula(len(attens)+3, 3*block+2, '=MIN('+qc_col+str(3+from_cal)+':'+qc_col+str(3+len(attens)-1)+')')
                qr_col = xlsxwriter.utility.xl_col_to_name(3*block+3)
                worksheet.write_formula(len(attens)+3, 3*block+3, '=MIN('+qr_col+str(3+from_cal)+':'+qr_col+str(3+len(attens)-1)+')')

                worksheet.write_formula(len(attens)+4, 3*block+1, '=MAX('+qi_col+str(3+from_cal)+':'+qi_col+str(3+len(attens)-1)+')')
                worksheet.write_formula(len(attens)+4, 3*block+2, '=MAX('+qc_col+str(3+from_cal)+':'+qc_col+str(3+len(attens)-1)+')')
                worksheet.write_formula(len(attens)+4, 3*block+3, '=MAX('+qr_col+str(3+from_cal)+':'+qr_col+str(3+len(attens)-1)+')')

                worksheet.write_formula(len(attens)+4, 3*block+1, '=MAX('+qi_col+str(3+from_cal)+':'+qi_col+str(3+len(attens)-1)+')')
                worksheet.write_formula(len(attens)+4, 3*block+2, '=MAX('+qc_col+str(3+from_cal)+':'+qc_col+str(3+len(attens)-1)+')')
                worksheet.write_formula(len(attens)+4, 3*block+3, '=MAX('+qr_col+str(3+from_cal)+':'+qr_col+str(3+len(attens)-1)+')')

                worksheet.write_formula(len(attens)+5, 3*block+1, '=AVERAGE('+qi_col+str(3+from_cal)+':'+qi_col+str(3+len(attens)-1)+')')
                worksheet.write_formula(len(attens)+5, 3*block+2, '=AVERAGE('+qc_col+str(3+from_cal)+':'+qc_col+str(3+len(attens)-1)+')')
                worksheet.write_formula(len(attens)+5, 3*block+3, '=AVERAGE('+qr_col+str(3+from_cal)+':'+qr_col+str(3+len(attens)-1)+')')

                worksheet.write_formula(len(attens)+6, 3*block+1, '=MEDIAN('+qi_col+str(3+from_cal)+':'+qi_col+str(3+len(attens)-1)+')')
                worksheet.write_formula(len(attens)+6, 3*block+2, '=MEDIAN('+qc_col+str(3+from_cal)+':'+qc_col+str(3+len(attens)-1)+')')
                worksheet.write_formula(len(attens)+6, 3*block+3, '=MEDIAN('+qr_col+str(3+from_cal)+':'+qr_col+str(3+len(attens)-1)+')')

                worksheet.write_formula(len(attens)+7, 3*block+1, '=STDEV('+qi_col+str(3+from_cal)+':'+qi_col+str(3+len(attens)-1)+')')
                worksheet.write_formula(len(attens)+7, 3*block+2, '=STDEV('+qc_col+str(3+from_cal)+':'+qc_col+str(3+len(attens)-1)+')')
                worksheet.write_formula(len(attens)+7, 3*block+3, '=STDEV('+qr_col+str(3+from_cal)+':'+qr_col+str(3+len(attens)-1)+')')

                block += 1

            kid_cells = np.arange(0, 3*len(kids), 3)
            qi_max_str = ''
            qc_max_str = ''
            qr_max_str = ''
            for c in kid_cells:
                qi_max_col = xlsxwriter.utility.xl_col_to_name(c+1)
                qi_max_str += qi_max_col+str(len(attens)+5)+','
                qc_max_col = xlsxwriter.utility.xl_col_to_name(c+2)
                qc_max_str += qc_max_col+str(len(attens)+6)+','
                qr_max_col = xlsxwriter.utility.xl_col_to_name(c+3)
                qr_max_str += qr_max_col+str(len(attens)+6)+','

            worksheet.write(len(attens)+9, 0, 'Mean: ', bold)
            worksheet.write(len(attens)+8, 1, 'Qi Max', bold)
            worksheet.write(len(attens)+8, 2, 'Qc Means', bold)
            worksheet.write(len(attens)+8, 3, 'Qr Means', bold)
            worksheet.write_formula(len(attens)+9, 1, '=AVERAGE('+qi_max_str[:-1]+')', bold)
            worksheet.write_formula(len(attens)+9, 2, '=AVERAGE('+qc_max_str[:-1]+')', bold)
            worksheet.write_formula(len(attens)+9, 3, '=AVERAGE('+qr_max_str[:-1]+')', bold)

            worksheet.write(len(attens)+10, 0, 'Median: ', bold)
            worksheet.write_formula(len(attens)+10, 1, '=MEDIAN('+qi_max_str[:-1]+')', bold)
            worksheet.write_formula(len(attens)+10, 2, '=MEDIAN('+qc_max_str[:-1]+')', bold)
            worksheet.write_formula(len(attens)+10, 3, '=MEDIAN('+qr_max_str[:-1]+')', bold)

            worksheet.write(len(attens)+11, 0, 'Stdev: ', bold)
            worksheet.write_formula(len(attens)+11, 1, '=STDEV('+qi_max_str[:-1]+')')
            worksheet.write_formula(len(attens)+11, 2, '=STDEV('+qc_max_str[:-1]+')')
            worksheet.write_formula(len(attens)+11, 3, '=STDEV('+qr_max_str[:-1]+')')

        workbook.close()

    def plot_ts_summary(self, kid, temp, atten, ignore=[[0,1], [0]], cmap='viridis'):
        """
        Show all the timestreams.
        Parameters
        ----------
        kid : int/list/array
            KID IDs. If 'None' take all the resonators available.
        temp : int/list/array
            Define the temperature. If 'None' take all the temperatures available.
        atten : int/list/array
            Define the attenuations. If 'None' take all the attenuations available.
        ignore : list
            Define the timestreams samples not used.
            First item for the low-frequency band.
            Second item for the high-frequency band.
        cmap : string
            Colormap.
        ----------
        """

        cmap = matplotlib.cm.get_cmap(cmap)

        high_cols, low_cols = [], []
        kids = self._get_kids_to_sweep(kid, mode='ts')
        for k, kid in enumerate(kids):
            k2 = int(kid[1:])
            #print(kid, temp, atten[k])
            tmp = self._get_temps_to_sweep(temp, kid, mode='ts')[0]
            att = self._get_atten_to_sweep(atten[k2], tmp, kid, mode='ts')[0]

            low_cols.append( len(self.data['ts'][kid][tmp][att]['I_on'][0]) )
            high_cols.append( len(self.data['ts'][kid][tmp][att]['I_on'][1]) )

        xl = len(kids)
        yl = np.max(low_cols)

        fig_I_low, ax_I_low = subplots(xl, yl, sharey='row')
        subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.08, hspace=0.1, wspace=0.035)
        fig_Q_low, ax_Q_low = subplots(xl, yl, sharey='row')
        subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.08, hspace=0.1, wspace=0.035)

        xh = len(kids)
        yh = np.max(high_cols)
        ymax = 10

        figs_I, axs_I = [], []
        figs_Q, axs_Q = [], []
        for i in range(int(yh/ymax)):
            fig_I_high, ax_I_high = subplots(xh, ymax, sharey='row')
            subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.08, hspace=0.1, wspace=0.035)
            figs_I.append(fig_I_high)
            axs_I.append(ax_I_high)

            fig_Q_high, ax_Q_high = subplots(xh, ymax, sharey='row')
            subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.08, hspace=0.1, wspace=0.035)
            figs_Q.append(fig_Q_high)
            axs_Q.append(ax_Q_high)

        norm_color = matplotlib.colors.Normalize(vmin=0, vmax=len(kids))
        for k, kid in enumerate(kids):
            k2 = int(kid[1:])
            tmp = self._get_temps_to_sweep(temp, kid)[0]
            att = self._get_atten_to_sweep(atten[k2], tmp, kid)[0]

            for ts in range(yl):
                try:
                    ts_low = self.data['ts'][kid][tmp][att]['ts_on'][0]
                    I_low = self.data['ts'][kid][tmp][att]['I_on'][0][ts]
                    Q_low = self.data['ts'][kid][tmp][att]['Q_on'][0][ts]

                    ax_I_low[k, ts].plot(ts_low, I_low, lw=0.75, color=cmap(norm_color(k)))
                    #ax_I_low[k, ts].grid()

                    ax_Q_low[k, ts].plot(ts_low, Q_low, lw=0.75, color=cmap(norm_color(k)))
                    #ax_Q_low[k, ts].grid()

                    if ts == 0:
                        ax_I_low[k, ts].set_ylabel(kid+'\n I[V]')
                        ax_Q_low[k, ts].set_ylabel(kid+'\n Q[V]')
                    #else:
                        #ax_I_low[k, ts].set_yticks([])
                        #ax_Q_low[k, ts].set_yticks([])

                    if k == len(kids)-1:
                        ax_I_low[k, ts].set_xlabel('\nTime[s]')
                        ax_Q_low[k, ts].set_xlabel(kid+'\nTime[s]')
                    else:
                        ax_I_low[k, ts].set_xticks([])
                        ax_Q_low[k, ts].set_xticks([])

                    if ts in ignore[0]:
                        ax_I_low[k, ts].patch.set_facecolor('red')
                        ax_I_low[k, ts].patch.set_alpha(0.2)

                        ax_Q_low[k, ts].patch.set_facecolor('red')
                        ax_Q_low[k, ts].patch.set_alpha(0.2)

                    else:
                        ax_I_low[k, ts].patch.set_facecolor('green')
                        ax_I_low[k, ts].patch.set_alpha(0.2)

                        ax_Q_low[k, ts].patch.set_facecolor('green')
                        ax_Q_low[k, ts].patch.set_alpha(0.2)

                except Exception as e:
                    msg(kid+'-'+tmp+'-'+att+'-'+str(e), 'warn')

            cnt = -1
            for th in range(yh):
                if th%ymax == 0:
                    cnt += 1

                m = th%ymax

                try:
                    ts_high = self.data['ts'][kid][tmp][att]['ts_on'][1]
                    I_high = self.data['ts'][kid][tmp][att]['I_on'][1][th]
                    Q_high = self.data['ts'][kid][tmp][att]['Q_on'][1][th]

                    axs_I[cnt][k, m].plot(ts_high, I_high, lw=0.75, color=cmap(norm_color(k)))
                    axs_Q[cnt][k, m].plot(ts_high, Q_high, lw=0.75, color=cmap(norm_color(k)))

                    if th == 0:
                        axs_I[cnt][k, m].set_ylabel(kid+'\n I[V]')
                        axs_Q[cnt][k, m].set_ylabel(kid+'\n Q[V]')
                    #else:
                        #axs_I[cnt][k, m].set_yticks([])
                        #axs_Q[cnt][k, m].set_yticks([])

                    if k == len(kids)-1:
                        axs_I[cnt][k, m].set_xlabel('\nTime[s]')
                        axs_Q[cnt][k, m].set_xlabel(kid+'\nTime[s]')
                    else:
                        axs_I[cnt][k, m].set_xticks([])
                        axs_Q[cnt][k, m].set_xticks([])

                    if th in ignore[1]:
                        axs_I[cnt][k, m].patch.set_facecolor('red')
                        axs_I[cnt][k, m].patch.set_alpha(0.2)

                        axs_Q[cnt][k, m].patch.set_facecolor('red')
                        axs_Q[cnt][k, m].patch.set_alpha(0.2)

                    else:
                        axs_I[cnt][k, m].patch.set_facecolor('green')
                        axs_I[cnt][k, m].patch.set_alpha(0.2)

                        axs_Q[cnt][k, m].patch.set_facecolor('green')
                        axs_Q[cnt][k, m].patch.set_alpha(0.2)

                except Exception as e:
                    msg(kid+'-'+tmp+'-'+att+'-'+str(e), 'warn')

    def plot_all_s21_kids(self, kid, temp, atten, sample=0, defined_attens=True, data_source='vna', cmap='viridis'):
        """
        Plot the S21 sweeps of all the detectors.
        Parameters
        ----------
        kid : int/list/array
            KID IDs. If 'None' take all the resonators available.
        temp : int/list/array
            Define the temperature. If 'None' take all the temperatures available.
        atten : int/list/array
            Define the attenuations. If 'None' take all the attenuations available.
        sample : int
            Number of sample.
        defined_attens : int/list/array
            List of attenuations with the previous-to-overdrive attenuation per KID.
        data_source(opt) : string
            'vna' to use the sweeps from the VNA.
            'homo' uses the Homodyne sweeps.
        cmap : string
            Colormap.
        ----------
        """

        cmap = matplotlib.cm.get_cmap(cmap)

        kids = self._get_kids_to_sweep(kid)

        xg = int(np.sqrt(len(kids)))
        yg = int(len(kids)/xg)

        #xg = 3
        #yg = 6

        fig, axm = subplots(xg, yg)

        for k, kid in enumerate(kids):

            msg(kid, 'info')

            i = k%yg
            j = int(k/yg)

            temps = self._get_temps_to_sweep(temp, kid)
            norm_color = matplotlib.colors.Normalize(vmin=0, vmax=len(temps))

            for t, tm in enumerate(temps):

                if defined_attens:
                    att = atten[k]
                else:
                    att = atten

                attens = self._get_atten_to_sweep(att, tm, kid)

                if len(temps) > 1:
                    alphas = np.linspace(1.0, 0.3, len(attens))
                    norm_color = matplotlib.colors.Normalize(vmin=0, vmax=len(temps))
                    sweep_case = 1
                elif len(attens) > 1:
                    norm_color = matplotlib.colors.Normalize(vmin=0, vmax=len(attens))
                    sweep_case = 2
                else:
                    sweep_case = 3

                for a, single_atten in enumerate(attens):
                    try:
                        if sweep_case == 1:
                            alpha = alphas[a]
                            single_color = cmap(norm_color(t))
                            plot_title = kid
                            curve_label = tm+'-'+single_atten
                            curve_label = tm
                        elif sweep_case == 2:
                            alpha = 1.0
                            single_color = cmap(norm_color(a))
                            plot_title = kid+'-'+tm
                            curve_label = single_atten
                        else:
                            alpha = 1.0
                            single_color = 'r'
                            plot_title = kid+'-'+tm+'-'+single_atten
                            curve_label = plot_title

                        if data_source == 'vna':
                            f, s21 = self.data['vna'][kid][tm][single_atten]['data'][sample]
                        elif data_source == 'homo':
                            f = self.data['ts'][kid][tm][single_atten]['f']
                            s21 = self.data['ts'][kid][tm][single_atten]['s21']

                        axm[j,i].plot(f/1e6, 20*np.log10(np.abs(s21)), color=single_color, alpha=alpha, lw=1.75, label=tm)
                        if t == 0 and a == 0:
                            axm[j,i].text(f[0]/1e6+0.65*(f[-1]-f[0])/1e6, np.min(20*np.log10(np.abs(s21))), kid+'-'+single_atten )

                        #axm[j,i].set_title(kid)
                        if i == 0:
                            axm[j,i].set_ylabel('S21 [dB]')
                        if j == yg-1:
                            axm[j,i].set_xlabel('Frequency [MHz]')
                            if i == xg-1:
                                axm[j,i].legend(ncol=2)

                    except Exception as e:
                        msg('Error plotting data\n'+str(e), 'warn')

    def plot_s21_kid(self, kid, temp=None, atten=None, sample=0, data_source='vna', fit=False):
        """
        Plot all the S21 sweeps for a given resonator.
        Parameters
        ----------
        kid : int/list/array
            KID IDs.
        temp : int/list/array
            Define the temperature. If 'None' take all the temperatures available.
        atten : int/list/array
            Define the attenuations. If 'None' take all the temperatures available.
        sample : int
            Number of sample.
        data_source(opt) : string
            'vna' to use the sweeps from the VNA.
            'homo' uses the Homodyne sweeps.
        fit : bool
            Display the 'fit'.
        ----------
        """

        cmap = matplotlib.cm.get_cmap('jet')

        jobs = []
        kids = self._get_kids_to_sweep(kid)
        for kid in kids:

            figure(kid)
            msg(kid, 'info')

            temps = self._get_temps_to_sweep(temp, kid)
            norm_color = matplotlib.colors.Normalize(vmin=0, vmax=len(temps))

            for t, temp in enumerate(temps):

                attens = self._get_atten_to_sweep(atten, temp, kid)

                if len(temps) > 1:
                    alphas = np.linspace(1.0, 0.3, len(attens))
                    norm_color = matplotlib.colors.Normalize(vmin=0, vmax=len(temps))
                    sweep_case = 1
                elif len(attens) > 1:
                    norm_color = matplotlib.colors.Normalize(vmin=0, vmax=len(attens))
                    sweep_case = 2
                else:
                    sweep_case = 3

                for a, atten in enumerate(attens):
                    try:
                        if sweep_case == 1:
                            alpha = alphas[a]
                            single_color = cmap(norm_color(t))
                            plot_title = kid
                            curve_label = temp+'-'+atten
                        elif sweep_case == 2:
                            alpha = 1.0
                            single_color = cmap(norm_color(a))
                            plot_title = kid+'-'+temp
                            curve_label = atten
                        else:
                            alpha = 1.0
                            single_color = 'r'
                            plot_title = kid+'-'+temp+'-'+atten
                            curve_label = plot_title

                        if data_source == 'vna':
                            f, s21 = self.data['vna'][kid][temp][atten]['data'][sample]
                        elif data_source == 'homo':
                            f = self.data['ts'][kid][temp][atten]['f']
                            s21 = self.data['ts'][kid][temp][atten]['s21']

                        subplot(121)
                        plot(f/1e6, 20*np.log10(np.abs(s21)), color=single_color, alpha=alpha, lw=1.75 )
                        if fit and 'fit' in self.data['vna'][kid][temp][atten]:
                            f_fit = self.data['vna'][kid][temp][atten]['fit']['freq_data']
                            s21_fit = self.data['vna'][kid][temp][atten]['fit']['fit_data']
                            plot(f_fit/1e6, 20*np.log10(np.abs(s21_fit)), '-', color='k', lw=1.25 )
                        title(plot_title)
                        xlabel('Frequency [MHz]')
                        ylabel('S21 [dB]')

                        subplot(122)
                        plot(s21.real, s21.imag, color=single_color, alpha=alpha, label=curve_label, lw=1.75)
                        if fit and 'fit' in self.data['vna'][kid][temp][atten]:
                            plot(s21_fit.real, s21_fit.imag, '-', color='k', lw=1.25 )
                        title(plot_title)
                        axis('equal')
                        xlabel('I[V]')
                        ylabel('Q[V]')

                    except Exception as e:
                        msg('Error plotting data\n'+str(e), 'warn')
        legend()

    def get_sweeps_from_vna(self, temp, atten, thresh=1e5):
        """
        Get the sweeps from the segmented VNA sweeps.
        Parameters
        ----------
        temp : int
            Temperature: base or blackbody.
        atten : int
            Attenuation.
        thresh : float
            Detection threshold.
        ----------
        """

        # Find KIDs
        n_kids = np.where(np.diff(f)>thresh)[0]
        n_kids = np.concatenate(([0], n_kids, [-1]))

        # Get detector
        from_idx = n_kids[kid]+1
        to_idx = n_kids[kid+1]
        # Get individual detector
        f_k = f[from_idx:to_idx]
        # I, Q
        I_k = I[from_idx:to_idx]
        Q_k = Q[from_idx:to_idx]
        s21_k =  I_k + 1j*Q_k

        return f_k, s21_k, hdr

    def _get_meas_type(self):
        """
        Get measurement type, dark or blackbody measurement.
        """
        if self.data_type.lower() == 'dark':
            type_data = 'D'
            nzeros = 4
        elif self.data_type.lower() == 'blackbody':
            type_data = 'B'
            nzeros = 3

        return type_data, nzeros

    def _get_kids_to_sweep(self, kid, mode='vna'):
        """
        Get kids to sweep.
        """
        if kid is None:
            vna_keys = self.data[mode].keys()
            kids = [ x for x in vna_keys if 'K' in x ]
        elif isinstance(kid, int):
            kids = ['K'+str(kid).zfill(3)]
        elif isinstance(kid, list):
            kids = kid

        kids = sorted(kids)
        return kids

    def _get_temps_to_sweep(self, temp, kid=None, mode='vna', vna_type='seg'):
        """
        Get the temperatures to sweep.
        """
        type_data, nzeros = self._get_meas_type()

        if temp is None:
            if vna_type == 'seg':
                if not kid is None:
                    temps = self.data[mode][kid].keys()
                else:
                    return None
            elif vna_type == 'con':
                temps = self.data[mode]['full'].keys()

        elif isinstance(temp, int) or isinstance(temp, np.int64):
            temps = [type_data+str(temp).zfill(nzeros)]
        elif isinstance(temp, list) or isinstance(temp, np.ndarray):
            temps = []
            for t in temp:
                if isinstance(t, int) or isinstance(t, np.int64):
                    temps.append(type_data+f'{t}')
                elif isinstance(t, str):
                    if t.startswith(type_data) and t[1:].replace('.', '', 1).isdigit():
                        temps.append(t)
        else:
            temps = [temp]

        temps = sorted(temps)
        return temps

    def _get_atten_to_sweep(self, atten, temp=None, kid=None, mode='vna'):
        """
        Get the attenuations to sweep.
        """

        if atten is None:
            if not (kid is None or temp is None):
                attens = self.data[mode][kid][temp].keys()
            else:
                return None
        elif isinstance(atten, int) or isinstance(atten, np.int64):
            attens = [f'A{atten:.1f}']
        elif isinstance(atten, list) or isinstance(atten, np.ndarray):
            attens = []
            for a in atten:
                if isinstance(a, int) or isinstance(a, np.int64):
                    attens.append(f'A{a:.1f}')
                elif isinstance(a, str):
                    if a.startswith('A') and a[1:].replace('.', '', 1).isdigit():
                        attens.append(a)
        else:
            attens = [atten]

        a_num = [float(i[1:]) for i in attens]
        sort_idx = np.argsort(a_num)

        attens = list(attens)

        n_attens = [ attens[sort_idx[i]] for i in range(len(a_num)) ]

        #attens = sorted(attens)
        return n_attens

    def _create_diry(self, diry):
        """
        Create a directory.
        Parameters
        ----------
        diry : string
            Path and name of the new folder
        ----------
        """
        try:
        	os.system('mkdir '+diry)
        except Exception as e:
        	msg('Directory not created. '+str(e), 'warn')

    def _fit_res_join(self, kid, temp, atten, f, s21, n, tau):
        """
            Fit resonator job.
        """
        fit_res = fit_resonator(f, s21, n=n, tau=tau)
        fitRes[kid+','+temp+','+atten] = fit_res

    def _find_kids_segmented_vna(self, f, thresh=1e4, exp=None):
        """
        Find the KIDs in the segmented VNA.
        Parameters
        ----------
        f : array
            Frequency array.
        thresh[opt] : float
            KID detection threshold.
        ----------
        """

        if exp != None:
            segs = np.reshape(np.arange(len(f)), (exp, int(len(f)/exp)))
            n_kids = np.zeros(exp, dtype=int)
            for i in range(exp):
                n_kids[i] = segs[i][0]
            n_kids = n_kids[1:]
        else:
            # Number of KIDs
            n_kids = np.where( np.abs(np.diff(f)) > thresh )[0]

        n_kids = np.concatenate(([0], n_kids, [-1]))

        return n_kids

    def _extract_features_from_name(self, name, **kwargs):
        """
        Extract the temperature and attenuation from the VNA name.
        Parameters
        ----------
        name : string
            Filename from where the temp and attenuation are extracted.
        ----------
        """
        # Key arguments
        # ----------------------------------------------
        # Temperature subfix
        temp_subfix = kwargs.pop('temp_subfix', 'TEMP')
        # Attenuation subfix
        atten_subfix = kwargs.pop('atten_subfix', 'ATT')
        # ----------------------------------------------

        # Get the index of the temperature
        try:

            if 'segmented' in name.lower():
                vna_type = 'seg'
            elif 'continuous' in name.lower():
                vna_type = 'con'
            else:
                vna_type = 'unkwnown'

            # Get the temperature
            start_idx_temp = name.index(temp_subfix)
            end_idx_temp = name[start_idx_temp:].index('_')+start_idx_temp
            temp = name[start_idx_temp+len(temp_subfix):end_idx_temp]
            temp_units = name.split('_')[3]

            # Get the attenuation
            start_idx_atten = name.index(atten_subfix)
            end_idx_atten = name[start_idx_atten:].index('_')+start_idx_atten
            atten = name[start_idx_atten+len(atten_subfix):end_idx_atten]

            # Sample number
            sample_number = name.split('_')[-1].split('.')[0]
            if sample_number.isnumeric():
                number_sample = int(sample_number)
            else:
                number_sample = 0

            return vna_type, temp, temp_units, atten, number_sample

        except ValueError as e:
            print(e)
            msg('Data not identified in the name file', 'fail')

        return None, None, None

    def _get_meas_char_from_foldername(self, foldername):
        """
        Get the date, analysis type and sample from the folder name.
        """
        sample = 1
        items = foldername.split('_')
        if len(items) == 4:
            date, data_type, _, meas = items
        elif len(items) == 5:
            date, data_type, _, meas, sample = items
            sample += 1
        else:
            return None

        return date, data_type, meas, sample
