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

import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['font.size'] = 16
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['font.family'] = 'serif'
#rc('font', family='serif', size='16')

from homodyne_funcs import *
from fit_resonators import *

from multiprocessing import Process, Manager
fitRes = Manager().dict()

sys.path.append('../')
from misc.msg_custom import *
from misc.misc_funcs import *
from misc.display_dicts import *

from physics.physics_funcs import *

#from kid_finder import *
from data_processing import *

from datetime import datetime


# G L O B A L   D E F I N I T I O N S
# ----------------------------------------------------------
TEMP_TYPE = {'mK':'D',  'K':'B'}
BACKUP_FILE = "proj_bkp"

lstyle = ['-', '--', 'dotted', '.-']
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
        # Trim a fraction of the sample
        w = kwargs.pop('w', 0.01) # 0.25
        # Overdriven power for each KID
        overdriven = kwargs.pop('overdriven', [])
        # Create project folder
        create_folder = kwargs.pop('create_folder', True)
        # Expected KIDs
        expected = kwargs.pop('expected', None)
        # Load saved project
        load_saved = kwargs.pop('load_saved', False)
        # Additional input attenuation
        add_in_att = kwargs.pop('add_in_att', 0)
        # Additional output attenuation
        add_out_att = kwargs.pop('add_out_att', 0)
        # Material
        material = kwargs.pop('material', 'Al')
        # Absorber dimensions
        #dims = kwargs.pop('dims', [1,1,1])
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

        if not diry is None:
            
            if not load_saved:

                foldername = diry.split('/')[-2]
                self.date, self.data_type, self.meas, self.sample = self._get_meas_char_from_foldername(foldername)
                
                # Loading data base
                msg('Loading directory...', 'info')
                self.data = self.load_data(diry, only_vna=only_vna, expected=expected, w=w)
                msg('Done :)', 'ok')
                
            else:
                data, diry, setup = self.load_proj(BACKUP_FILE)
                self.data = data
                self.date, self.data_type, self.meas, self.sample = setup

        self.data_diry = diry

        # Create working directory
        if create_folder:
            self._create_diry(self.work_dir+self.project_name)
            self._create_diry(self.work_dir+self.project_name+'/fit_res_results')
            self._create_diry(self.work_dir+self.project_name+'/fit_res_results/vna')
            self._create_diry(self.work_dir+self.project_name+'/fit_res_results/homodyne')
            self._create_diry(self.work_dir+self.project_name+'/fit_res_results/summary_plots')

            self._create_diry(self.work_dir+self.project_name+'/fit_res_dict')
            self._create_diry(self.work_dir+self.project_name+'/fit_res_dict/vna')
            self._create_diry(self.work_dir+self.project_name+'/fit_res_dict/homodyne')
            self._create_diry(self.work_dir+self.project_name+'/fit_res_dict/responsivity')
            self._create_diry(self.work_dir+self.project_name+'/fit_res_dict/nep')

            self._create_diry(self.work_dir+self.project_name+'/fit_psd_results')
            self._create_diry(self.work_dir+self.project_name+'/fit_psd_dict')

            self._create_diry(self.work_dir+self.project_name+'/backup_data')

        self.add_in_atten = add_in_att
        self.add_out_atten = add_out_att

        self.found_kids = []

        # Material properties
        self.material = material
        #self.dims = dims
        self.Delta = 1

        self.od_cols = 5

        self._edit_mode = False
        self._range_flag = False
        
        self._range = [0, 0]

        self._xc, self._yc, self._theta = 0, 0, 0
        self.Qs_derot, self.Is_derot = 0, 0
        self.kid_mdl = None

        self.phase_ref, self.phase_ref_high = 0, 0

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

        if directories is None:
            print ('Directory doesnt exist')
            return -1

        elif len(directories) == 0:
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
                                (ts_low_off, I_low_off, Q_low_off, hr_low_off), (ts_high_off, I_high_off, Q_high_off, hr_high_off), (f_s21_high, s21_high) = get_homodyne_data(full_path)

                                # Frequency sweeps from homodyne.
                                data['ts'][kid][temp][atten]['f'] = f_s21
                                data['ts'][kid][temp][atten]['s21'] = s21
                                # High resolution sweep
                                data['ts'][kid][temp][atten]['f_high'] = f_s21_high
                                data['ts'][kid][temp][atten]['s21_high'] = s21_high
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
                                # Sweep header.
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
                            hdul.close()
                            data['vna']['full'][temp][atten]['header'][n_sample] = hdr

                        elif vna_type == 'seg':
                            # Extract data by KID
                            # ------------------------------------------------
                            n_kids = find_kids_segmented_vna(f, thresh=1e4, exp=expected)

                            # Split data per KID
                            for kid in range(len(n_kids)-1):
                                # Get detector
                                from_idx = n_kids[kid]+1
                                to_idx = n_kids[kid+1]

                                # Get individual detector
                                f_kid = f[from_idx:to_idx]
                                s21_kid = s21[from_idx:to_idx]

                                f_kid = f_kid[int(w*len(f_kid)):len(f_kid)-int(w*len(f_kid))]
                                s21_kid = s21_kid[int(w*len(s21_kid)):len(s21_kid)-int(w*len(s21_kid))]

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
                                hdul.close()
                                data['vna'][str_kid][temp][atten]['header'][n_sample] = hdr

        return data

    def save_proj(self, filename=None):
        """
        Save analysis in a numpy file.
        Parameters
        ----------
        filename : string
            Saved data filename.
        ----------
        """

        if filename is None:
            filename = BACKUP_FILE

        msg('Saving project: '+filename, 'info')
        np.save(self.work_dir+self.project_name+'/backup_data/'+filename, self.data)
        np.save(self.work_dir+self.project_name+'/backup_data/data_diry.npy', self.data_diry)
        np.save(self.work_dir+self.project_name+'/backup_data/data_setup.npy', [self.date, \
                                                                                self.data_type, \
                                                                                self.meas, \
                                                                                self.sample])
        msg('Done :)', 'ok')

    def load_proj(self, filename=None):
        """
        Load a saved project.
        Parameters
        ----------
        filename : string
            Loaded data filename.
        ----------
        """

        if filename is None:
            filename = BACKUP_FILE

        msg('Loading project: '+filename, 'info')
        data = np.load(self.work_dir+self.project_name+'/backup_data/'+filename+'.npy', allow_pickle=True).item()
        data_diry = np.load(self.work_dir+self.project_name+'/backup_data/data_diry.npy', allow_pickle=True).item()
        setup = np.load(self.work_dir+self.project_name+'/backup_data/data_setup.npy', allow_pickle=True)
        msg('Done :)', 'ok')

        return data, data_diry, setup

    def fit_vna_resonators(self, kid=None, temp=None, atten=None, sample=0, complete=True, n=3.5, **kwargs):
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
        # Overwrite ?
        overwrite = kwargs.pop('overwrite', False)
        # Verbose
        verbose = kwargs.pop('verbose', False)
        # ----------------------------------------------

        jobs = []
        kids = self._get_kids_to_sweep(kid)
        for kid in kids:
            msg(kid, 'info')

            temps = self._get_temps_to_sweep(temp, kid)
            for temp in temps:
                msg(temp, 'info')

                attens = self._get_atten_to_sweep(atten, temp, kid)
                for atten in attens:
                    msg(atten, 'info')
                    msg('Fit '+kid, 'info')

                    flag_do_it = False
                    if not complete:
                        flag_do_it = True
                    elif complete and (not ('fit' in self.data['vna'][kid][temp][atten])):
                        flag_do_it = True
                    elif not sample in self.data['vna'][kid][temp][atten]['fit'].keys():
                        flag_do_it = True

                    if flag_do_it or overwrite:
                        try:
                            f_vna, s21_vna = self.data['vna'][kid][temp][atten]['data'][sample]

                            p = Process(target=self._fit_res_join, args=(kid, temp, atten, f_vna, s21_vna, n, tau ))
                            jobs.append(p)
                            p.start()

                        except KeyError as err:
                            msg('Data not found.\n'+str(err), 'fail')

                        except Exception as e:
                            msg('Fit is taking too much time.\n'+str(e), 'fail')
                    else:
                        print('++++++++++++++++++++++')
                        print(kid+'-'+temp+'-'+atten+'-'+str(sample)+' done')

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

                Qi_err = np.sqrt( (e1*Qc_err)**2 + (e2*Qr_err)**2 )

                msg('Qi_error: '+str(Qi_err), 'info')
                self.data['vna'][kid][temp][atten]['fit'][sample]['Qi_err'] = Qi_err

            except:
                msg('Error getting Qi-error', 'warn')

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

                            values = []
                            for sample in samples:
                                if isinstance(sample, int):

                                    value = self.data['vna'][kid][tmp][att]['fit'][sample][item]
                                    values.append(value)

                            self.data['vna'][kid][tmp][att]['fit'][item] = np.mean(values)
                            self.data['vna'][kid][tmp][att]['fit'][item+'_err'] = np.std(values)

                    except:
                        print('No fit in: ', kid, tmp, att)

    def find_overdriven_atts(self, temp, sample=0, thresh=0.7):
        """
        Find the pre-overdriven attenuations given the fit results + manual selection.
        Parameters
        ----------
        temp : int, string
            Temperature. If 'None' it will take all the temperatures.
        sample : int
            Sample number. If 'None' take all the samples/repeats.
        thresh : float
            Non-linearity threshold to define an overdriven state.
        ----------
        """
        ioff()
        
        # Select all the KIDs
        kids = self._get_kids_to_sweep(None)
        # Temporal overdriven atts
        self.temp_att = np.zeros_like(kids, dtype=float)

        cnt = 0
        fig_cnt = -1
        tot_cnt = 0
        self.n_fig_od = 0

        text_color = 'white'
        box_color = 'purple'

        for k, kid in enumerate(kids):
            # Select all the attenuations
            tmp = self._get_temps_to_sweep(temp, kid)[0]

            catt = 0
            attens = self._get_atten_to_sweep(None, tmp, kid)
            kid_non = np.zeros_like(attens, dtype=float)
            for a, att in enumerate(attens):
                try:
                    # Check the non-linearity
                    kid_non[a] = self.data['vna'][kid][tmp][att]['fit'][sample]['non']
                except:
                    catt += 1

            if catt < len(attens):
                idx = len(kid_non) - np.where(kid_non[::-1]>thresh)[0]
                if len(idx) > 0:
                    idx = idx[0]
                else:
                    idx = 0
                text_size = 14
            else:
                idx = int(len(attens)/2)
                self.od_cols = len(attens)
                text_size = 9

            if k%6 == 0:
                if fig_cnt >= 0:
                    self._create_od_fig(fig, ax)

                fig, ax = subplots(6, self.od_cols)
                subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.07, hspace=0.0, wspace=0.0)
                            
                self.over_atts_mtx = np.zeros((6, self.od_cols))
                self.over_atts_mask = np.zeros((6, self.od_cols), dtype=bool)

                fig_cnt += 1
                self.n_fig_od = fig_cnt
                cnt = 0

            # Assign overdriven attenuations
            self.temp_att[k] = float(attens[idx][1:])

            for i in np.arange(self.od_cols):
                idx_map = i + idx - int(self.od_cols/2)

                ii = cnt%self.od_cols
                jj = int(cnt/self.od_cols)

                if idx_map >= 0 and idx_map<len(attens):
                    f, s21 = self.data['vna'][kid][tmp][attens[idx_map]]['data'][sample]
                    ax[jj,ii].plot(s21.real, s21.imag, 'r.-')
                    ax[jj,ii].tick_params(axis='x', labelsize=text_size)
                    ax[jj,ii].tick_params(axis='y', labelsize=text_size)

                    try:
                        f_fit = self.data['vna'][kid][tmp][attens[idx_map]]['fit'][sample]['freq_data']
                        fit_s21 = self.data['vna'][kid][tmp][attens[idx_map]]['fit'][sample]['fit_data']
                        ax[jj,ii].plot(fit_s21.real, fit_s21.imag, 'k')
                    except:
                        pass

                    ax[jj,ii].axis('equal')

                    if i == int(self.od_cols/2):
                        ax[jj,ii].patch.set_facecolor('green')
                    else:
                        ax[jj,ii].patch.set_facecolor('blue')

                    ax[jj,ii].patch.set_alpha(0.2)

                    ax[jj,ii].text(0.2, 0.1, attens[idx_map] + ' dB', \
                                    {'fontsize': text_size, 'color':text_color}, \
                                    bbox=dict(facecolor=box_color, alpha=0.95), \
                                    transform=ax[jj,ii].transAxes)
                    self.over_atts_mtx[jj,ii] = float(attens[idx_map][1:])
                    self.over_atts_mask[jj,ii] = True

                if jj == self.od_cols or tot_cnt == len(kids)-1:
                    ax[jj,ii].set_xlabel("I [V]")
                if cnt%self.od_cols == 0:
                    ax[jj,ii].set_ylabel(kid+"\nQ [V]")

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
            for j in range(self.od_cols):
                if event.inaxes == self.ax_od[i, j]:
                    if self.over_atts_mask[i, j]:
                        self.update_overdriven_plot(i, j)
                        self.temp_att[self.n_fig_od*6+i] = self.over_atts_mtx[i, j]

    def update_overdriven_plot(self, i, j):
        """
        Update overdriven plot.
        """
        for m in range(self.od_cols):
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

                msg('Undoing changes', 'info')

    def find_kids(self, f, s21, down_factor=35, baseline_params=(501, 5), Qr_lim=[1500, 150000], Qc_lim=[1000, 150000], inter=True):
        """
        Find resonators from the VNA sweep.
        Parameters
        ----------
        f : array
            Frequency array.
        s21 : array
            S21 array.
        down_factor : int
            Downsampling factor to smooth data and extract baseline.
        baseline_params : tuple
            Savinsky-Golay filter:
                idx 0: Number of points.
                idx 1: Order.
        Qr_lim : list
            Qr upper(1) and lower(0) limits. 
        Qc_lim : list
            Qc upper(1) and lower(0) limits.
        inter : bool
            Interactive mode. 
        ----------
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

            if range_a < 0:
                range_a = 0
            if range_b > len(f):
                range_b = len(f)-1

            # Take a sample
            fsm = f[range_a:range_b]
            s21_sm = s21[range_a:range_b]

            ar, ai, Qr, fr, Qc, phi = coarse_fit(fsm, s21_sm, tau=50e-9)

            if (Qr > Qr_lim[0] and Qr < Qr_lim[1]) and (Qc > Qc_lim[0] and Qc < Qc_lim[1]):
                flags.append(True)
            else:
                flags.append(False)

            nw_peaks.append(peak)

        # Interactive mode
        if inter:
            self.interactive_mode_find_kids(f, s21, nw_peaks, flags)
        else:
            sel_peaks = []
            for p, peak in enumerate(nw_peaks):
                if flags[p]:
                    sel_peaks.append(f[peak])
            self.found_kids = sel_peaks

    def interactive_mode_find_kids(self, f, s21, peaks, flags):
        """
        Interactive mode to clean psd data.
        """

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

        # Initial plot
        self.flag_update =  True
        self.update_plot_find_kids(self._freq, self._peaks, self._flags)

        self._onclick_xy = self._fig.canvas.mpl_connect('button_press_event', self._onclick_find_kids)
        self._keyboard = self._fig.canvas.mpl_connect('key_press_event', self._key_pressed_find_kids)

        show()

    def update_plot_find_kids(self, freq, peaks, flags):
        """
        Update finding KIDs plot.
        """
        # Plot original and fit
        if self.flag_update:
            self._ax.plot(self._freq, 20*np.log10(np.abs(self._s21)), 'b')

            instructions_text = "x : save and quit\nq : quit\nd : default\nu : update\ne : edit mode\n  a : add resonances\n  r : remove resonances\n    w : select range"
            self._ax.text(0.03, 0.05, instructions_text, fontsize=13, transform=self._ax.transAxes)

        for p, peak in enumerate(peaks):
            if flags[p]:
                self._ax.axvline(freq[peak], color='r')
            else:
                self._ax.axvline(freq[peak], color='k', lw=0.35)

        if self._mode:
            self._ax.patch.set_facecolor('green')
            self._ax.patch.set_alpha(0.2)
        else:
            self._ax.patch.set_facecolor('red')
            self._ax.patch.set_alpha(0.2)

        self._ax.set_xlabel(r'Frequency [Hz]')
        self._ax.set_ylabel(r'S21 [dB]')

        if self._edit_mode:
            if self._mode:
                str_mode = 'Add resonances'
            else:
                str_mode = 'Remove resonances'
            self._ax.set_title('EDITION MODE. '+str_mode)            
        else:
            self._ax.set_title('VISUALISATION MODE')

        # Add a text box
        n_kids = np.sum(self._flags)
        summary_text = f"Resonators : {n_kids}"
        self._ax.text(0.03, 0.95, summary_text, fontsize=16, transform=self._ax.transAxes)

    def _key_pressed_find_kids(self, event, thresh=5e4):
        """
        Keyboard event to save/discard line fitting changes
        Keys:
            'x' : save and quit
            'q' : quit
            'd' : go back to default settings
            'u' : update
            'a' : add resonances
            'r' : remove resonances
            'e' : edit plot
            'z' : add/remove a tonelist where the cursor is
        """

        sys.stdout.flush()

        if event.key in ['x', 'q', 'd', 'u', 'a', 'r', 'e', 'w', 'z']:

            if event.key == 'x':

                self.flag_update = False

                self._fig.canvas.mpl_disconnect(self._onclick_xy)
                self._fig.canvas.mpl_disconnect(self._key_pressed_find_kids)
                close(self._fig)

                # Save data
                sel_peaks = []
                for p, peak in enumerate(self._peaks):
                    if self._flags[p]:
                        sel_peaks.append(self._freq[peak])
                self.found_kids = sel_peaks

                # C R E A T E   T O N E S L I S T
                # --------------------------------
                sort_tones = np.sort(self.found_kids)

                tonelist_name = 'Toneslist-'+self.project_name+'.txt'
                with open(tonelist_name, 'w') as file:
                    file.write('Name\tFreq\tOffset att\tAll\tNone\n')
                    for kid in range(len(sort_tones)):
                        kid_name = 'K'+str(kid).zfill(3)
                        file.write(f'{kid_name}\t{sort_tones[kid]:.0f}\t0\t1\t0\n')
                
                msg('Tonelist file save as: '+tonelist_name, 'info')
                msg('Changes saved!', 'info')

            elif event.key == 'u':
                cla()
                self.flag_update = True
                self.update_plot_find_kids(self._freq, self._peaks, self._flags)
                self._fig.canvas.draw_idle()

            elif event.key == 'a':
                self._mode = True
                self.flag_update = False
                self._range_flag = False
                #cla()
                self.update_plot_find_kids(self._freq, self._peaks, self._flags)
                self._fig.canvas.draw_idle()

            elif event.key == 'r':
                self._mode = False
                self.flag_update = False
                self._range_flag = False
                #cla()
                self.update_plot_find_kids(self._freq, self._peaks, self._flags)
                self._fig.canvas.draw_idle()
            
            elif event.key == 'w':
                self.flag_update = False
                self._range_flag = not self._range_flag

            elif event.key == 'z':
                self.flag_update = False
                self._range_flag = False

                upper_limit = np.max(self._range)
                lower_limit = np.min(self._range)

                for p, peak in enumerate(self._peaks):
                    fp = self._freq[peak]
                    if (fp > lower_limit) and (fp < upper_limit):
                        self._flags[p] = False

                #cla()
                self.update_plot_find_kids(self._freq, self._peaks, self._flags)
                self._fig.canvas.draw_idle()              

            elif event.key == 'e':
                self._edit_mode = not self._edit_mode
                if self._edit_mode:
                    if self._mode:
                        str_mode = 'Add resonances'
                    else:
                        str_mode = 'Remove resonances'
                    self._ax.set_title('EDITION MODE. '+str_mode)
                    self._ax.tick_params(color='blue', labelcolor='blue')
                    for spine in self._ax.spines.values():
                        spine.set_edgecolor('blue')
                else:
                    self._range_flag = False
                    self._ax.set_title('VISUALISATION MODE')
                    self._ax.tick_params(color='black', labelcolor='black')
                    for spine in self._ax.spines.values():
                        spine.set_edgecolor('black')
                self._fig.canvas.draw_idle()

            elif event.key == 'd':
                self._freq = self._freq_backup
                self._s21 = self._s21_backup
                self._peaks = self._peaks_backup
                self._flags = self._flags_backup
                cla()
                self.update_plot_find_kids(self._freq, self._peaks, self._flags)
                self._fig.canvas.draw_idle()
                msg('Undoing changes', 'info')

    def _onclick_find_kids(self, event, thresh=5e4):
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
                    if self._mode:
                        if np.abs(ix - self._freq[peak]) < 1e3:
                            self._flags[p] = True 
                            flag_done = False
                            break
                    else:
                        if np.abs(ix - self._freq[peak]) < thresh:
                            self._flags[p] = False
                            flag_done = False
                            break

                if self._edit_mode:
                    if flag_done and self._mode:
                        ix_idx = np.where(ix<self._freq)[0][0]
                        self._peaks.append(ix_idx)
                        self._flags.append(True)

                    if self._mode:
                        self._ax.axvline(ix, color='g')
                    else:
                        if self._range_flag:
                            self._ax.axvline(ix, color='b', linestyle='dashed', linewidth=1.5)
                            self._range[1] = self._range[0]
                            self._range[0] = ix
                        else:
                            #cla()
                            #self.update_plot_find_kids(self._freq, self._s21, self._peaks, self._flags)
                            self._ax.axvline(ix, color='m', linestyle='dashed')

                    self._fig.canvas.draw_idle()

    def split_continuous_by_kid(self, temp=None, atten=None, lws=6, Qr=1000):
        """
        Divide the VNA by the found detectors. Under construction...
        """

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

    def get_all_psd(self, kid, temp, atten, **kwargs):
        """
        Get the PSD for all the kid/temp/atten defined.
        Parameters
        ----------
        kid : int, string
            KID ID. If 'None' it will take all the resonators.
        temp : int, string
            Temperature. If 'None' it will take all the temperatures, whether base
            temperatura or Blackbody temperature.
        atten : string
            Attenuation. If 'None' it will select all the attenuations.
        ----------
        """
        # Key arguments
        # ----------------------------------------------
        # Timestreams to ignore
        ignore = kwargs.pop('ignore', [[0,1], [0,1]])
        # Fit PSD?
        fit_psd = kwargs.pop('fit_psd', True)
        # PSD type
        psd_type = kwargs.pop('psd_type', 'df')
        # Plot fit results?
        plot_fit = kwargs.pop('plot_fit', True)
        # Reduced number of points
        n_pts = kwargs.pop('n_pts', 500)
        # Interactive mode
        inter = kwargs.pop('inter', True)
        # Savgol smoothing values
        smooth_params = kwargs.pop('smooth_params', [21, 3])
        # ----------------------------------------------

        kids = self._get_kids_to_sweep(kid, mode='ts')
        for kid in kids:
            msg(kid, 'info')

            temps = self._get_temps_to_sweep(temp, kid, mode='ts')
            for tmp in temps:

                attens = self._get_atten_to_sweep(atten, tmp, kid, mode='ts')
                for att in attens:
                    self.get_psd_on_off(kid, tmp, att, ignore=ignore, fit=fit_psd, plot_fit=plot_fit, \
                                        n_pts=n_pts, inter=inter, smooth_params=smooth_params, psd_type=psd_type)

    def get_responsivity(self, kid, tmt, **kwargs):
        """
        Get responsivity
        Parameters
        ----------
        kid : int, string
            KID IDs. If 'None' it will take all the resonators.
        temp_conv(opt) : string
            Convert temperature to a defined parameter.
                'power': get power in W.
                'Nqp' : get number of quasiparticles.
        var(opt) : string
            Parameter to evaluate the responsivity:
            'fr'(default): Resonance frequency
            'phase': Resonator phase 
            'Qr': Total quality factor.
            'Qc': Coupling quality factor.
            'Qi': Intrinsec quality factor.
        sample(opt) : int
            Sample number. If 'None' take all the samples/repeats.
        dims(opt) : list
            Device dimensions in um to get the Volume in um³.
        nu(opt) : float
            Bandwidth. Only use to get power if pwr_method is 'bandwidth'.
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
        pwr_method(opt) : string
            Method to get the power if temp_conv = 'power'. 
            'fts' : from a spectrum defined in diry_fts.
            'bandwidth' : from a defined bandwidth.
        res_method(opt) : string
            Method to get responsivity:
            'grad' : from the gradient
            'fit' : fitting the curve aP**b.
        plot_res(opt) : bool
            Plot the responsivity as var vs power/BB/Nqp.
        nqp_fit_pts(opt) : int
            Number of points to fit the Nqp counting from the end. 
        ----------
        """

        # Key arguments
        # ----------------------------------------------
        # Convert temperature to a defined parameter.
        temp_conv = kwargs.pop('temp_conv', 'Nqp')
        # Parameter to evaluate the responsivity.
        var = kwargs.pop('var', 'fr')
        # Sample number.
        sample = kwargs.pop('sample', 0)
        # Detector dimensions.
        dims = kwargs.pop('dims', [1,1,1])
        self.dims = dims
        # Bandwidth.
        nu = kwargs.pop('nu', 20e9)
        # List of detectors to flag.
        flag_kid = kwargs.pop('flag_kid', [])
        # Customise the plots.
        custom = kwargs.pop('custom', None)
        # Data source.
        data_source = kwargs.pop('data_source', 'vna')
        # Folder that contains the FTS measurements to convert BB data to power.
        diry_fts = kwargs.pop('diry_fts', "")
        # Parameter from fit.
        from_fit = kwargs.pop('from_fit', False)
        # Method to get the resonance frequency 'fr'.
        method = kwargs.pop('method', 'min')
        #  Method to get responsivity.
        res_method = kwargs.pop('res_method', 'grad')
        #  Method to get power.
        pwr_method = kwargs.pop('pwr_method', 'bandwidth')
        # Number of points to fit the Nqp counting from the end.
        nqp_fit_pts = kwargs.pop('nqp_fit_pts', -4) #-4
        # Plot responsivity as var vs power/BB/Nqp
        plot_res = kwargs.pop('plot_res', True)
        # Smooth transmission?
        smooth = kwargs.pop('smooth', False)
        # Savgol smoothing values
        smooth_params = kwargs.pop('smooth_params', [7, 3])
        # ----------------------------------------------

        ioff()

        #print('-------------->', nqp_fit_pts)

        if from_fit == False and method == 'min':
            if not var in ['fr', 'Qi']:
                msg("Only var = 'fr' is valid under these conditions", "fail")
                return

        # Select the pre-overdriven attenuation.
        atten = self.overdriven

        if self.data_type.lower() == 'dark':
            temp_field = 'SAMPLETE'
        elif self.data_type.lower() == 'blackbody':
            temp_field = 'BLACKBOD'

        kids = self._get_kids_to_sweep(kid)

        xg = int(np.sqrt(len(kids)))
        yg = int(len(kids)/xg)

        tot_kids = len(self._get_kids_to_sweep(None))
        if tmt == None:
            tot_temps = len(self._get_temps_to_sweep(None, kids[0]))
        else:
            tot_temps = len(tmt)

        S = np.zeros((tot_kids, tot_temps))
        pwrs = np.zeros((tot_kids, tot_temps))
        tmps = np.zeros((tot_kids, tot_temps))

        if plot_res:
            lstyle_pointer = 0

            fig, ax = subplots(1,1, figsize=(15,9))
            subplots_adjust(left=0.110, right=0.99, top=0.97, bottom=0.07, hspace=0.0, wspace=0.0)

        cnt = 0
        for kid in kids:
            
            base_f0 = 0
        
            k = int(kid[1:])
            msg(kid, 'info')

            i = k%yg
            j = int(k/yg)

            temps = self._get_temps_to_sweep(tmt, kid)
            xs = np.zeros_like(temps, dtype=float)
            real_temp = np.zeros_like(temps, dtype=float)

            # T O T A L   V O L U M E
            # ----------------------------
            if isinstance(self.dims, dict):
                dims_list = self.dims[kid]
            elif isinstance(self.dims, list):
                dims_list = self.dims

            print('D I M E N S I O N S')
            print(kid)
            print(dims_list)
            print('-------------------')

            V = float(dims_list[0])*float(dims_list[1])*float(dims_list[2]) 

            # -----------------------------

            for t, tm in enumerate(temps):
                
                att = self._get_atten_to_sweep(atten[k], tm, kid)[0]
                
                try:

                    if False:    
                        xs[t] = np.nan

                    else:

                        # F I T
                        # -------------
                        if from_fit:

                            if data_source == 'vna':
                                
                                f, s21 = self.data['vna'][kid][tm][att]['data'][sample]

                                if (not 'fit' in self.data['vna'][kid][tm][att]) or kid in flag_kid:

                                    if var in ['fr', 'phase']:
                                        msg(kid+'-'+tm+'-'+att+'. Fit data not available, using the min criteria.', 'info')

                                        #s21_mag = 20*np.log10(np.abs(s21))

                                        if smooth:
                                            I_sm = savgol_filter(s21.real, *smooth_params)
                                            Q_sm = savgol_filter(s21.imag, *smooth_params)
                                            #s21_mag = savgol_filter(s21_mag, *smooth_params)
                                        else:
                                            I_sm = s21.real
                                            Q_sm = s21.real

                                        s21_mag = np.sqrt(I_sm**2 + Q_sm**2)

                                        f_idx = np.argmin(s21_mag)
                                        x = f[f_idx]

                                        if var == 'phase':
                                            # Get f0 at base temperature
                                            if base_f0 == 0:
                                                base_f0 = np.copy(x)

                                            f0_ref = base_f0

                                            figure('IQ_derot-vs-F0', figsize=(16,12))
                                            x = self.get_phase_shift(f_idx, x, s21, f, f0_ref, f0_thresh=5e4, label=tm)

                                    else:
                                        x = None
                                        
                                else:
                                
                                    if var == 'phase':
                                        print('P H A S E', tm)
                                        x = self.data['vna'][kid][tm][att]['fit'][0]['fr']
                                        f_idx = np.where(f>=x)[0][0]
                                        
                                        # Get f0 at base temperature
                                        if base_f0 == 0:
                                            base_f0 = np.copy(x)

                                        f0_ref = base_f0

                                        figure('IQ_derot-vs-F0', figsize=(16,12))
                                        x = self.get_phase_shift(f_idx, x, s21, f, f0_ref, f0_thresh=5e4, label=tm)

                                    else:
                                        x = self.data['vna'][kid][tm][att]['fit'][0][var]

                            elif data_source == 'homo':

                                if var in ['fr', 'phase']:

                                    if var == 'phase':

                                        f = self.data['ts'][kid][tm][att]['f']
                                        s21 = self.data['ts'][kid][tm][att]['s21']
                                        
                                        x = self.data['ts'][kid][tm][att]['fit'][0]['fr']
                                        f_idx = np.where(f>=x)[0][0]
                                        
                                        # Get f0 at base temperature
                                        if base_f0 == 0:
                                            base_f0 = np.copy(x)

                                        f0_ref = base_f0
                                        
                                        figure('IQ_derot-vs-F0', figsize=(16,12))
                                        x = self.get_phase_shift(f_idx, x, s21, f, f0_ref, f0_thresh=5e4, label=tm)

                                    else:
                                        x = self.data['ts'][kid][tm][att]['fit'][0][var]

                                else:
                                    x = self.data['ts'][kid][tm][att]['fit_psd']['params'][var]

                        # N O    F I T
                        # -------------
                        else:

                            if var in ['fr', 'phase']:

                                if method == 'min':

                                    f, s21 = self.data['vna'][kid][tm][att]['data'][sample]
                                    s21_mag = 20*np.log10(np.abs(s21))

                                    if smooth:
                                        figure('s21')
                                        plot(f, 20*np.log10(np.abs(s21)))
                                        #if kid in ['K000', 'K001', 'K002', 'K003', 'K004', 'K005', 'K006']:
                                        #    s21_mag = savgol_filter(s21_mag, 31, 3)
                                        #else:
                                        #    s21_mag = savgol_filter(s21_mag, *smooth_params)
                                        
                                        I_sm = savgol_filter(s21.real, *smooth_params)
                                        Q_sm = savgol_filter(s21.imag, *smooth_params)

                                        #s21_mag = savgol_filter(s21_mag, *smooth_params)
                                        
                                        s21_mag = np.sqrt(I_sm**2 + Q_sm**2)
                                        plot(f, 20*np.log10(s21_mag), 'k--')

                                    f_idx = np.argmin(s21_mag)
                                    x = f[f_idx]
                                    #axvline(x, color='r')

                                    if var == 'phase':
                                        # Get f0 at base temperature
                                        if base_f0 == 0:
                                            base_f0 = np.copy(x)

                                        f0_ref = base_f0

                                        figure('IQ_derot-vs-F0', figsize=(16,12))
                                        x = self.get_phase_shift(f_idx, x, s21, f, f0_ref, f0_thresh=5e4, label=tm)

                        xs[t] = x
                
                except Exception as e:
                    msg('Reading responsivity variable.\n'+str(e), 'warn')

                try:
                    real_temp[t] = float(self.data['vna'][kid][tm][att]['header'][0][temp_field])
                except:
                    real_temp[t] = np.nan

            if var == 'phase':
                savefig(self.work_dir+self.project_name+'/fit_res_dict/responsivity/'+kid+'-derotated-IQ-F0.png')
                close('IQ_derot-vs-F0')

            try:
                power = []
                all_temps = [] 
                for rt in real_temp:
                    print('***********************************************')
                    msg('Temperature[K]: '+str(rt), 'info')
                    if temp_conv == 'power':
                        try:
                            if pwr_method == 'fts':
                                
                                # This is for CPW2-Chip13
                                #if k < 8:
                                #    fc = 150e9
                                #else:
                                #    fc = 100e9

                                # This is for ANL FT163 Chip 4
                                #if k in [1, 2, 3, 6, 7, 0, 4, 8, 13]:
                                #    fc = 150e9
                                #elif k in [9, 11, 12, 14, 15]:
                                #    fc = 97e9

                                # This is for SOUK-TG-Al
                                #if k in [0, 1, 2]:
                                #    fc = 118e9
                                #elif k in [3,4,5,6,7,8,9,10,11,12]:
                                #    fc = 104e9

                                print('From FTS')
                                p = get_power_from_FTS(diry_fts, k, rt, modes=1) #, f0=fc)

                            elif pwr_method == 'bandwidth':
                                print('From bandwidth')
                                print(f'BW [GHz]: {nu}')
                                p = bb2pwr(rt, nu)
                        except:
                            p = np.nan

                        msg('Power[pW]: '+str(p*1e12), 'info')
                        power.append(p)

                    elif temp_conv == 'Nqp':
                        Delta = get_Delta(Tcs[self.material])
                        self.Delta = Delta

                        nqp = get_nqp(N0s[self.material], rt, Delta)
                        Nqp = nqp * V
                        power.append(Nqp)

                        msg('Material: '+self.material, 'info')
                        msg('Volume [um³]: '+str(V), 'info')
                        msg('Energy gap: '+str(Delta), 'info')
                        msg('Nqp: '+str(Nqp), 'info')

                    elif temp_conv == 'temp':
                        msg('Temperature: '+str(temp_conv), 'info')
                        power.append(rt)
                    
                    all_temps.append(rt)

                power = np.array(power)
                all_temps = np.array(all_temps)

                # Get the responsivity
                # ------------------------------------------------------------------
                if temp_conv == 'power':
                    
                    if res_method == "fit":
                        print('F I T   R E S   C U R V E')
                        popt, pcov = curve_fit(f0_vs_pwr_model, power, xs, p0=[1e4, -0.5])
                        a, b = popt
                        dF0_dP = a*b*power**(b-1)

                        """
                        b0 = f0_vs_pwr_model(power, a, b) - power*dF0_dP
                        P_fit = np.linspace(power[0], power[-2], 1000)
                        figure(kid)
                        plot(power, f0_vs_pwr_model(power, *popt), 'rs-', label='fit')
                        plot(power, xs, 'bs')
                        for f in range(len(dF0_dP)):
                            print('plotting')
                            plot(P_fit, P_fit*dF0_dP[f] + b0[f], 'k-')
                        show()
                        """

                    elif res_method == "grad":
                        print('G R A D I E N T')
                        dF0_dP = np.gradient(xs, power)

                    S[k] = dF0_dP

                elif temp_conv == 'Nqp':

                    dF0_dNqp, b = np.polyfit(power[nqp_fit_pts:], xs[nqp_fit_pts:], 1)
                    
                    Nqps_fit = np.linspace(power[0], power[-1], 1000)
                    plot(power, xs, 'rs-')
                    plot(Nqps_fit, Nqps_fit*dF0_dNqp + b, 'k')
                    
                    S[k] = dF0_dNqp

                elif temp_conv == 'temp':

                    S[k] = xs

                pwrs[k] = power
                tmps[k] = all_temps

            except Exception as e:
                msg(str(e), 'warn')

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

                if cnt%10 == 0 and cnt > 0:
                    lstyle_pointer += 1

                if var == 'fr':
                    xs_plot = (xs - xs[0])/xs[0]
                    ax.set_ylabel('ffs [ppm]', fontsize=18, weight='bold')
                else:
                    xs_plot = xs
                    label_text = var
                    if var == 'phase':
                        label_text += ' [rad]'
                    ax.set_ylabel(label_text, fontsize=18, weight='bold')

                if not np.sum([np.isnan(i) for i in xs_plot]) == len(xs_plot):
                    
                    if var == 'fr':
                        ks = 1e6
                    else:
                        ks = 1

                    nw_power = []
                    nw_xs_plot = []
                    for c in range(len(xs_plot)):
                        if not np.isnan(xs_plot[c]):
                            nw_xs_plot.append(xs_plot[c])
                            nw_power.append(power[c])
                        else:
                            print('NAN!')

                    nw_xs_plot = np.array(nw_xs_plot)
                    nw_power = np.array(nw_power)

                    opt_eff = 1 #0.004

                    if not custom is None:
                        color = custom[0][k]
                        mk = custom[1][k]
                        lsty = custom[2][k] 
                        ax.plot(1e12*opt_eff*nw_power, ks*nw_xs_plot, label=kid, linestyle=lsty, marker=mk, color=color)
                    else:                      
                        ax.plot(1e12*opt_eff*nw_power, ks*nw_xs_plot, label=kid, linestyle=lstyle[lstyle_pointer], marker=lmarker[lstyle_pointer])

                    if temp_conv == 'power':
                        ax.set_xlabel('Power [pW]', fontsize=18, weight='bold')
                    elif temp_conv == 'Nqp':
                        ax.set_xlabel('Nqp', fontsize=18, weight='bold')
                    elif temp_conv == 'temp':
                        ax.set_xlabel('Temperature [K]', fontsize=18, weight='bold')

                    ax.legend(ncol=2)

            cnt += 1

        if plot_res:
            show()
            fig.savefig(self.work_dir+self.project_name+'/fit_res_dict/responsivity/responsivity-'+str(var)+'-'+temp_conv+'-'+self.data_type+'.png')

        # Save results
        np.save(self.work_dir+self.project_name+'/fit_res_dict/responsivity/responsivity-'+str(var)+'-'+temp_conv+'-'+self.data_type, S)
        print(S)
        np.save(self.work_dir+self.project_name+'/fit_res_dict/responsivity/responsivity-powers-'+str(var)+'-'+self.data_type, pwrs)
        np.save(self.work_dir+self.project_name+'/fit_res_dict/responsivity/responsivity-temps-'+str(var)+'-'+self.data_type, tmps)


    def get_phase_shift(self, f_idx, x, s21, f, f0_ref, f0_thresh=5e4, label='0.0'):
        """
        Get the phase shift in radians.
        """

        I0 = s21.real[f_idx]
        Q0 = s21.imag[f_idx]
        
        xc, yc, theta, Is_derot, Qs_derot, kid_mdl = get_rot_iq_circle(x, I0, Q0, f, s21.real, s21.imag, f, s21, f0_thresh=f0_thresh)

        print('R O T A C I O N')
        print(xc, yc, theta)

        f0_idx = np.where(f>=f0_ref)[0][0]

        I_f0 = Is_derot[f0_idx]
        Q_f0 = Qs_derot[f0_idx]

        plot(Is_derot, Qs_derot, label=label)
        plot(Is_derot[f0_idx], Qs_derot[f0_idx], 'r*')
        axis('equal')
        grid(True)
        legend()

        phase_f0 = np.arctan2(Q_f0, I_f0)

        if phase_f0 > np.pi/4:
            phase_f0 = phase_f0-2*np.pi

        return phase_f0

    def get_all_NEP(self, kid, temp, **kwargs):
        """
        Get all the NEP(Noise Equivalent Power).
        Parameters
        ----------
        kid : int, string
            KID ID. If 'None' it will take all the resonators.
        temp : int, string
            Temperature. If 'None' it will take all the temperatures.
        ----------
        """
        # Key arguments
        # ----------------------------------------------
        # Fixed frequencies
        fixed_freqs = kwargs.pop('fixed_freqs', [1, 10, 100])
        colors = ['r', 'g', 'b']
        # NEP at a given temp
        fixed_temp = kwargs.pop('fixed_temp', "B000")
        # Frequency width
        df = kwargs.pop('df', 0.5)
        # User defined temps
        dtemps = kwargs.pop('dtemps', None)
        # ----------------------------------------------

        NEPs = np.zeros((2, len(fixed_freqs)), dtype=float)
        kids = self._get_kids_to_sweep(kid, mode='ts')

        # Get responsivity
        # NOTE. The filenames have to chance.
        #S = np.load(self.work_dir+self.project_name+'/fit_res_dict/responsivity/responsivity-fr-Nqp-'+self.data_type+'.npy')
        S = np.load(self.work_dir+self.project_name+'/fit_res_dict/responsivity/responsivity-fr-power-'+self.data_type+'.npy')
        pwrs = np.load(self.work_dir+self.project_name+'/fit_res_dict/responsivity/responsivity-powers-fr-'+self.data_type+'.npy')

        ioff()
        #fig_nep_kids, ax_nep_kids = subplots(1, 1, figsize=(20,12))
        #subplots_adjust(left=0.110, right=0.99, top=0.97, bottom=0.07, hspace=0.0, wspace=0.0)

        for kid in kids:
            k = int(kid[1:])
            msg(kid, 'info')

            fig_kid, ax_kid = subplots(1, 1, figsize=(21,6))
            subplots_adjust(left=0.060, right=0.98, top=0.95, bottom=0.110, hspace=0.2, wspace=0.2)
            fig_neps, ax_neps = subplots(1, 1, figsize=(8,8))
            subplots_adjust(left=0.110, right=0.98, top=0.95, bottom=0.110, hspace=0.0, wspace=0.0)

            temps = self._get_temps_to_sweep(temp, kid, mode='ts')
            for t, tmp in enumerate(temps):
                
                avail_temps = sorted(list(self.data['vna'][kid].keys()))

                try:

                    att = self._get_atten_to_sweep(self.overdriven[k], tmp, kid, mode='ts')[0]

                    f_clean, psd_clean = self.data['ts'][kid][tmp][att]['psd']['clean']
                    f_clean_bin, psd_clean_bin = self.data['ts'][kid][tmp][att]['psd']['binned']['clean']

                    # Noise params
                    psd_params = self.data['ts'][kid][tmp][att]['fit_psd']['params']
                    tqp = psd_params['tau']

                    # Resonator params
                    if 'fit' in self.data['vna'][kid][tmp][att]:
                        f0 = self.data['vna'][kid][tmp][att]['fit']['fr']
                        Qr = self.data['vna'][kid][tmp][att]['fit']['Qr'] 
                    else:
                        msg('Fit for f0 and Qr are not available.\nf0 selected as the minimum of S21 magnitude.\nQr defined as zero, i.e. ring-down time will not be considered.', 'warn')
                        fd, s21_d = self.data['vna'][kid][tmp][att]['data'][0]
                        f0 = fd[np.nanargmin(s21_d)]
                        Qr = 0

                    #print('C H E C K   H E R E ! ! !')
                    tt = avail_temps.index(tmp)
                    #print(tmp, t, tt, k)

                    NEP, NEP_ncorr = self.get_NEP(f_clean, psd_clean, tqp, S[k][tt], Qr, f0)

                    # Get NEP from binning data
                    NEP_bin, NEP_bin_ncorr = self.get_NEP(f_clean_bin, psd_clean_bin, tqp, S[k][tt], Qr, f0)

                    for i, fx in enumerate(fixed_freqs):
                        idx_from = np.where(f_clean > fx-df*(i+1))[0][0]
                        idx_to = np.where(f_clean < fx+df*(i+1))[0][-1]
                        NEPs[0][i] = np.mean(NEP[idx_from:idx_to])
                        NEPs[1][i] = np.std(NEP[idx_from:idx_to])
                        
                        ax_neps.errorbar(pwrs[k][t], NEPs[0][i], yerr=NEPs[1][i], color=colors[i], marker='s', ecolor='k', capsize=2)
                        if t == 0:
                            ax_neps.plot(pwrs[k][t], NEPs[0][i], 's-', color=colors[i], label=str(fx)+' Hz')
                        else:
                            ax_neps.plot(pwrs[k][t], NEPs[0][i], 's-', color=colors[i])

                    if tmp == fixed_temp:
                        temp_NEP = NEP
                        temp_freq = f_clean

                    ax_kid.loglog(f_clean, NEP, label=tmp)

                    np.save(self.work_dir+self.project_name+'/fit_res_dict/nep/nep-bin-'+kid+'-'+tmp+'-'+att, [f_clean_bin, NEP_bin])
                    np.save(self.work_dir+self.project_name+'/fit_res_dict/nep/nep-bin-nocorr-'+kid+'-'+tmp+'-'+att, [f_clean_bin, NEP_bin_ncorr])
                    np.save(self.work_dir+self.project_name+'/fit_res_dict/nep/nep-nocorr-'+kid+'-'+tmp+'-'+att, [f_clean, NEP_ncorr])

                    np.save(self.work_dir+self.project_name+'/fit_res_dict/nep/nep-'+kid+'-'+tmp+'-'+att, [f_clean, NEP])
                    np.save(self.work_dir+self.project_name+'/fit_res_dict/nep/neps_pts-'+kid+'-'+tmp+'-'+att, NEPs)

                except Exception as e:
                    msg('Error getting the NEP for: '+kid+'-'+tmp, 'warn')
                    print(e)

            ax_kid.set_title(kid)
            ax_kid.set_xlabel('Frequency [MHz]')
            ax_kid.set_ylabel('NEP [W/sqrt(Hz)]')
            ax_kid.grid(True, which="both", ls="-")
            ax_kid.legend()

            ax_neps.set_title(kid)
            if self.data_type.lower() == 'blackbody': 
                ax_neps.set_xlabel('Power [W]')
            elif self.data_type.lower() == 'dark': 
                ax_neps.set_xlabel('Nqp')
            ax_neps.set_ylabel('NEP [W/sqrt(Hz)]')
            ax_neps.grid(True, which="both", ls="-")
            ax_neps.legend()

            """
            try:
                ax_nep_kids.loglog(temp_freq, temp_NEP, label=kid)
                ax_nep_kids.set_xlabel('Frequency [MHz]')
                ax_nep_kids.set_ylabel('NEP [W/sqrt(Hz)]')
            except:
                pass
            """
                
        np.save(self.work_dir+self.project_name+'/fit_res_dict/nep/nep_freqs', fixed_freqs)

        #ax_nep_kids.grid(True, which="both", ls="-")
        #ax_nep_kids.legend()
        
        show()


    def get_NEP(self, f, psd, tqp, S, Qr, f0, **kwargs):
        """
        Get the NEP.
        Parameters
        ----------
        f : array
            frequency
        psd : array
            Noise PSD
        tqp : float
            Quasiparticle lifetime
        S : float
            Responsivity
        Qr : float
            Quality factor
        f0 : float
            Resonance frequency
        ----------
        """
        # Key arguments
        # ----------------------------------------------
        # Detector efficiency
        eta = kwargs.pop('eta', 0.6)
        # ----------------------------------------------

        if self.data_type.lower() == 'dark':
            self.Delta = get_Delta(Tcs[self.material])
            #tqp = tqp/2
            NEP = np.sqrt(psd) * (( (eta*tqp/self.Delta)*(np.abs(S)) )**(-1)) * np.sqrt(1 + (2*np.pi*f*tqp)**2 ) * np.sqrt(1 + (2*np.pi*f*Qr/np.pi/f0)**2)
            NEP_no_corr = np.sqrt(psd) * (( (eta*tqp/self.Delta)*(np.abs(S)) )**(-1))
            print('Dark NEP')

        elif self.data_type.lower() == 'blackbody':
            NEP = np.sqrt(psd) * ( (np.abs(S))**(-1)) * np.sqrt(1 + (2*np.pi*f*tqp)**2 ) * np.sqrt(1 + (2*np.pi*f*Qr/np.pi/f0)**2)
            NEP_no_corr = np.sqrt(psd) * ( (np.abs(S))**(-1) )
            print('Optical NEP')

        print(f'-----------------------')
        print(f'tqp [us]: {1e6*tqp:.3f}')
        print(f'Responsivity: {S:.2f}')

        return NEP, NEP_no_corr

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
    
    def get_psd_on_off(self, kid, temp, atten, **kwargs):
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
        # Key arguments
        # ----------------------------------------------
        # Timestreams to ignore
        ignore = kwargs.pop('ignore', [[0,1], [0]])
        # Fit PSD?
        fit = kwargs.pop('fit', True)
        # PSD type
        psd_type = kwargs.pop('psd_type', 'df')
        # Plot fit results?
        plot_fit = kwargs.pop('plot_fit', True)
        # Reduced number of points
        n_pts = kwargs.pop('n_pts', 500)
        # Interactive mode
        inter = kwargs.pop('inter', False)
        # Savgol smoothing values
        smooth_params = kwargs.pop('smooth_params', [21, 3])
        # ----------------------------------------------

        name = psd_type+'-'+str(kid)+'-'+str(temp)+'-'+str(atten)

        self.data['ts'][kid][temp][atten]['psd'] = {}
        f_on, psd_on, xr_on = self.calculate_psd(kid, temp, atten, mode='on', ignore=ignore, psd_type=psd_type)
        self.data['ts'][kid][temp][atten]['psd']['on'] = [f_on, psd_on]
        f_off, psd_off, xr_off = self.calculate_psd(kid, temp, atten, mode='off', ignore=ignore, psd_type=psd_type)
        self.data['ts'][kid][temp][atten]['psd']['off'] = [f_off, psd_off]

        #f0 = self.data['vna'][kid][temp][atten]['fit']['fr']
        f0 = self.data['ts'][kid][temp][atten]['hdr_on'][0]['SYNTHFRE']
        try:
            Qr = self.data['vna'][kid][temp][atten]['fit'][0]['Qr']
        except Exception as e:
            msg('Qr not found. Qr set to zero, i.e. ring-down negligible.', 'warn')
            Qr = 0

        # S A V E   P S D   O N / O F F
        # -----------------------------
        # Save PSD binning
        down_f_on, down_psd_on = log_binning(f_on, psd_on, n_pts=n_pts)
        down_f_off, down_psd_off = log_binning(f_off, psd_off, n_pts=n_pts)
        self.data['ts'][kid][temp][atten]['psd']['binned'] = {}
        self.data['ts'][kid][temp][atten]['psd']['binned']['on'] = [down_f_on, down_psd_on]
        self.data['ts'][kid][temp][atten]['psd']['binned']['off'] = [down_f_off, down_psd_off]
        #np.save(self.work_dir+self.project_name+'/fit_psd_dict/psd_bin-'+name, \
        #        {'on': [down_f_on, down_psd_on], 'off': [down_f_off, down_psd_off]})            

        # S A V E   P S D   O N - O F F
        # -----------------------------
        # Get PSD ON - OFF
        
        level_on = np.mean(psd_on[np.where(f_on>40e3)[0][0] : np.where(f_on>50e3)[0][0]])
        level_off = np.mean(psd_off[np.where(f_off>40e3)[0][0] : np.where(f_off>50e3)[0][0]])
        print('------>>>>>****', level_on)
        print('------>>>>>****', level_off)

        #psd_off = (level_on/level_off)*psd_off
        psd_mix = psd_on # - psd_off

        f_clean = f_on[psd_mix>0]
        psd_clean = psd_mix[psd_mix>0]

        self.data['ts'][kid][temp][atten]['psd']['clean'] = [f_clean, psd_clean]

        # Bin data
        down_f_mix, down_psd_mix = log_binning(f_clean, psd_clean, n_pts=n_pts)
        self.data['ts'][kid][temp][atten]['psd']['binned']['clean'] = [down_f_mix, down_psd_mix]
        #np.save(self.work_dir+self.project_name+'/fit_psd_dict/mix_psd_bin-'+name, [down_f_mix, down_psd_mix])

        # Save PSDs
        np.save(self.work_dir+self.project_name+'/fit_psd_dict/psd-'+name, self.data['ts'][kid][temp][atten]['psd'])

        # F I T 
        # -----------------------------
        if fit:
            plot_name = kid + '-' + temp + '-' + atten

            fit_psd_params, k_knee = fit_mix_psd(f_on, psd_mix, f0, Qr, plot_name=plot_name, \
                                                trim_range=[0.2, 7.5e4], n_pts=n_pts, inter=inter, \
                                                smooth_params=smooth_params)
            
            self.data['ts'][kid][temp][atten]['fit_psd'] = {}
            self.data['ts'][kid][temp][atten]['fit_psd']['params'] = fit_psd_params
            #self.data['ts'][kid][temp][atten]['fit_psd']['psd'] = [f_nep, psd_nep]
            self.data['ts'][kid][temp][atten]['fit_psd']['k_knee'] = k_knee

            # Save PSD fit    
            np.save(self.work_dir+self.project_name+'/fit_psd_dict/fit_psd-'+name, self.data['ts'][kid][temp][atten]['fit_psd'])

        # P L O T   F I T 
        # -----------------------------
        if plot_fit:

            fig, ax = subplots(1,1, figsize=(20,12))
            subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.07, hspace=0.0, wspace=0.0)

            # Plot PSD on/off
            ax.loglog(f_on, psd_on, 'm', lw=0.75, alpha=0.45)
            ax.loglog(f_off, psd_off, 'g', lw=0.75, alpha=0.45)
            # PSD on-off
            ax.loglog(f_clean, psd_clean, lw=1.0)

            ax.set_ylim([np.min(psd_off)-0.11*np.min(psd_off), np.max(psd_on)+2*np.max(psd_on)])
            ax.grid(True, which="both", ls="-")

            base_temp = self.data['ts'][kid][temp][atten]['s21_hr']['SAMPLETE']
            ax.set_title('PSD-noise-'+name+'-'+str(base_temp)+' K')
            ax.set_xlabel('Frequency[Hz]')
            if psd_type == 'df':
                ax.set_ylabel(r'PSD [Hz$^2$/Hz]')
            elif psd_type == 'phase':
                ax.set_ylabel(r'PSD [rad$^2$/Hz]')

            if fit:
                fit_PSD = spectra_noise_model(f_clean, fit_psd_params['gr'], fit_psd_params['tau'], fit_psd_params['tls_a'],
                                         fit_psd_params['tls_b'], fit_psd_params['amp_noise'], Qr, f0)
                
                ax.loglog(f_clean, fit_PSD, 'k', lw=2.5, label='fit')

                # Generation-Recombination noise
                gr = gr_noise(f_clean, fit_psd_params['gr'], fit_psd_params['tau'], Qr, f0)
                # TLS noise
                tls = tls_noise(f_clean, fit_psd_params['tls_a'], fit_psd_params['tls_b'], fit_psd_params['tau'], Qr, f0)

                ax.loglog(f_clean, gr, 'r-', lw=2.5, label='gr noise')
                if psd_type == 'df':
                    ax.text(0.5, gr[0]*1.5, f'GR:{gr[0]:.3f} Hz^2/Hz')
                elif psd_type == 'phase':
                    ax.text(0.5, gr[0]*1.5, f'GR:{1e6*gr[0]:.3f} u rad^2/Hz')

                ax.loglog(f_clean, tls, 'b-', lw=2.5, label='1/f')
                ax.loglog(f_clean, fit_psd_params['amp_noise']*np.ones_like(f_clean), 'g-', label='amp', lw=2)

                tau = fit_psd_params['tau']
                ax.text(0.8, 0.8, \
                        f'Qr  : {round(Qr,-1):,.0f}\nf0   : {round(f0/1e6,-1):,.1f} MHz\ntau : {tau*1e6:.1f} us\nTLS_b : {fit_psd_params['tls_b']:.1f}', \
                        transform=ax.transAxes)

                ax.axhline(fit_psd_params['gr'], color='k', linestyle='dashed', lw=2)

                knee = self.data['ts'][kid][temp][atten]['fit_psd']['k_knee']
                ax.axvline(knee, color='m', lw=2.5)
                ax.text(knee, gr[0]*2.5, f'1/f knee: {knee:.1f} Hz')

            fig.savefig(self.work_dir+self.project_name+'/fit_psd_results/'+name+'.png')
            close(fig)

    def calculate_psd(self, kid, temp, atten, mode='on', ignore=[[0,1,2], [0]], psd_type='df'):
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
        psd_type : string
            'df' resonance frequency shift.
            'phase' resonator phase.
        ----------
        """

        if not psd_type in ["df", "phase"]:
            msg("PSD type not valid", "fail")
            return -1

        #try:

        # H O M O D Y N E   S W E E P
        # -----------------------------------------------------
        f_s21 = self.data['ts'][kid][temp][atten]['f']
        s21_h = self.data['ts'][kid][temp][atten]['s21']
        Is_h = s21_h.real
        Qs_h = s21_h.imag

        f_high = self.data['ts'][kid][temp][atten]['f_high']
        s21_high = self.data['ts'][kid][temp][atten]['s21_high']

        # Low-frequency PSD
        # -----------------------------------------------------
        I_low = self.data['ts'][kid][temp][atten]['I_'+mode][0]
        Q_low = self.data['ts'][kid][temp][atten]['Q_'+mode][0]
        I_low_f, Q_low_f = [], []
        for i in range(len(I_low)):
            if not i in ignore[0]:
                I_low_f.append(I_low[i])
                Q_low_f.append(Q_low[i])

        print('Low-freq samples: ', len(I_low_f))

        hdr_low = self.data['ts'][kid][temp][atten]['hdr_'+mode][0]
        fs = hdr_low['SAMPLERA']
        f0 = hdr_low['SYNTHFRE']

        print('Sample frequency: ', fs)
        print('Resonance frequency [SYNTHFRE]: ', f0)

        if psd_type == 'df':

            I0 = hdr_low['IF0']
            Q0 = hdr_low['QF0']
            
            if mode == 'on':

                print('F 0   O N   R E S O N A N C E')
                print(f0, I0, Q0)
                self._xc, self._yc, self._theta, self.Is_derot, self.Qs_derot, self.kid_mdl = get_rot_iq_circle(f0, I0, Q0, f_s21, Is_h, Qs_h, f_s21, s21_h, f0_thresh=5e4)
                xr_low, It_derot, Qt_derot, self.phase_ref = df_from_derot_circle(self._xc, self._yc, self._theta, I_low_f, Q_low_f, self.kid_mdl, f0, name='low')

                fig, axs = subplots(1, 1, figsize=(20, 12))
                axs.plot(Is_h, Qs_h, 'k.-')
                axs.plot(I0, Q0, 'gs')
                axs.plot(self.Is_derot, self.Qs_derot, 'r.-')

                I0_derot = (I0 - self._xc)*np.cos(-self._theta)-(Q0 - self._yc)*np.sin(-self._theta)
                Q0_derot = (I0 - self._xc)*np.sin(-self._theta)+(Q0 - self._yc)*np.cos(-self._theta)

                axs.plot(I0_derot, Q0_derot, 'gs')

                for t in range(len(I_low_f)):
                    axs.plot(I_low_f[t], Q_low_f[t], ',')
                    axs.plot(It_derot[t], Qt_derot[t], ',')

                axs.axis('equal')
                axs.set_xlabel('I[V]')
                axs.set_ylabel('Q[V]')
                axs.grid()

                fig.savefig(kid+'-'+temp+'-'+atten+'-'+mode+'-low.png')
                close(fig)
                                
            elif mode == 'off':
                
                xr_low, didq_mag_low = self.calculate_df(I_low_f, Q_low_f, hdr_low)

                """
                print('F 0   O F F   R E S O N A N C E')
                I0_off = Is_h[np.where(f_s21 >= f0)[0][0]]
                Q0_off = Qs_h[np.where(f_s21 >= f0)[0][0]]

                I0_off_derot = (I0_off - self._xc)*np.cos(-self._theta)-(Q0_off - self._yc)*np.sin(-self._theta)
                Q0_off_derot = (I0_off - self._xc)*np.sin(-self._theta)+(Q0_off - self._yc)*np.cos(-self._theta)

                add_rot = np.arctan2(Q0_off_derot, I0_off_derot)
                f0_on = self.data['ts'][kid][temp][atten]['s21_hr']['F0FOUND']
                print(f0, f0_on, I0_off_derot, Q0_off_derot, add_rot)
                
                xr_low, It_derot, Qt_derot, _ = df_from_derot_circle(self._xc, self._yc, self._theta, I_low_f, Q_low_f, self.kid_mdl, f0_on, name='low', mode='off', add_rot=add_rot)
                """

                """
                fig, axs = subplots(1, 1, figsize=(20, 12))
                axs.plot(Is_h, Qs_h, 'k.-')
                axs.plot(I0_off, Q0_off, 'gs')
                axs.plot(self.Is_derot, self.Qs_derot, 'r.-')

                axs.plot(I0_off_derot, Q0_off_derot, 'gs')

                for t in range(len(I_low_f)):
                    axs.plot(I_low_f[t], Q_low_f[t], ',')
                    axs.plot(It_derot[t], Qt_derot[t], ',')

                    #figt, axst = subplots(1, 1, figsize=(20,12))
                    #axst.plot(xr_low[t])
                    #figt.savefig(kid+'-'+temp+'-'+atten+'-'+mode+'-ts-'+str(t)+'-low.png')
                    #close(figt)                    

                axs.axis('equal')
                axs.set_xlabel('I[V]')
                axs.set_ylabel('Q[V]')
                axs.grid()

                fig.savefig(kid+'-'+temp+'-'+atten+'-'+mode+'-low-off.png')
                close(fig)
                """

        elif psd_type == 'phase':

            I0 = hdr_low['IF0']
            Q0 = hdr_low['QF0']

            if mode == 'on':
                self._xc, self._yc, self._theta, self.Is_derot, self.Qs_derot, self.kid_mdl = get_rot_iq_circle(f0, I0, Q0, f_s21, Is_h, Qs_h, f_s21, s21_h, f0_thresh=5e4)

            xr_low = derot_phase(self._xc, self._yc, self._theta, I_low_f, Q_low_f)

        freq_low, psd_low = get_psd(np.array(xr_low), fs)

        # High-frequency PSD
        # -----------------------------------------------------
        I_high = self.data['ts'][kid][temp][atten]['I_'+mode][1]
        Q_high = self.data['ts'][kid][temp][atten]['Q_'+mode][1]

        I_high_f, Q_high_f = [], []
        for i in range(len(I_high)):
            if not i in ignore[1]:
                I_high_f.append(I_high[i])
                Q_high_f.append(Q_high[i])

        print('High-freq samples: ', len(I_high_f))

        hdr_high = self.data['ts'][kid][temp][atten]['hdr_'+mode][1]
        fs = hdr_high['SAMPLERA']
        f0 = hdr_high['SYNTHFRE']

        print('Sample frequency: ', fs)
        print('Resonance frequency [SYNTHFRE]: ', f0)

        if psd_type == 'df':

            if mode == 'on':
                
                xr_high, It_derot, Qt_derot, self.phase_ref_high = df_from_derot_circle(self._xc, self._yc, self._theta, I_high_f, Q_high_f, self.kid_mdl, f0, name='high')

                fig, axs = subplots(1, 1, figsize=(20, 12))
                axs.plot(Is_h, Qs_h, 'k.-')
                axs.plot(self.Is_derot, self.Qs_derot, 'r.-')

                for t in range(len(I_high_f)):
                    axs.plot(I_high_f[t], Q_high_f[t], ',')
                    axs.plot(It_derot[t], Qt_derot[t], ',') 

                axs.axis('equal')
                axs.set_xlabel('I[V]')
                axs.set_ylabel('Q[V]')
                axs.grid()

                fig.savefig(kid+'-'+temp+'-'+atten+'-'+mode+'-high.png')
                close(fig)
                
            elif mode == 'off':
                
                xr_high, didq_mag_high = self.calculate_df(I_high_f, Q_high_f, hdr_high)

                """
                print('F 0   O F F   R E S O N A N C E')
                I0_off = Is_h[np.where(f_s21 >= f0)[0][0]]
                Q0_off = Qs_h[np.where(f_s21 >= f0)[0][0]]

                I0_off_derot = (I0_off - self._xc)*np.cos(-self._theta)-(Q0_off - self._yc)*np.sin(-self._theta)
                Q0_off_derot = (I0_off - self._xc)*np.sin(-self._theta)+(Q0_off - self._yc)*np.cos(-self._theta)

                add_rot = np.arctan2(Q0_off_derot, I0_off_derot)
                f0_on = self.data['ts'][kid][temp][atten]['s21_hr']['F0FOUND']
                print(f0, f0_on, I0_off_derot, Q0_off_derot, add_rot)

                xr_high, It_derot, Qt_derot, _ = df_from_derot_circle(self._xc, self._yc, self._theta, I_high_f, Q_high_f, self.kid_mdl, f0_on, name='high', mode='off', add_rot=add_rot)
                """
               
        elif psd_type == 'phase':

            xr_high = derot_phase(self._xc, self._yc, self._theta, I_high_f, Q_high_f)

        freq_high, psd_high = get_psd(np.array(xr_high), fs)

        f, psd = mix_psd([freq_low, freq_high], [psd_low, psd_high])

        return f, psd, (xr_low, xr_high)

        #except Exception as e:
        #    msg('Data not available.\n'+str(e), 'fail')
        #    return -1

    def despike(self, kid=None, temp=None, atten=None, ignore=[[0,1], [0]], \
                win_size=350, sigma_thresh=3.5, peak_pts=4, **kwargs):
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
        # Key arguments
        # ----------------------------------------------
        # Verbose
        verbose = kwargs.pop('verbose', False)
        # ----------------------------------------------

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
            split_name = f.split('-')
            kid = split_name[1]
            temp = split_name[2]
            atten = split_name[3]
            ns = int((f.split('-')[-1][1:]).split('.')[0])

            try:
                sample = np.load(folder+'/fit_res_dict/'+data_type+'/'+f, allow_pickle=True).item()

                if not 'fit' in self.data['vna'][kid][temp][atten]:
                    self.data['vna'][kid][temp][atten]['fit'] = {}
                self.data['vna'][kid][temp][atten]['fit'][ns] = sample

            except Exception as e:
                print('Resonator fit. Fail loading '+f+'\n'+str(e))

    def load_psd(self, folder):
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

        files = next(walk(folder+'/fit_psd_dict/'), (None, None, []))[2]

        for f in files:
            try:
                split_name = f[:-4].split('-')
               
                file_type = split_name[0]
                kid = split_name[2]
                temp = split_name[3]
                atten = split_name[4]
                #ns = int((f.split('-')[-1][1:]).split('.')[0])

                data = np.load(folder+'/fit_psd_dict/'+f, allow_pickle=True).item()

                #if not file_type == 'mix_psd_bin':
                #    data = data.item()

                print(data.keys(), self.data['ts'].keys())

                if file_type == 'fit_psd':
                    self.data['ts'][kid][temp][atten]['fit_psd'] = data
                elif file_type == 'psd':
                    self.data['ts'][kid][temp][atten]['psd'] = data
                #elif file_type == 'psd_bin':
                #    self.data['ts'][kid][temp][atten]['psd']['binned'] = data
                #elif file_type == 'mix_psd_bin':
                #    self.data['ts'][kid][temp][atten]['psd']['binned']['clean'] = data

            except Exception as e:
                print('PSD fit. Fail loading '+f+'\n'+str(e))

    def vna_xls_report(self, name=None, kids=None):
        """
        Create the report of results from the VNA measurements.
        Parameters
        ----------
        name  : string
            File name.
        ----------
        """

        if name == None:
            name = self.project_name+'-VNA_report.xlsx'

        workbook = xlsxwriter.Workbook(self.work_dir+self.project_name+'/'+name)

        bold = workbook.add_format({'bold': True})

        kids = self._get_kids_to_sweep(kids, mode='vna')
        temps = self._get_temps_to_sweep(None, kids[0], mode='vna')

        # It should be only the defined temperatures.
        for t, temp in enumerate(temps):
            worksheet = workbook.add_worksheet(temp)
            block = 0

            f0s_max, qis_max, qcs_max = [], [], []
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

                f0s, qis, qcs = [], [], []
                for atten in attens:
                    if k == 0:
                        worksheet.write(1, 0, 'Att[dB]', bold)
                        worksheet.write(col+2, 0, atten, bold)

                    try:
                        f0 = self.data['vna'][kid][temp][atten]['fit'][0]['fr']

                        qi = self.data['vna'][kid][temp][atten]['fit'][0]['Qi']
                        worksheet.write(col+2, 3*block+1, qi)
                        qc = self.data['vna'][kid][temp][atten]['fit'][0]['Qc']
                        worksheet.write(col+2, 3*block+2, qc)
                        qr = self.data['vna'][kid][temp][atten]['fit'][0]['Qr']
                        worksheet.write(col+2, 3*block+3, qr)

                        if float(atten[1:]) >= self.overdriven[k]:
                            f0s.append(f0)
                            qis.append(qi)
                            qcs.append(qc)

                    except Exception as e:
                        print(e)

                    col += 1

                try:
                    id_max_qi = np.argmax(qis)
                    f0s_max.append(f0s[id_max_qi])
                    qis_max.append(qis[id_max_qi])
                    qcs_max.append(qcs[id_max_qi])

                except Exception as e:
                    f0s_max.append(f0)
                    qis_max.append(-1)
                    qcs_max.append(-1)

                att_num = [float(a[1:]) for a in attens]
                try:
                    from_cal = np.where(np.array(att_num)>=self.overdriven[k])[0][0]
                except:
                    from_cal = 0

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

            if t == 0:
                np.save(self.work_dir+self.project_name+'/summary', [f0s_max, qis_max, qcs_max])

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


    # S O M E   U S E F U L   P L O T S
    # --------------------------------------------------------------------------
    def plot_dip_depths(self, kid=None, temp=None, atten=None, sample=0, cmap='jet', *args, **kwargs):
        """
        Create dip depths plot.
        Let's start with CPW
        Parameters
        ----------
        ----------
        """

        # Key arguments
        # ----------------------------------------------
        # Baseline limits
        baseline_lims = kwargs.pop('baseline_lims', (50, 50))
        # ----------------------------------------------

        kids = self._get_kids_to_sweep(kid)

        temporal_temps = []
        join_temps = []
        for kid in kids:
            join_temps.append(self._get_temps_to_sweep(temp, kid, mode='vna'))
            temporal_temps.append(len(self._get_temps_to_sweep(temp, kid, mode='vna')))

        n_temps = join_temps[np.argmax(temporal_temps)]

        xg = int(np.sqrt(len(n_temps)))
        yg = int(np.ceil(len(n_temps)/xg))

        ioff()
        fig_axm, axm = subplots(xg, yg, sharey=True, sharex=True, figsize=(20,12))
        subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.07, hspace=0.0, wspace=0.0)
        fig_max, ax_max = subplots(xg, yg, sharey=True, sharex=True, figsize=(20,12))
        subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.07, hspace=0.0, wspace=0.0)
        
        for t, tm in enumerate(n_temps):

            i = t%yg
            j = int(t/yg)

            lstyle_pointer = 0
            max_dd, f0max = [], []

            for k in range(len(kids)):

                kid = kids[k]

                if k%10 == 0 and k > 0:
                    lstyle_pointer += 1
                   
                attens = self._get_atten_to_sweep(atten, tm, kid)

                atts_num, dds = [], []
                for single_atten in attens:
                    try:

                        if float(single_atten[1:]) >= self.overdriven[k]: # and float(single_atten[1:]) < 56:

                            f, s21 = self.data['vna'][kid][tm][single_atten]['data'][sample]

                            if self.data_type.lower() == 'dark':                              
                                real_temp = f'{self.data['vna'][kid][tm][single_atten]['header'][sample]['SAMPLETE']*1e3:.0f}mK'
                            elif self.data_type.lower() == 'blackbody':                              
                                real_temp = f'{self.data['vna'][kid][tm][single_atten]['header'][sample]['BLACKBOD']:.1f}K'

                            # Get dip-depths
                            if kid == 'K1111':
                                dd_pts = 41
                            else:
                                dd_pts = 51

                            s21_mag = 20*np.log10(np.abs(s21))
                            
                            s21_baseline = np.concatenate((s21_mag[:baseline_lims[0]], s21_mag[-baseline_lims[1]:]))
                            f_baseline = np.concatenate((f[:baseline_lims[0]], f[-baseline_lims[1]:]))
                            
                            base_interp = scipy.interpolate.interp1d(f_baseline, s21_baseline)
                            fit_baseline = base_interp(f)

                            s21_clear = s21_mag - fit_baseline

                            s21_sm = savgol_filter(s21_clear, dd_pts, 3)
                            min_s21 = np.min(s21_sm)

                            #figure()
                            #print(baseline_lims)
                            #plot(f, s21_sm)

                            dd = np.abs(min_s21)
                            #print(dd)
                            dds.append(dd)

                            extra_att = self.data['vna'][kid][tm][single_atten]['header'][0]['ATT_UC'] + \
                                        self.data['vna'][kid][tm][single_atten]['header'][0]['ATT_C'] + \
                                        self.data['vna'][kid][tm][single_atten]['header'][0]['ATT_RT']

                            vna_pwr = self.data['vna'][kid][tm][single_atten]['header'][0]['VNAPOWER']
                                        
                            atts_num.append(-1*(float(single_atten[1:])+extra_att+self.add_in_atten)+vna_pwr )
                
                    except Exception as e:
                        msg(str(e), 'fail')


                if xg == 1 and yg == 1:
                    axm.plot(atts_num, dds, 'D', linestyle=lstyle[lstyle_pointer], label=kid)

                elif xg == 1:
                    axm[i].plot(atts_num, dds, 'D', linestyle=lstyle[lstyle_pointer], label=kid)

                else:
                    axm[j, i].plot(atts_num, dds, 'D', linestyle=lstyle[lstyle_pointer], label=kid)

                try:
                    # Get the maximum Q
                    max_dd.append(np.max(dds))

                    max_att = attens[np.argmax(dds)]
                    ft, s21t = self.data['vna'][kid][tm][max_att]['data'][sample]
                    f0max.append(ft[np.argmin(np.abs(s21t))])
                
                except:
                    pass

                # D I P   D E P T H S
                self._create_diry(self.work_dir+self.project_name+'/dip_depths')
                np.save(self.work_dir+self.project_name+'/dip_depths/dip_depths_'+kid+'-'+tm, [atts_num, dds])

            if xg == 1 and yg == 1:
                ax_max.plot(np.array(f0max)*1e-6, max_dd, 'ks-')

                for m in range(len(max_dd)):
                    ax_max.text( f0max[m]*1e-6, max_dd[m]-0.01, 'K'+str(m).zfill(3) )

            elif xg == 1:
                ax_max[i].plot(np.array(f0max)*1e-6, max_dd, 'ks-')

                for m in range(len(max_dd)):
                    ax_max[i].text( f0max[m]*1e-6, max_dd[m]-0.01, 'K'+str(m).zfill(3) )
            
            else:
                ax_max[j, i].plot(np.array(f0max)*1e-6, max_dd, 'ks-')

                for m in range(len(max_dd)):
                    ax_max[j, i].text( f0max[m]*1e-6, max_dd[m]-0.01, 'K'+str(m).zfill(3) )

            if len(n_temps) > 1:
                if self.data_type.lower() == 'dark':
                    subfix = 'mK'
                    real_temp = f'{self.data['vna'][kid][tm][max_att]['header'][0]['SAMPLETE']*1e3:.0f}'

                elif self.data_type.lower() == 'blackbody':
                    subfix = 'K'
                    real_temp = f'{self.data['vna'][kid][tm][max_att]['header'][0]['BLACKBOD']:.1f}'

                if xg == 1:
                    axm[i].text(0.7, 0.85, real_temp + ' '+subfix, {'fontsize': 15, 'color': 'white'}, bbox=dict(facecolor='blue', alpha=0.95), transform=axm[i].transAxes,)
                    ax_max[i].text(0.7, 0.85, real_temp + ' '+subfix, {'fontsize': 15, 'color': 'white'}, bbox=dict(facecolor='blue', alpha=0.95), transform=ax_max[i].transAxes,)

                else:
                    axm[j, i].text(0.7, 0.85, real_temp + ' '+subfix, {'fontsize': 15, 'color': 'white'}, bbox=dict(facecolor='blue', alpha=0.95), transform=axm[j, i].transAxes,)
                    ax_max[j, i].text(0.7, 0.85, real_temp + ' '+subfix, {'fontsize': 15, 'color': 'white'}, bbox=dict(facecolor='blue', alpha=0.95), transform=ax_max[j, i].transAxes,)

            if i == 0:
                if len(n_temps) == 1:
                    axm.set_ylabel('Dip-depth [dB]', fontsize=18, weight='bold')
                    ax_max.set_ylabel('Dip-depth [dB]', fontsize=18, weight='bold')

                else:
                    if xg == 1:
                        axm[i].set_ylabel('Dip-depth [dB]', fontsize=18, weight='bold')
                        ax_max[i].set_ylabel('Dip-depth [dB]', fontsize=18, weight='bold')

                    else:
                        axm[j, i].set_ylabel('Dip-depth [dB]', fontsize=18, weight='bold')
                        ax_max[j, i].set_ylabel('Dip-depth [dB]', fontsize=18, weight='bold')

            if len(n_temps) == 1:
                axm.grid(True, which="both", ls="-")
                ax_max.grid(True, which="both", ls="-")
            
            else:
                if xg == 1:
                    axm[i].grid(True, which="both", ls="-")
                    ax_max[i].grid(True, which="both", ls="-")

                else:
                    axm[j, i].grid(True, which="both", ls="-")
                    ax_max[j, i].grid(True, which="both", ls="-")

            if j == xg - 1:
                if len(n_temps) == 1:
                    axm.set_xlabel('Drive power [dBm]', fontsize=18, weight='bold')
                    ax_max.set_xlabel('Resonance frequency [MHz]', fontsize=18, weight='bold')

                else:
                    if xg == 1:
                        axm[i].set_xlabel('Drive power [dBm]', fontsize=18, weight='bold')
                        ax_max[i].set_xlabel('Resonance frequency [MHz]', fontsize=18, weight='bold')

                    else:
                        axm[j, i].set_xlabel('Drive power [dBm]', fontsize=18, weight='bold')
                        ax_max[j, i].set_xlabel('Resonance frequency [MHz]', fontsize=18, weight='bold')

                if t == len(n_temps)-1:
                    if len(n_temps) == 1:
                        axm.legend(ncol=2)

                    else:
                        if xg == 1:
                            axm[i].legend(ncol=2)

                        else:
                            axm[j, i].legend(ncol=2)

        show()

        # Save figures
        fig_max.savefig(self.work_dir+self.project_name+'/fit_res_results/summary_plots/Max-dip-depths.png')
        fig_axm.savefig(self.work_dir+self.project_name+'/fit_res_results/summary_plots/Dip-depths-vs-drive-power.png')

    def plot_qs_vs_drive_power(self, kid=None, temp=None, atten=None, **kwargs):
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
        # Key arguments
        # ----------------------------------------------
        # Ignore detectors
        ignore = kwargs.pop('ignore', [])
        # cmap
        cmap = kwargs.pop('cmap', 'tab10')
        # ----------------------------------------------

        ioff()
        cmap_obj = matplotlib.cm.get_cmap(cmap)

        kids = self._get_kids_to_sweep(kid, mode='vna')
        ignore_kids = self._get_kids_to_sweep(ignore, mode='vna')

        temporal_temps = []
        join_temps = []
        for kid in kids:
            join_temps.append(self._get_temps_to_sweep(temp, kid, mode='vna'))
            temporal_temps.append(len(self._get_temps_to_sweep(temp, kid, mode='vna')))

        n_temps = join_temps[np.argmax(temporal_temps)]

        x = int(np.sqrt(len(n_temps)))
        y = int(np.ceil(len(n_temps)/x))

        fig_qi, ax_qi = subplots(x, y, sharey=True, sharex=True, figsize=(20,12))
        subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.07, hspace=0.0, wspace=0.0)
        fig_qc, ax_qc = subplots(x, y, sharey=True, sharex=True, figsize=(20,12))
        subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.07, hspace=0.0, wspace=0.0)
        fig_qr, ax_qr = subplots(x, y, sharey=True, sharex=True, figsize=(20,12))
        subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.07, hspace=0.0, wspace=0.0)

        # Dip depths
        fig_max, ax_max = subplots(x, y, sharey=True, sharex=True, figsize=(20,12))
        subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.07, hspace=0.0, wspace=0.0)
        fig_mean, ax_mean = subplots(x, y, sharey=True, sharex=True, figsize=(20,12))
        subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.07, hspace=0.0, wspace=0.0)

        for t, tmp in enumerate(n_temps):

            i = t%y
            j = int(t/y)

            lstyle_pointer = 0
            
            flag_color_loop = False
            if cmap in ['tab10', 'tab20']:
                flag_color_loop = True

            if flag_color_loop:
                vmax_color = int(cmap[-2:])
            else:
                vmax_color = len(kids)

            norm_color = matplotlib.colors.Normalize(vmin=0, vmax=vmax_color)

            max_qi = []
            mean_qc = []
            f0_max = []
            
            for k, kid in enumerate(kids):

                if not kid in ignore_kids:

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

                                    # Get the attenuations
                                    extra_att = self.data['vna'][kid][tmp][att]['header'][0]['ATT_UC'] + \
                                                self.data['vna'][kid][tmp][att]['header'][0]['ATT_C'] + \
                                                self.data['vna'][kid][tmp][att]['header'][0]['ATT_RT'] 
                                                # This should be commented just for RV2-Chip 2

                                    #print(self.data['vna'][kid][tmp][att]['header'][0]['ATT_UC'])
                                    #print(self.data['vna'][kid][tmp][att]['header'][0]['ATT_C'])
                                    #print(self.data['vna'][kid][tmp][att]['header'][0]['ATT_RT'])

                                    vna_pwr = self.data['vna'][kid][tmp][att]['header'][0]['VNAPOWER']
                                    atts_num.append(-1*(float(att[1:])+extra_att+self.add_in_atten)+vna_pwr )

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
                    
                    # Get the max Qi
                    try:
                        max_qi.append(np.max(qi))

                        # Get the mean Qc
                        mean_qc.append(np.mean(qc))


                        max_att = attens[np.argmax(qi)]
                        ft, s21t = self.data['vna'][kid][tmp][max_att]['data'][0]
                        f0_max.append(ft[np.argmin(np.abs(s21t))])

                    except:
                        pass


                    if k%10 == 0 and k > 0:
                        lstyle_pointer += 1

                    if flag_color_loop:
                        k_color = k%int(cmap[-2:])
                    else:
                        k_color = k

                    if len(n_temps) == 1:

                        ax_qi.errorbar(atts_num, qi, yerr=qi_err, color=cmap_obj(norm_color(k_color)), marker='s', ecolor='k', capsize=2, linestyle=lstyle[lstyle_pointer])
                        ax_qi.plot(atts_num, qi, 's', label=kid, color=cmap_obj(norm_color(k_color)), linestyle=lstyle[lstyle_pointer])

                        ax_qc.errorbar(atts_num, qc, yerr=qc_err, color=cmap_obj(norm_color(k_color)), marker='s', ecolor='k', capsize=2, linestyle=lstyle[lstyle_pointer])
                        ax_qc.plot(atts_num, qc, '^', label=kid, color=cmap_obj(norm_color(k_color)), linestyle=lstyle[lstyle_pointer])

                        ax_qr.errorbar(atts_num, qr, yerr=qr_err, color=cmap_obj(norm_color(k_color)), marker='s', ecolor='k', capsize=2, linestyle=lstyle[lstyle_pointer])
                        ax_qr.plot(atts_num, qr, 'o', label=kid, color=cmap_obj(norm_color(k_color)), linestyle=lstyle[lstyle_pointer])

                        """
                        #ax_qi.errorbar(atts_num, qi, yerr=qi_err, marker='s', ecolor='k', capsize=2, linestyle=lstyle[lstyle_pointer])
                        ax_qi.plot(atts_num, qi, 's', label=kid,  linestyle=lstyle[lstyle_pointer])

                        #ax_qc.errorbar(atts_num, qc, yerr=qc_err, marker='s', ecolor='k', capsize=2, linestyle=lstyle[lstyle_pointer])
                        ax_qc.plot(atts_num, qc, '^', label=kid,  linestyle=lstyle[lstyle_pointer])

                        #ax_qr.errorbar(atts_num, qr, yerr=qr_err, marker='s', ecolor='k', capsize=2, linestyle=lstyle[lstyle_pointer])
                        ax_qr.plot(atts_num, qr, 'o', label=kid,  linestyle=lstyle[lstyle_pointer])
                        """
                        
                    else:
                        if x == 1:
                            ax_qi[i].errorbar(atts_num, qi, yerr=qi_err, color=cmap_obj(norm_color(k_color)), marker='s', ecolor='k', capsize=2, linestyle=lstyle[lstyle_pointer])
                            ax_qi[i].plot(atts_num, qi, 's', label=kid, color=cmap_obj(norm_color(k_color)), linestyle=lstyle[lstyle_pointer])
                            
                            ax_qc[i].errorbar(atts_num, qc, yerr=qc_err, color=cmap_obj(norm_color(k_color)), marker='s', ecolor='k', capsize=2, linestyle=lstyle[lstyle_pointer])
                            ax_qc[i].plot(atts_num, qc, '^', label=kid, color=cmap_obj(norm_color(k_color)), linestyle=lstyle[lstyle_pointer])

                            ax_qr[i].errorbar(atts_num, qr, yerr=qr_err, color=cmap_obj(norm_color(k_color)), marker='s', ecolor='k', capsize=2, linestyle=lstyle[lstyle_pointer])
                            ax_qr[i].plot(atts_num, qr, 'o', label=kid, color=cmap_obj(norm_color(k_color)), linestyle=lstyle[lstyle_pointer])

                        else:
                            ax_qi[j, i].errorbar(atts_num, qi, yerr=qi_err, color=cmap_obj(norm_color(k_color)), marker='s', ecolor='k', capsize=2, linestyle=lstyle[lstyle_pointer])
                            ax_qi[j, i].plot(atts_num, qi, 's', label=kid, color=cmap_obj(norm_color(k_color)), linestyle=lstyle[lstyle_pointer])
                            
                            ax_qc[j, i].errorbar(atts_num, qc, yerr=qc_err, color=cmap_obj(norm_color(k_color)), marker='s', ecolor='k', capsize=2, linestyle=lstyle[lstyle_pointer])
                            ax_qc[j, i].plot(atts_num, qc, '^', label=kid, color=cmap_obj(norm_color(k_color)), linestyle=lstyle[lstyle_pointer])

                            ax_qr[j, i].errorbar(atts_num, qr, yerr=qr_err, color=cmap_obj(norm_color(k_color)), marker='s', ecolor='k', capsize=2, linestyle=lstyle[lstyle_pointer])
                            ax_qr[j, i].plot(atts_num, qr, 'o', label=kid, color=cmap_obj(norm_color(k_color)), linestyle=lstyle[lstyle_pointer])

            if x == 1 and y == 1:
                ax_max.plot(np.array(f0_max)*1e-6, max_qi, 'ks-')
                ax_mean.plot(np.array(f0_max)*1e-6, mean_qc, 'ks-')

                for m in range(len(max_qi)):
                    ax_max.text( f0_max[m]*1e-6, max_qi[m]-0.1, 'K'+str(m).zfill(3) )
                    ax_mean.text( f0_max[m]*1e-6, mean_qc[m]-0.1, 'K'+str(m).zfill(3) )

            elif x == 1:
                ax_max[i].plot(np.array(f0_max)*1e-6, max_qi, 'ks-')
                ax_mean[i].plot(np.array(f0_max)*1e-6, mean_qc, 'ks-')

                for m in range(len(max_qi)):
                    ax_max[i].text( f0_max[m]*1e-6, max_qi[m]-0.1, 'K'+str(m).zfill(3) )
                    ax_mean[i].text( f0_max[m]*1e-6, mean_qc[m]-0.1, 'K'+str(m).zfill(3) )
            
            else:
                ax_max[j, i].plot(np.array(f0_max)*1e-6, max_qi, 'ks-')
                ax_mean[j, i].plot(np.array(f0_max)*1e-6, mean_qc, 'ks-')

                for m in range(len(max_qi)):
                    ax_max[j, i].text( f0_max[m]*1e-6, max_qi[m]-0.1, 'K'+str(m).zfill(3) )
                    ax_mean[j, i].text( f0_max[m]*1e-6, mean_qc[m]-0.1, 'K'+str(m).zfill(3) )

            if len(n_temps) > 1:
                if self.data_type.lower() == 'dark':
                    subfix = 'mK'
                    real_temp = f'{self.data['vna'][kid][tmp][att]['header'][0]['SAMPLETE']*1e3:.0f}'

                elif self.data_type.lower() == 'blackbody':
                    subfix = 'K'
                    real_temp = f'{self.data['vna'][kid][tmp][att]['header'][0]['BLACKBOD']:.1f}'

                if x == 1:
                    ax_qr[i].text(0.7, 0.85, real_temp + ' '+subfix, {'fontsize': 15, 'color': 'white'}, bbox=dict(facecolor='blue', alpha=0.95), transform=ax_qr[i].transAxes,)
                    ax_qc[i].text(0.7, 0.85, real_temp + ' '+subfix, {'fontsize': 15, 'color': 'white'}, bbox=dict(facecolor='blue', alpha=0.95), transform=ax_qc[i].transAxes,)
                    ax_qi[i].text(0.7, 0.85, real_temp + ' '+subfix, {'fontsize': 15, 'color': 'white'}, bbox=dict(facecolor='blue', alpha=0.95), transform=ax_qi[i].transAxes,)
                    ax_max[i].text(0.7, 0.85, real_temp + ' '+subfix, {'fontsize': 15, 'color': 'white'}, bbox=dict(facecolor='blue', alpha=0.95), transform=ax_max[i].transAxes,)
                    ax_mean[i].text(0.7, 0.85, real_temp + ' '+subfix, {'fontsize': 15, 'color': 'white'}, bbox=dict(facecolor='blue', alpha=0.95), transform=ax_mean[i].transAxes,)

                else:
                    ax_qr[j, i].text(0.7, 0.85, real_temp + ' '+subfix, {'fontsize': 15, 'color': 'white'}, bbox=dict(facecolor='blue', alpha=0.95), transform=ax_qr[j, i].transAxes,)
                    ax_qc[j, i].text(0.7, 0.85, real_temp + ' '+subfix, {'fontsize': 15, 'color': 'white'}, bbox=dict(facecolor='blue', alpha=0.95), transform=ax_qc[j, i].transAxes,)
                    ax_qi[j, i].text(0.7, 0.85, real_temp + ' '+subfix, {'fontsize': 15, 'color': 'white'}, bbox=dict(facecolor='blue', alpha=0.95), transform=ax_qi[j, i].transAxes,)
                    ax_max[j, i].text(0.7, 0.85, real_temp + ' '+subfix, {'fontsize': 15, 'color': 'white'}, bbox=dict(facecolor='blue', alpha=0.95), transform=ax_max[j, i].transAxes,)
                    ax_mean[j, i].text(0.7, 0.85, real_temp + ' '+subfix, {'fontsize': 15, 'color': 'white'}, bbox=dict(facecolor='blue', alpha=0.95), transform=ax_mean[j, i].transAxes,)

            if i == 0:
                if len(n_temps) == 1:
                    ax_qi.set_ylabel('Qi', fontsize=18, weight='bold')
                    ax_qc.set_ylabel('Qc', fontsize=18, weight='bold')
                    ax_qr.set_ylabel('Qr', fontsize=18, weight='bold')
                    ax_max.set_ylabel('max Qi', fontsize=18, weight='bold')
                    ax_mean.set_ylabel('mean Qc', fontsize=18, weight='bold')

                else:
                    if x == 1:
                        ax_qi[i].set_ylabel('Qi', fontsize=18, weight='bold')
                        ax_qc[i].set_ylabel('Qc', fontsize=18, weight='bold')
                        ax_qr[i].set_ylabel('Qr', fontsize=18, weight='bold')
                        ax_max[i].set_ylabel('max Qi', fontsize=18, weight='bold')
                        ax_mean[i].set_ylabel('mean Qc', fontsize=18, weight='bold')
                    else:
                        ax_qi[j, i].set_ylabel('Qi', fontsize=18, weight='bold')
                        ax_qc[j, i].set_ylabel('Qc', fontsize=18, weight='bold')
                        ax_qr[j, i].set_ylabel('Qr', fontsize=18, weight='bold')
                        ax_max[j, i].set_ylabel('max Qi', fontsize=18, weight='bold')
                        ax_mean[j, i].set_ylabel('mean Qc', fontsize=18, weight='bold')

            if len(n_temps) == 1:
                ax_qi.grid(True, which="both", ls="-")
                ax_qc.grid(True, which="both", ls="-")
                ax_qr.grid(True, which="both", ls="-")
                ax_max.grid(True, which="both", ls="-")
                ax_mean.grid(True, which="both", ls="-")
            
            else:
                if x == 1:
                    ax_qi[i].grid(True, which="both", ls="-")
                    ax_qc[i].grid(True, which="both", ls="-")
                    ax_qr[i].grid(True, which="both", ls="-")
                    ax_max[i].grid(True, which="both", ls="-")
                    ax_mean[i].grid(True, which="both", ls="-")
                else:
                    ax_qi[j, i].grid(True, which="both", ls="-")
                    ax_qc[j, i].grid(True, which="both", ls="-")
                    ax_qr[j, i].grid(True, which="both", ls="-")
                    ax_max[j, i].grid(True, which="both", ls="-")
                    ax_mean[j, i].grid(True, which="both", ls="-")

            if j == x-1:
                if len(n_temps) == 1:
                    ax_qi.set_xlabel('Drive power [dBm]', fontsize=18, weight='bold')
                    ax_qc.set_xlabel('Drive power [dBm]', fontsize=18, weight='bold')
                    ax_qr.set_xlabel('Drive power [dBm]', fontsize=18, weight='bold')
                    ax_max.set_xlabel('Resonance frequency [Hz]', fontsize=18, weight='bold')
                    ax_mean.set_xlabel('Resonance frequency [Hz]', fontsize=18, weight='bold')

                else:

                    if x == 1:
                        ax_qi[i].set_xlabel('Drive power [dBm]', fontsize=18, weight='bold')
                        ax_qc[i].set_xlabel('Drive power [dBm]', fontsize=18, weight='bold')
                        ax_qr[i].set_xlabel('Drive power [dBm]', fontsize=18, weight='bold')
                        ax_max[i].set_xlabel('Resonance frequency [Hz]', fontsize=18, weight='bold')
                        ax_mean[i].set_xlabel('Resonance frequency [Hz]', fontsize=18, weight='bold')
                    else:
                        ax_qi[j, i].set_xlabel('Drive power [dBm]', fontsize=18, weight='bold')
                        ax_qc[j, i].set_xlabel('Drive power [dBm]', fontsize=18, weight='bold')
                        ax_qr[j, i].set_xlabel('Drive power [dBm]', fontsize=18, weight='bold')
                        ax_max[j, i].set_xlabel('Resonance frequency [Hz]', fontsize=18, weight='bold')
                        ax_mean[j, i].set_xlabel('Resonance frequency [Hz]', fontsize=18, weight='bold')

                if t == len(n_temps)-1:
                    if len(n_temps) == 1:
                        ax_qi.legend(ncol=2)
                        ax_qc.legend(ncol=2)
                        ax_qr.legend(ncol=2)
                        #ax_max.legend(ncol=2)
                        #ax_mean.legend(ncol=2)

                    else:
                        if x == 1:
                            ax_qi[i].legend(ncol=2)
                            ax_qc[i].legend(ncol=2)
                            ax_qr[i].legend(ncol=2)
                            #ax_max[i].legend(ncol=2)
                            #ax_mean[i].legend(ncol=2)
                        else:
                            ax_qi[j, i].legend(ncol=2)
                            ax_qc[j, i].legend(ncol=2)
                            ax_qr[j, i].legend(ncol=2)
                            #ax_max[j, i].legend(ncol=2)
                            #ax_mean[j, i].legend(ncol=2)

            t += 1

        show()

        # Save figures
        fig_qr.savefig(self.work_dir+self.project_name+'/fit_res_results/summary_plots/Qr_vs_power.png')
        fig_qc.savefig(self.work_dir+self.project_name+'/fit_res_results/summary_plots/Qc_vs_power.png')
        fig_qi.savefig(self.work_dir+self.project_name+'/fit_res_results/summary_plots/Qi_vs_power.png')
        fig_max.savefig(self.work_dir+self.project_name+'/fit_res_results/summary_plots/Qi_max.png')
        fig_mean.savefig(self.work_dir+self.project_name+'/fit_res_results/summary_plots/Qc_mean.png')

    def plot_ts_summary(self, kid, temp, atten=None, ignore=[[0,1], [0]], cmap='viridis'):
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

        if atten is None:
            atten = self.overdriven

        if len(atten) <= 0: 
            msg('No overdriven attenuations available', 'fail')
            return 
        else:
            # G E N E R A T E   F I G U R E S
            # -------------------------------
            for k, kid in enumerate(kids):
                k2 = int(kid[1:])
                tmp = self._get_temps_to_sweep(temp, kid, mode='ts')[0]
                att = self._get_atten_to_sweep(atten[k2], tmp, kid, mode='ts')[0]

                if att in self.data['ts'][kid][tmp]:
                    low_cols.append( len(self.data['ts'][kid][tmp][att]['I_on'][0]) )
                    high_cols.append( len(self.data['ts'][kid][tmp][att]['I_on'][1]) )

                else:
                    low_cols.append(None)
                    high_cols.append(None)

            xl = len(kids)
            yl = np.max(low_cols)

            #print()

            fig_I_low, ax_I_low = subplots(xl, yl, sharey='row', figsize=(20,12))
            subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.08, hspace=0.1, wspace=0.035)
            fig_Q_low, ax_Q_low = subplots(xl, yl, sharey='row', figsize=(20,12))
            subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.08, hspace=0.1, wspace=0.035)

            xh = len(kids)
            yh = np.max(high_cols)
            ymax = 10

            figs_I, axs_I = [], []
            figs_Q, axs_Q = [], []
            for i in range(int(yh/ymax)):
                fig_I_high, ax_I_high = subplots(xh, ymax, sharey='row', figsize=(20,12))
                subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.08, hspace=0.1, wspace=0.035)
                figs_I.append(fig_I_high)
                axs_I.append(ax_I_high)

                fig_Q_high, ax_Q_high = subplots(xh, ymax, sharey='row', figsize=(20,12))
                subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.08, hspace=0.1, wspace=0.035)
                figs_Q.append(fig_Q_high)
                axs_Q.append(ax_Q_high)

            # -------------------------------

            fig_name = ""
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

                        if len(kids) == 1:
                            ax_I_low[ts].plot(ts_low, I_low, lw=0.75, color=cmap(norm_color(k)))
                            ax_Q_low[ts].plot(ts_low, Q_low, lw=0.75, color=cmap(norm_color(k)))

                            ax_I_low[ts].text(0.85, 0.85, str(ts), {'fontsize': 10, 'color': 'white'}, bbox=dict(facecolor='blue', alpha=0.95), transform=ax_I_low[ts].transAxes)
                            ax_Q_low[ts].text(0.85, 0.85, str(ts), {'fontsize': 10, 'color': 'white'}, bbox=dict(facecolor='blue', alpha=0.95), transform=ax_Q_low[ts].transAxes)

                            if ts == 0:
                                ax_I_low[ts].set_ylabel(kid+'\n I[V]')
                                ax_Q_low[ts].set_ylabel(kid+'\n Q[V]')

                            if k == len(kids)-1:
                                ax_I_low[ts].set_xlabel('\nTime[s]')
                                ax_Q_low[ts].set_xlabel(kid+'\nTime[s]')
                            else:
                                ax_I_low[ts].set_xticks([])
                                ax_Q_low[ts].set_xticks([])

                            if ts in ignore[0]:
                                ax_I_low[ts].patch.set_facecolor('red')
                                ax_I_low[ts].patch.set_alpha(0.2)

                                ax_Q_low[ts].patch.set_facecolor('red')
                                ax_Q_low[ts].patch.set_alpha(0.2)

                            else:
                                ax_I_low[ts].patch.set_facecolor('green')
                                ax_I_low[ts].patch.set_alpha(0.2)

                                ax_Q_low[ts].patch.set_facecolor('green')
                                ax_Q_low[ts].patch.set_alpha(0.2)

                        else:
                            ax_I_low[k, ts].plot(ts_low, I_low, lw=0.75, color=cmap(norm_color(k)))
                            ax_Q_low[k, ts].plot(ts_low, Q_low, lw=0.75, color=cmap(norm_color(k)))

                            ax_I_low[k, ts].text(0.85, 0.85, str(ts), {'fontsize': 10, 'color': 'white'}, bbox=dict(facecolor='blue', alpha=0.95), transform=ax_I_low[k, ts].transAxes)
                            ax_Q_low[k, ts].text(0.85, 0.85, str(ts), {'fontsize': 10, 'color': 'white'}, bbox=dict(facecolor='blue', alpha=0.95), transform=ax_Q_low[k, ts].transAxes)

                            if ts == 0:
                                ax_I_low[k, ts].set_ylabel(kid+'\n I[V]')
                                ax_Q_low[k, ts].set_ylabel(kid+'\n Q[V]')

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

                        if len(kids) == 1:
                            axs_I[cnt][m].plot(ts_high, I_high, lw=0.75, color=cmap(norm_color(k)))
                            axs_Q[cnt][m].plot(ts_high, Q_high, lw=0.75, color=cmap(norm_color(k)))

                            axs_I[cnt][m].text(0.85, 0.85, str(th), {'fontsize': 10, 'color': 'white'}, bbox=dict(facecolor='blue', alpha=0.95), transform=axs_I[cnt][m].transAxes)
                            axs_Q[cnt][m].text(0.85, 0.85, str(th), {'fontsize': 10, 'color': 'white'}, bbox=dict(facecolor='blue', alpha=0.95), transform=axs_I[cnt][m].transAxes)

                            if th == 0:
                                axs_I[cnt][m].set_ylabel(kid+'\n I[V]')
                                axs_Q[cnt][m].set_ylabel(kid+'\n Q[V]')

                            if k == len(kids)-1:
                                axs_I[cnt][m].set_xlabel('\nTime[s]')
                                axs_Q[cnt][m].set_xlabel(kid+'\nTime[s]')
                            else:
                                axs_I[cnt][m].set_xticks([])
                                axs_Q[cnt][m].set_xticks([])

                            if th in ignore[1]:
                                axs_I[cnt][m].patch.set_facecolor('red')
                                axs_I[cnt][m].patch.set_alpha(0.2)

                                axs_Q[cnt][m].patch.set_facecolor('red')
                                axs_Q[cnt][m].patch.set_alpha(0.2)

                            else:
                                axs_I[cnt][m].patch.set_facecolor('green')
                                axs_I[cnt][m].patch.set_alpha(0.2)

                                axs_Q[cnt][m].patch.set_facecolor('green')
                                axs_Q[cnt][m].patch.set_alpha(0.2)

                        else:                        
                            axs_I[cnt][k, m].plot(ts_high, I_high, lw=0.75, color=cmap(norm_color(k)))
                            axs_Q[cnt][k, m].plot(ts_high, Q_high, lw=0.75, color=cmap(norm_color(k)))

                            axs_I[cnt][k, m].text(0.85, 0.85, str(th), {'fontsize': 10, 'color': 'white'}, bbox=dict(facecolor='blue', alpha=0.95), transform=axs_I[cnt][k, m].transAxes)
                            axs_Q[cnt][k, m].text(0.85, 0.85, str(th), {'fontsize': 10, 'color': 'white'}, bbox=dict(facecolor='blue', alpha=0.95), transform=axs_I[cnt][k, m].transAxes)

                            if th == 0:
                                axs_I[cnt][k, m].set_ylabel(kid+'\n I[V]')
                                axs_Q[cnt][k, m].set_ylabel(kid+'\n Q[V]')

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

                fig_name += kid+'-'

            # Save figures
            fig_name = fig_name[:-1]
            kids_name = ""
                        
            for k in kids:
                kids_name += k + "-"
            self._create_diry(self.work_dir+self.project_name+'/fit_psd_results/'+kids_name+tmp+'-'+att)

            ioff()
            fig_I_low.savefig(self.work_dir+self.project_name+'/fit_psd_results/'+kids_name+tmp+'-'+att+'/I_low-ts-'+fig_name+'.png')
            close(fig_I_low)
            fig_Q_low.savefig(self.work_dir+self.project_name+'/fit_psd_results/'+kids_name+tmp+'-'+att+'/Q_low-ts-'+fig_name+'.png')
            close(fig_Q_low)

            # Save figures
            for c in range(len(figs_I)):
                figs_I[c].savefig(self.work_dir+self.project_name+'/fit_psd_results/'+kids_name+tmp+'-'+att+'/I_high-ts-'+fig_name+'-'+str(c)+'.png')
                close(figs_I[c])
                figs_Q[c].savefig(self.work_dir+self.project_name+'/fit_psd_results/'+kids_name+tmp+'-'+att+'/Q_high-ts-'+fig_name+'-'+str(c)+'.png')
                close(figs_Q[c])

            close("all")

    def plot_all_s21_kids(self, kid, temp, atten, sample=0, over_attens=True, data_source='vna', cmap='jet', fig_name=None):
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
        yg = int(np.ceil(len(kids)/xg))

        fig, axm = subplots(xg, yg, figsize=(20,12))
        subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.08, hspace=0.15)#, wspace=0.035)
        for k in range(xg*yg):

            i = k%yg
            j = int(k/yg)

            if k < len(kids):

                kid = kids[k]

                temps = self._get_temps_to_sweep(temp, kid)
                norm_color = matplotlib.colors.Normalize(vmin=0, vmax=len(temps))

                for t, tm in enumerate(temps):
                    
                    if over_attens:
                        att = self.overdriven[k]
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

                            elif sweep_case == 2:
                                alpha = 1.0
                                single_color = cmap(norm_color(a))

                            else:
                                alpha = 1.0
                                single_color = 'r'

                            if data_source == 'vna':
                                f, s21 = self.data['vna'][kid][tm][single_atten]['data'][sample]
                                
                                if self.data_type.lower() == 'dark':                              
                                    real_temp = f'{self.data['vna'][kid][tm][single_atten]['header'][sample]['SAMPLETE']*1e3:.0f}mK'
                                elif self.data_type.lower() == 'blackbody':                              
                                    real_temp = f'{self.data['vna'][kid][tm][single_atten]['header'][sample]['BLACKBOD']:.1f}K'

                            elif data_source == 'homo':
                                f = self.data['ts'][kid][tm][single_atten]['f']
                                s21 = self.data['ts'][kid][tm][single_atten]['s21']
                                if self.data_type.lower() == 'dark':                              
                                    real_temp = f'{self.data['ts'][kid][tm][single_atten]['hdr_on'][0]['SAMPLETE']*1e3:.0f}mK'
                                elif self.data_type.lower() == 'blackbody':                              
                                    real_temp = f'{self.data['ts'][kid][tm][single_atten]['hdr_on'][0]['BLACKBOD']:.1f}K'

                            if False:
                                add = -20+2.5+0.5
                            else:
                                add = 0
                                
                            if xg == 1 and yg == 1:
                                axm.plot(f/1e6, 20*np.log10(np.abs(s21))+add, color=single_color, alpha=alpha, lw=1.75, label=real_temp)
                            elif xg == 1 or yg == 1:
                                axm[i].plot(f/1e6, 20*np.log10(np.abs(s21))+add, color=single_color, alpha=alpha, lw=1.75, label=real_temp)
                            else:
                                axm[j,i].plot(f/1e6, 20*np.log10(np.abs(s21))+add, color=single_color, alpha=alpha, lw=1.75, label=real_temp)
                            
                            if t == 0 and a == len(attens)-1:
                                if xg == 1 and yg == 1:
                                    axm.text(f[0]/1e6+0.65*(f[-1]-f[0])/1e6, np.min(20*np.log10(np.abs(s21)))+add+ \
                                                0.1*(np.max(20*np.log10(np.abs(s21)))-np.min(20*np.log10(np.abs(s21)))), \
                                                kid+'\n'+single_atten[1:]+'dB', {'fontsize':17, 'color':'blue'})
                                                                
                                elif xg == 1 or yg == 1:
                                    axm[i].text(f[0]/1e6+0.65*(f[-1]-f[0])/1e6, np.min(20*np.log10(np.abs(s21)))+add+ \
                                                0.1*(np.max(20*np.log10(np.abs(s21)))-np.min(20*np.log10(np.abs(s21)))), \
                                                kid+'\n'+single_atten[1:]+'dB', {'fontsize':17, 'color':'blue'})
                                else:
                                    axm[j,i].text(f[0]/1e6+0.65*(f[-1]-f[0])/1e6, np.min(20*np.log10(np.abs(s21)))+add+ \
                                                0.1*(np.max(20*np.log10(np.abs(s21)))-np.min(20*np.log10(np.abs(s21)))), \
                                                kid+'\n'+single_atten[1:]+'dB', {'fontsize':17, 'color':'blue'})

                            if i == 0:
                                if xg == 1 and yg == 1:
                                    axm.set_ylabel('S21 [dB]', fontsize=18, weight='bold')
                                elif xg == 1 or yg == 1:
                                    axm[i].set_ylabel('S21 [dB]', fontsize=18, weight='bold')
                                else:
                                    axm[j,i].set_ylabel('S21 [dB]', fontsize=18, weight='bold')
                            if j == xg-1:
                                if xg == 1 and yg == 1:
                                    axm.set_xlabel('Frequency [MHz]', fontsize=18, weight='bold')
                                elif xg == 1 or yg == 1:
                                    axm[i].set_xlabel('Frequency [MHz]', fontsize=18, weight='bold')
                                else:
                                    axm[j,i].set_xlabel('Frequency [MHz]', fontsize=18, weight='bold')

                        except Exception as e:
                            msg('Error plotting data\n'+str(e), 'warn')

            else:
                if xg == 1 and yg == 1:
                    axm.axis('off')
                elif xg == 1 or yg == 1:
                    axm[i].axis('off')
                else:
                    axm[j,i].axis('off')

            if k == len(kids)-1:
                if xg == 1 and yg == 1:
                    axm.legend(ncol=2)
                elif xg == 1 or yg == 1:
                    axm[i].legend(ncol=2)
                else:
                    axm[j,i].legend(ncol=2)

        show()

        # Save figures
        if fig_name is None:
            fig_name = 'S21_per_kid'

        fig.savefig(self.work_dir+self.project_name+'/fit_res_results/summary_plots/'+fig_name+'.png')


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

        kids = self._get_kids_to_sweep(kid)
        for kid in kids:

            msg(kid, 'info')

            temps = self._get_temps_to_sweep(temp, kid)

            for t, tmp in enumerate(temps):

                fig, axs = subplots(1, 2, figsize=(20,12))
                subplots_adjust(left=0.05, right=0.99, top=0.97, bottom=0.08, wspace=0.12)

                norm_color = matplotlib.colors.Normalize(vmin=0, vmax=len(temps))

                attens = self._get_atten_to_sweep(atten, tmp, kid)
                if len(attens) > 1:
                    norm_color = matplotlib.colors.Normalize(vmin=0, vmax=len(attens))
                    sweep_case = 2                
                elif len(temps) > 1:
                    alphas = np.linspace(1.0, 0.3, len(attens))
                    norm_color = matplotlib.colors.Normalize(vmin=0, vmax=len(temps))
                    sweep_case = 1
                else:
                    sweep_case = 3

                for a, att in enumerate(attens):
                    try:
                        """
                        if sweep_case == 1:
                            alpha = 1.0 #alphas[a]
                            single_color = cmap(norm_color(t))
                            plot_title = kid
                            curve_label = tmp+'-'+att
                        """
                        if sweep_case == 1 or sweep_case == 2:
                            alpha = 1.0
                            single_color = cmap(norm_color(a))
                            plot_title = kid+'-'+tmp
                            curve_label = att
                        else:
                            alpha = 1.0
                            single_color = 'r'
                            plot_title = kid+'-'+tmp+'-'+att
                            curve_label = plot_title

                        if data_source == 'vna':
                            f, s21 = self.data['vna'][kid][tmp][att]['data'][sample]
                        elif data_source == 'homo':
                            f = self.data['ts'][kid][tmp][att]['f']
                            s21 = self.data['ts'][kid][tmp][att]['s21']

                        axs[0].plot(f/1e6, 20*np.log10(np.abs(s21)), color=single_color, alpha=alpha, lw=1.75 )
                        if fit and 'fit' in self.data['vna'][kid][tmp][att]:
                            f_fit = self.data['vna'][kid][tmp][att]['fit'][sample]['freq_data']
                            s21_fit = self.data['vna'][kid][tmp][att]['fit'][sample]['fit_data']
                            axs[0].plot(f_fit/1e6, 20*np.log10(np.abs(s21_fit)), '-', color='k', lw=1.25 )
                        
                        axs[0].set_title(plot_title, fontsize=18, weight='bold')
                        axs[0].set_xlabel('Frequency [MHz]', fontsize=18, weight='bold')
                        axs[0].set_ylabel('S21 [dB]', fontsize=18, weight='bold')

                        axs[1].plot(s21.real, s21.imag, color=single_color, alpha=alpha, label=curve_label, lw=1.75)
                        if fit and 'fit' in self.data['vna'][kid][tmp][att]:
                            axs[1].plot(s21_fit.real, s21_fit.imag, '-', color='k', lw=1.25 )
                        axs[1].set_title(plot_title, fontsize=18, weight='bold')
                        axs[1].axis('equal')
                        axs[1].set_xlabel('I[V]', fontsize=18, weight='bold')
                        axs[1].set_ylabel('Q[V]', fontsize=18, weight='bold')

                    except Exception as e:
                        msg('Error plotting data\n'+str(e), 'warn')

                axs[1].legend()

                # Save figures
                fig_name = 'S21_'+plot_title
                fig.savefig(self.work_dir+self.project_name+'/fit_res_results/summary_plots/'+fig_name+'.png')
                close(fig)

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
        Parameters
        ----------
        kid : int/list
            Detectors as a list of ints, single ints
            or None for all the objects.
        ----------
        """

        if kid is None:
            vna_keys = self.data[mode].keys()
            kids = [ x for x in vna_keys if 'K' in x ]

        elif isinstance(kid, int):
            kids = ['K'+str(kid).zfill(3)]
        
        elif isinstance(kid, list):
            kids = []
            for k in kid:
                if isinstance(k, int):
                    kids.append('K'+str(k).zfill(3))
                else:
                    kids.append(k)
        
        kids = sorted(kids)

        return kids

    def _get_temps_to_sweep(self, temp, kid=None, mode='vna', vna_type='seg'):
        """
        Get the temperatures to sweep.
        Parameters
        ----------
        temp : int/list
            Base/Blackbody temperatures as a list of 
            ints, single ints or None for all the objects.
        kid(opt) : int/list
            Detectors as a list of ints, single ints
            or None for all the objects.
        ----------
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
                    temps.append(type_data+str(t).zfill(nzeros))
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
        Parameters
        ----------
        atten : int/list
            Input attenuations as a list of ints,
            single ints or None for all the objects.
        temp(opt) : int/list
            Base/Blackbody temperatures as a list of 
            ints, single ints or None for all the objects.
        kid(opt) : int/list
            Detectors as a list of ints, single ints
            or None for all the objects.
        ----------
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

        elif isinstance(atten, float):
            attens = [f'A{atten:.1f}']

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
        print(kid+','+temp+','+atten)
        fitRes[kid+','+temp+','+atten] = fit_res

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
            msg('Data not identified in the name file.\n'+str(e), 'fail')

        return None, None, None

    def _get_meas_char_from_foldername(self, foldername):
        """
        Get the date, analysis type and sample from the folder name.
        Parameters
        ----------
        foldername : string
            Foldername from where data is extracted.
        ----------
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
