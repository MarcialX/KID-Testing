# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# Interpreter to run KID-Testing analysis tasks
# do_task.py
#
# Marcial Becerril, @ 10 Jul 2024
# Latest Revision: 10 Jul 2024, 13:06 GMT-6
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# Becerril-TapiaM@cardiff.ac.uk
#
# --------------------------------------------------------------------------------- #

import os
import numpy as np 

import argparse
import yaml

from homodyne import *

from misc.msg_custom import *


# G L O B A L   P A R A M E T E R S
# --------------------------------------------------
PARAMS_FILE = "params.yaml"
BACKUP_FILE = "proj_bkp"


# R E A D   A R G U M E N T S
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--tasks', '-t', help="Tasks file", type=str, required=True)

args = parser.parse_args()

tasks = args.tasks

# Loading tasks file
# --------------------------------------------------
msg('Loading tasks file', 'info')
try:
    with open(tasks) as file:
        tasks_data = yaml.safe_load(file)
    msg('Tasks file loaded', 'ok')
except Exception as e:
    msg('Fail loading tasks file. '+str(e), 'fail')

# Data selection
# --------------------------------------------------
TEMPS = tasks_data['TEMPS']
ATTS = tasks_data['ATTS']
KIDS = tasks_data['KIDS']

SAMPLES = tasks_data['SAMPLES']

LOAD_PROJ = tasks_data['LOAD_PROJ']
SAVE_PROJ = tasks_data['SAVE_PROJ']

# In case all the resonators are selected this is the
# general definition. However, it changes slightly for
# task: "fit_res"
if KIDS == "all":
    kids = None 
if TEMPS == "all":
    temps = None 
if ATTS == "all":
    atts = None
if SAMPLES == "all":
    samples = None

# Load gral params file
msg('Loading general params file', 'info')
try:
    with open(PARAMS_FILE) as file:
        gral_params = yaml.safe_load(file)
    msg('General params file loaded', 'ok')
except Exception as e:
    msg('Fail loading general params file. '+str(e), 'fail')


print(gral_params)

data_path = gral_params['DATA_FOLDER']
project_name = gral_params['PROJECT_NAME']
project_path = gral_params['PROJECT_FOLDER']

atts_overdriven = gral_params['ATT_OVRDRN']
add_in_atten = gral_params['IN_ADD_ATT']
add_out_atten = gral_params['OUT_ADD_ATT']

# R U N   T A S K S
# --------------------------------------------------
# Create the homodyne object
if LOAD_PROJ:
    used_project = project_path 
else: 
    used_project = data_path

h = Homodyne(used_project, work_dir=project_path, proj_name=project_name, load_saved=LOAD_PROJ, overdriven=atts_overdriven, \
            add_in_atten=add_in_atten, add_out_atten=add_out_atten)

TASKS = tasks_data['TASKS']

for s, step in enumerate(TASKS):

    # Task name
    task_name = TASKS[step]['name']
    # Task parameters
    if 'params' in TASKS[step]:
        task_params = TASKS[step]['params']

    print("T A S K : ", task_name)

    # Run the step
    # ---------------------------------
    # ---> Fit resonator
    if task_name == "fit_res":
        
        type_data = task_params['type']
        n = task_params['n']

        if KIDS == "all":
            kids = None 
            kid = 'K000'

        if TEMPS == "all":
            temps = h.data[type_data][kid].keys()
        else:
            temps = TEMPS

        for temp in temps:
            if ATTS == "all":
                attens = h.data[type_data][kid][temp].keys()
            else:
                attens = ATTS

            for atten in attens:
                if SAMPLES == "all":
                    samples = h.data[type_data][kid][temp][atten]['data'].keys()
                else:
                    samples = SAMPLES

                for sample in samples:
                    if type_data == 'vna':
                        h.fit_vna_resonators(kids, temp, atten, sample=sample, n=n, verbose=True)

    # ---> Load fit
    elif task_name == "load_fit_res":

        path = task_params['path']
        if path == "":
            path = project_path+'/'+project_name

        type_data = task_params['type']

        h.load_fit(folder=path, data_type=type_data)     

    # ---> Merge sample results
    elif task_name == "merge_vna":

        xls_report = task_params['xls_report']

        h.merge_fit_res(kids, temps, atts, samples)

        if xls_report:
            h.vna_xls_report()

    # ---> Get overdriven attenuations
    elif task_name == "find_overdriven":

        ref_temp = task_params['temp']
        ref_sample = task_params['sample']
        non_thresh = task_params['thresh']

        h.find_overdriven_atts(ref_temp, sample=ref_sample, thresh=non_thresh)

        # Save atts in params.yaml file
        with open(project_path+project_name+'/'+PARAMS_FILE, 'r') as f:
            lines = f.readlines()
            updated_lines = []
            str_atts = ','.join(str(x) for x in h.overdriven)
            for line in lines:
                if "ATT_OVRDRN" in line:
                    line = line[:line.index(":") + 1] + ' [' + str_atts + ']\n'
            
                updated_lines.append(line)

        with open(project_path+project_name+'/'+PARAMS_FILE, 'w') as wf:
            for line in updated_lines:
                wf.write(line)


    # ---> Summary plots pt. 1
    elif task_name == "summary_plots_1":

        plot_name = TASKS[step]['plot']
        for name in plot_name:
            # Get params
            the_args = {}
            plot_params = plot_name[name]
            if name == 'Q_vs_pwr':
                # Quality factors vs drive power
                if 'cmap' in plot_params:
                    the_args['cmap'] = plot_params['cmap']
                h.plot_qs_vs_drive_power(kids, temps, atts, **the_args)

            elif name == 's21':
                # Plot all the S21 for all the KIDs
                for p in plot_params.keys():
                    if p == 'sample':
                        the_args['sample'] = plot_params['sample']
                    elif p == 'cmap':
                        the_args['cmap'] = plot_params['cmap']
                    elif p == 'fig_name':
                        the_args['fig_name'] = plot_params['fig_name']        
                    elif p == 'data_source':
                        the_args['data_source'] = plot_params['data_source']   
                    elif p == 'over_attens':
                        the_args['over_attens'] = plot_params['over_attens']   

                h.plot_all_s21_kids(kids, temps, atts, **the_args)

            elif name == 's21_per_kid':
                # Plot all the S21 for all the KIDs
                for p in plot_params.keys():
                    if p == 'sample':
                        the_args['sample'] = plot_params['sample']
                    elif p == 'fit':
                        the_args['fit'] = plot_params['fit']            
                    elif p == 'data_source':
                        the_args['data_source'] = plot_params['data_source']

                h.plot_s21_kid(kids, temps, atts, **the_args)





