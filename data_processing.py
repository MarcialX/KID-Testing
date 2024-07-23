# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------- #
# MUSCAT data reduction
# data_processing.py
# Functions to process time streams data
#
# Marcial Becerril, @ 16 January 2022
# Latest Revision: 16 Jan 2022, 21:49 GMT-6
#
# TODO list:
# Functions missing:
#	+ Savgol filter
#	+ PCA filter
#	+ High/low-pass filter
#	+ Stop band filters (60 Hz signal, PTC, etc.)
#
# For all kind of problems, requests of enhancements and bug reports, please
# write to me at:
#
# mbecerrilt92@gmail.com
# mbecerrilt@inaoep.mx
#
# --------------------------------------------------------------------------------- #


import numpy as np
import time

from random import seed
from random import random

from scipy import signal
from scipy.signal import butter, lfilter

from matplotlib.pyplot import *

from misc.msg_custom import *


# C O S M I C   R A Y   F I L T E R
# ---------------------------------
def cr_filter(stream, win_size=100, sigma_thresh=4, peak_pts=5, verbose=True):
	"""
	Cosmic Ray filter
	Remove glitches and likely cosmic ray events.
	Parameters
	----------
	Input:
		stream : array
			Time stream array
		win_size : int
			Window size filter
		sigma_thresh : float
			Sigma threshold
	Output:
		stream_filter : array
			Time stream data filtered
	----------
	"""

	cr_events = 0

	stream_filter = np.copy(stream)

	if verbose:
		msg('Starting Cosmic Ray removal procedure...', 'info')

	prev_cr_pts = np.array([])

	start_time = time.time()
	check_time = time.time()

	while True:
		# Derivate data
		data_win_diff = np.diff(stream_filter)
		# Sigma diff
		sigma_diff = np.nanstd(data_win_diff)
		# Mean diff
		offset_diff = np.nanmean(data_win_diff)

		# Cosmic ray events
		cr_idx = np.where((data_win_diff > offset_diff+sigma_thresh*sigma_diff) |
						   (data_win_diff < offset_diff-sigma_thresh*sigma_diff) )[0]

		num_cr_pts = len(cr_idx)

		if check_time - start_time > 10:
			break

		if num_cr_pts <= 0 or np.array_equal(prev_cr_pts, cr_idx):
			break

		if verbose:
			msg('Cosmic ray events: '+str(num_cr_pts), 'info')

		# Get statistics per each point
		for cr in cr_idx:
			data_win = stream_filter[cr-int(win_size/2):cr+int(win_size/2)]
			sort_data = np.sort(data_win)
			edge_data = sort_data[:int(3*win_size/4)]
			# Sigma window
			sigma = np.std(edge_data)
			# Data offset
			offset = np.mean(edge_data)

			#plot(data_win)
			#axhline(offset+sigma_thresh*sigma, color='r')
			#axhline(offset-sigma_thresh*sigma, color='k')
			#print(data_win[0], offset+sigma_thresh*sigma)
			#break

			if len(data_win)>0:
				if np.abs(data_win[int(win_size/2)]) < np.abs(offset):
					# Replace peak as the middle point between the neighbours
					stream_filter[cr] = (stream_filter[cr+1]+stream_filter[cr-1])/2

				else:
					# Validate
					cr_rec = np.where((data_win > offset+sigma_thresh*sigma) |
								   (data_win < offset-sigma_thresh*sigma) )[0]

					diff_cr_rec = np.diff(cr_rec)

					if np.count_nonzero(diff_cr_rec == 1) < peak_pts:
						# Replace point
						# ----------------
						# New random points with normal distribution
						new_sample = np.random.normal(offset, sigma)
						# Update points
						stream_filter[cr] = new_sample
						# ----------------
					else:
						stream_filter[cr] = (stream_filter[cr+1]+stream_filter[cr-1])/2


		check_time = time.time()

		prev_cr_pts = cr_idx

	if verbose:
		msg('Done', 'info')

	return stream_filter, cr_events


# F I L T E R S
# ---------------------------------
def _butter_hipass_model(cutoff, fs, order=5):
	"""
	Butterworth high-pass model.
	Parameters
	----------
	cutoff : float
		Frequency cutoff.
	fs : float 
		Sampling frequency.
	order : int
		Filter order.
	----------
	"""
	nyq = 0.5 * fs
	normal_cutoff = cutoff / nyq
	b, a = butter(order, normal_cutoff, btype='high', analog=False)

	return b, a

def butter_hipass_filter(data, cutoff, fs, order=5):
	"""
	Apply butterworth high-pass filter.
	Parameters
	----------
	data : array
		Raw data.
	cutoff : float
		Frequency cutoff.
	fs : float 
		Sampling frequency.
	order : int
		Filter order.
	----------
	"""
	b, a = _butter_hipass_model(cutoff, fs, order=order)
	y = lfilter(b, a, data)
	
	return y

def _sinc_lopass_model(fc=0.1, b=0.08):
	"""
	Windowed-sinc low-pass model.
	Parameters
	----------
	fc : float
		Frequency cutoff.
	b : float
		b parameter.
	----------
	"""

	N = int(np.ceil((4 / b)))

	if not N % 2: N += 1  # Make sure that N is odd.
	n = np.arange(N)

	# Compute sinc filter.
	h = np.sinc(2 * fc * (n - (N - 1) / 2))

	# Compute Blackman window.
	w = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + \
	    0.08 * np.cos(4 * np.pi * n / (N - 1))

	# Multiply sinc filter by window.
	h = h * w

	# Normalize to get unity gain.
	h = h / np.sum(h)

	return h

def sinc_lopass_filter(data, fc, b):
	"""
	Apply sinc low pass filter.
	Parameters
	----------
	data : array
		Raw data array.
	fc : float
		Frequency cutoff.
	b : float
		b parameter.
	----------
	"""
	
	h = _sinc_lopass_model(fc, b)
	filter_data = np.convolve(data, h)
	filter_data = filter_data[int((len(h)-1)/2):-int((len(h)-1)/2)]
	
	return filter_data

def _notch_model(fs, f0, Q):
	"""
	Notch filter model.
	Parameters
	------------
	Input:
		fs: sample frequency
		f0: Frequency to remove
		Q: Quality factor
	Output:
		filter_signal: Filtered signal
	"""

	b, a = signal.iirnotch(f0, Q, fs)
	return b, a

def notch_filter(data, fs, f0, Q):
	"""
	Apply notch filter.
	Parameters
	------------
	Input:
		fs: sample frequency
		f0: Frequency to remove
		Q: Quality factor
	Output:
		filter_signal: Filtered signal
	"""

	b, a = _notch_model(fs, f0, Q)
	filter_data = signal.filtfilt(b, a, data)
	
	return filter_data

# C I R C L E   D E R O T A T I O N
# ---------------------------------
def fitCirc(x, y):
    """
	Fit the IQ circle.
	This code belongs to Andreas Papagiorgou.
	Parameters
	----------
	x, y : array
		I/Q sweep frequency.
	----------
    """

    x_m = np.mean(x)
    y_m = np.mean(y)

    # calculation of the reduced coordinates
    u = x - x_m
    v = y - y_m

    # linear system defining the center (uc, vc) in reduced coordinates:
    #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
    Suv  = np.sum(u*v)
    Suu  = np.sum(u**2)
    Svv  = np.sum(v**2)
    Suuv = np.sum(u**2 * v)
    Suvv = np.sum(u * v**2)
    Suuu = np.sum(u**3)
    Svvv = np.sum(v**3)

    # Solving the linear system
    A = np.array([ [ Suu, Suv ], [Suv, Svv]])
    B = np.array([ Suuu + Suvv, Svvv + Suuv ])/2.0
    uc, vc = np.linalg.solve(A, B)

    xc_1 = x_m + uc
    yc_1 = y_m + vc

    # Calculate distances from centre (xc_1, yc_1)
    Ri_1     = np.sqrt((x-xc_1)**2 + (y-yc_1)**2)
    R_1      = np.mean(Ri_1)
    residu_1 = np.sum((Ri_1-R_1)**2)

    return xc_1, yc_1, R_1
