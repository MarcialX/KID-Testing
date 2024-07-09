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


import numpy as _np
import time

from tqdm import tqdm
from random import seed
from random import random

from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

from scipy.signal import butter, lfilter, freqz
from scipy import signal
from scipy import optimize
from scipy.optimize import curve_fit

from matplotlib.pyplot import *

from misc.msg_custom import *



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
	data_len = len(stream)

	stream_filter = _np.copy(stream)

	# Get stats moments of general data
	gral_std = _np.nanstd(stream_filter)
	gral_mean = _np.nanmean(stream_filter)

	if verbose:
		msg('Starting Cosmic Ray removal procedure...', 'info')

	prev_cr_pts = _np.array([])

	start_time = time.time()
	check_time = time.time()

	while True:
		# Derivate data
		data_win_diff = _np.diff(stream_filter)
		# Sigma diff
		sigma_diff = _np.nanstd(data_win_diff)
		# Mean diff
		offset_diff = _np.nanmean(data_win_diff)

		# Cosmic ray events
		cr_idx = _np.where((data_win_diff > offset_diff+sigma_thresh*sigma_diff) |
						   (data_win_diff < offset_diff-sigma_thresh*sigma_diff) )[0]

		num_cr_pts = len(cr_idx)

		if check_time - start_time > 10:
			break

		if num_cr_pts <= 0 or _np.array_equal(prev_cr_pts, cr_idx):
			break

		if verbose:
			msg('Cosmic ray events: '+str(num_cr_pts), 'info')

		# Get statistics per each point
		for cr in cr_idx:
			data_win = stream_filter[cr-int(win_size/2):cr+int(win_size/2)]
			sort_data = _np.sort(data_win)
			edge_data = sort_data[:int(3*win_size/4)]
			# Sigma window
			sigma = _np.std(edge_data)
			# Data offset
			offset = _np.mean(edge_data)

			#plot(data_win)
			#axhline(offset+sigma_thresh*sigma, color='r')
			#axhline(offset-sigma_thresh*sigma, color='k')
			#print(data_win[0], offset+sigma_thresh*sigma)
			#break

			if len(data_win)>0:
				if _np.abs(data_win[int(win_size/2)]) < _np.abs(offset):
					# Replace peak as the middle point between the neighbours
					#print('Cosmic ray close to source')
					stream_filter[cr] = (stream_filter[cr+1]+stream_filter[cr-1])/2

				else:
					#print('Typical')
					# Validate
					cr_rec = _np.where((data_win > offset+sigma_thresh*sigma) |
								   (data_win < offset-sigma_thresh*sigma) )[0]

					diff_cr_rec = _np.diff(cr_rec)

					if _np.count_nonzero(diff_cr_rec == 1) < peak_pts:
						# Replace point
						# ----------------
						# New random points with normal distribution
						new_sample = _np.random.normal(offset, sigma)
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


def butter_hipass(cutoff, fs, order=5):
	nyq = 0.5 * fs
	normal_cutoff = cutoff / nyq
	b, a = butter(order, normal_cutoff, btype='high', analog=False)
	return b, a


def butter_hipass_filter(data, cutoff, fs, order=5):
	b, a = butter_hipass(cutoff, fs, order=order)
	y = lfilter(b, a, data)
	return y

# Windowed-sinc filter. LOW-PASS
def sinc_lopass(data, fc=0.1, b=0.08):
	"""
		Windowed-sinc low pass filter
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
		Apply sinc low pass filter
	"""
	h = sinc_lopass(fc, b)
	filter_data = np.convolve(data, h)
	filter_data = filter_data[int((len(h)-1)/2):-int((len(h)-1)/2)]
	return filter_data


def notch(fs, f0, Q):
	"""
		Notch filter
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
	b, a = notch(fs, f0, Q)
	filter_data = signal.filtfilt(b, a, data)
	return filter_data


def pro_sv_filter(stream, *args, **kwargs):
	"""
		Through Savinsky-Golay algorithm this function substrate the baseline.
	"""

	# Key arguments
	# ----------------------------------------------
	# Distance between lines
	window = kwargs.pop('window', 501)
	# Minimum height for the lines
	order = kwargs.pop('order', 3)
	# Linewidth definition
	# 	- # of points for one linewdith
	lw = kwargs.pop('lw', 50)
	#	- Signal linewidths
	#	- Noise linewidths
	sg_lw = kwargs.pop('sg_lw', 4)
	ns_lw = kwargs.pop('ns_lw', 2)
	# ----------------------------------------------

	# 1. Find the peaks associated to a detection
	# Data has to be free os spikes signals!
	height = savgol_filter(stream, 1001, 3)
	scale = 6*_np.nanstd(stream[:100]) + height

	peaks = find_profile_peaks(stream, height=scale, **kwargs)
	print(len(peaks), ' peaks detected')

	# 2. Remove those detected signals
	noise_data = extract_noise(stream, peaks, lw=lw, sg_lw=sg_lw, ns_lw=ns_lw)

	# 3. Get baseline from the noise
	base_noise_signal = savgol_filter(noise_data, window, order)

	noise_data = noise_data - base_noise_signal

	# 4. Remove baseline from the signal
	signal_data = stream - base_noise_signal

	return noise_data, signal_data


def extract_noise(stream, peaks, *args, **kwargs):
	"""
		Extracting noise
	"""

	# Key arguments
	# ----------------------------------------------
	# Linewidth definition
	# 	- # of points for one linewdith
	lw = kwargs.pop('lw', 50)
	#	- Signal linewidths
	#	- Noise linewidths
	sg_lw = kwargs.pop('sg_lw', 4)
	ns_lw = kwargs.pop('ns_lw', 2)
	# ----------------------------------------------

	# Noise processing
	noise_data = _np.copy(stream)
	noise_raw = _np.copy(stream)

	from_sm = peaks - int(lw*(sg_lw+ns_lw)/2)
	to_sm = peaks + int(lw*(sg_lw+ns_lw)/2)

	from_sg = peaks - int(lw*sg_lw/2)
	to_sg = peaks + int(lw*sg_lw/2)

	med_ts = _np.median(stream)

	for i, peak in enumerate(peaks):

		if stream[peak]>med_ts:
			# a. Get sample
			sample = stream[from_sm[i]:to_sm[i]]
			x = _np.arange(from_sm[i], to_sm[i])

			# b. Remove signal and fit a line with the noise
			no_signal = _np.concatenate(( sample[:from_sg[i]-from_sm[i]], sample[to_sg[i]-from_sm[i]:] ))
			x_no_signal = _np.concatenate(( x[:from_sg[i]-from_sm[i]], x[to_sg[i]-from_sm[i]:] ))

			if len(no_signal) == len(x_no_signal):
				# Get sample baseline
				z = _np.polyfit(x_no_signal, no_signal, 1)
				p = _np.poly1d(z)

				sample_base = p(x_no_signal)

				# c. Replace signal data with noise
				corr_base = no_signal - sample_base

				mean_ns = _np.nanmean(corr_base)
				std_ns = _np.nanstd(corr_base)

				# Generate the replacement signal
				noise_signal = _np.random.normal(mean_ns, std_ns, to_sg[i]-from_sg[i])
				x_signal = _np.arange(from_sg[i], to_sg[i])

				signal_base = p(x_signal)

				# Replace sample
				noise_data[from_sg[i]:to_sg[i]] = noise_signal + signal_base
				noise_raw[from_sg[i]:to_sg[i]] = _np.nan*noise_raw[from_sg[i]:to_sg[i]]

	return noise_data, noise_raw


def edge_noise(stream, *args, **kwargs):
	"""
		Get noise from edges of detections.
	"""

	# Key arguments
	# ----------------------------------------------
	# Distance between lines
	window = kwargs.pop('window', 1001)
	# Minimum height for the lines
	order = kwargs.pop('order', 3)
	# Linewidth definition
	# 	- # of points for one linewdith
	lw = kwargs.pop('lw', 50)
	#	- Signal linewidths
	#	- Noise linewidths
	sg_lw = kwargs.pop('sg_lw', 4)
	ns_lw = kwargs.pop('ns_lw', 4)
	# ----------------------------------------------

	# 1. Find the peaks associated to a detection
	# Data has to be free os spikes signals!
	height = savgol_filter(stream, window, order)
	scale = 6*_np.nanstd(stream[:100]) + height

	peaks = find_profile_peaks(stream, height=scale, **kwargs)
	print(len(peaks), ' peaks detected')

	# 2. Remove those detected signals
	noise_data = _np.copy(stream)

	from_sm = peaks - int(lw*(sg_lw+ns_lw)/2)
	to_sm = peaks + int(lw*(sg_lw+ns_lw)/2)

	from_sg = peaks - int(lw*sg_lw/2)
	to_sg = peaks + int(lw*sg_lw/2)

	noise = _np.array([])
	for i, peak in enumerate(peaks):
		# a. Get sample to get the noise
		sample = _np.concatenate((stream[from_sm[i]:from_sg[i]], stream[to_sm[i]:to_sg[i]]))
		noise = _np.concatenate((noise, sample))

	return noise


def find_profile_peaks(data_array, **kwargs):
    """
        Find profile peaks
        Parameters
        ----------
            data_array : array
            dist : float
                Minimum distance between peaks
            height_div : float
                Factor which divide the maximum value of the array, to define the
                minimum peak detectable
        ----------
    """

    # Find peaks keyword parameters
    # Distance between lines
    dist = kwargs.pop('dist', 250.0)
    # Minimum height for the lines
    height = kwargs.pop('height', 10.0)

    #print(dist, height)

    # Height division
    peaks, _ = find_peaks(data_array, distance=dist, height=height)

    #print(peaks)

    return peaks


def cleanPca(data, nComps, time_chunck, all_data=False, equal_segments=True, comps_detector=None, test_pca=False):
	"""
		PCA cleaning. Designed and coded by Andreas Papageorgiou
		Parameters
		----------
		inData: 2d array
			Input data from several KIDs
		nComps: float
			Number of components to substract
		time_chunck: int
			Time chunck length
	    ----------
	"""

	out_data = _np.zeros_like(data)

	if all_data:
		time_chunck = data.shape[1]
		n_runs = int(_np.ceil(data.shape[1]/time_chunck))

	# Check if it is an analysis using equal segments or time defined chunk
	else:
		if equal_segments:
			time_chunck = int(data.shape[1]/time_chunck)
			n_runs = int(data.shape[1]/time_chunck)
		else:
			n_runs = int(_np.ceil(data.shape[1]/time_chunck))

	nDet = data.shape[0]

	print('Times to run PCA: ', n_runs)

	for run in range(n_runs):

		# Choose the time chunk data
		if equal_segments:
			if (run+1) == n_runs:
				end_tchunck = data.shape[1]
			else:
				end_tchunck = (run+1)*time_chunck

		else:
			if (run+1)*time_chunck >= data.shape[1]:
				end_tchunck = data.shape[1]
			else:
				end_tchunck = (run+1)*time_chunck

		print("From: ", run*time_chunck, " to: ", end_tchunck)

		inData = data[:, run*time_chunck:end_tchunck]

		tmpStd = _np.nanstd(inData, axis=1)
		tmpStd[tmpStd == 0] = 1
		inData = (inData.T/tmpStd).T

		pca = PCA(n_components=nComps, svd_solver='full')

		t0 = time.time()
		Y = pca.fit(inData.T)
		t1 = time.time()
		print('end pca', "%.2fs"%(t1-t0))

		loadings = pca.components_.T
		components =  Y.fit_transform(inData.T)
		print('loadings', loadings.shape)
		print('components', components.shape)

		if test_pca:
			msg('So far, the testing only works with one run', 'warn')
			return inData, loadings, components, tmpStd

		for i in range(nComps):
			print('cleaning comp', i, end='\r')
			for j in range(nDet):
				if i < comps_detector[j]:
					inData[j, :] -= components[:,i]*loadings[j,i]

		print('cleaned comp', i)
		#inData -= corr.T
		inData = (inData.T*tmpStd).T
		#inData = inData.T

		out_data[:, run*time_chunck:end_tchunck] = inData

	return out_data


def gaussian(x, A, mu, sigma, offset):
    """
        Gaussian function
        Parameters
        ----------
        x : int/float/array
        A : float
            Amplitude
        mu : float
            Mean
        sigma : float
            Dispersion
        offset : float
            Offset
        ----------
    """
    return offset+A*_np.exp(-((x-mu)**2)/(2*sigma**2))


def twoD_Gaussian(pos, amplitude, xo, yo, sigma, offset):
	"""
		2D Gaussian function.
		---------
		Parameters
			Input:
				pos:     	[list] Points of the 2D map
				amplitude:  [float] Amplitude
				xo, yo:     [float] Gaussian profile position
				sigma:      [float] Dispersion profile
				offset:     [float] Offset profile
			Ouput:
				g:			[list] Gaussian profile unwraped in one dimension
	"""
	x = pos[0]
	y = pos[1]
	xo = float(xo)
	yo = float(yo)
	g = offset + amplitude*np.exp(-(((x - xo)**2/2./sigma**2) + ((y - yo)**2/2./sigma**2)))

	return g.ravel()


def twoD_ElGaussian(pos, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
	"""
		2D Elliptical Gaussian function.
		---------
		Parameters
			Input:
				pos:             	[list] Points of the 2D map
				amplitude:          [float] Amplitude
				xo, yo:             [float] Gaussian profile position
				sigma_x, sigma_y:   [float] X-Y Dispersion profile
				theta:              [float] Major axis inclination
				offset:             [float] Offset profile
			Ouput:
				g:			[list] Gaussian profile unwraped in one dimension
	"""
	x = pos[0]
	y = pos[1]
	xo = float(xo)
	yo = float(yo)
	a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
	b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
	c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
	g = offset+amplitude*np.exp(-(a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))

	return g.ravel()


def guess_beam_params(data, fit_func, sig=2.5, pix_size=3, map_size=(20, 20)):
	"""
	    Guess the basic parameters from the raw data
	        kid: (string) detector the guess parameters
	"""

	# Max ID
	max_idx = _np.where(data==_np.nanmax(data))
	# Amplitude
	amp = _np.nanmax(data)

	sig = sig/pix_size

	try:
		# Offset position
		x0 = max_idx[1][0]
		y0 = max_idx[0][0]
		# Offset level
	except Exception as e:
		x0 = map_size[0]
		y0 = map_size[1]

	offset = _np.nanmedian(data)

	if fit_func == 'Gauss':
		# Sigma
		# 2 pixels by default
		sigma = sig
		return (amp, x0, y0, sigma, offset)

	elif fit_func == 'ElGauss':
		# Sigma X-Y
		sigma_x = sig
		sigma_y = sig
		# Theta
		theta = 0.

		return (amp, x0, y0, sigma_x, sigma_y, theta, offset)


def fit_beam(x, y, data, func, pix_size=3, mask=None, fa=False):


	if not mask is None:
		data_fit = data[mask]
		x_fit = x[mask]
		y_fit = y[mask]
	else:
		data_fit = data
		x_fit = x
		y_fit = y

	# Initial guess
	initial_guess = guess_beam_params(data, func, pix_size=pix_size, map_size=(len(x_fit)/2, len(y_fit)/2))

	# Fitting
	if func == 'Gauss':
		popt, pcov = optimize.curve_fit(twoD_Gaussian, (x_fit, y_fit), data_fit.ravel(), p0=initial_guess)
	elif func == 'ElGauss':
		popt, pcov = optimize.curve_fit(twoD_ElGaussian, (x_fit, y_fit), data_fit.ravel(), p0=initial_guess)

	return popt, pcov
