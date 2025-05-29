import os
import argparse

import numpy as np

from src.utils import read_X, read_X_with_Y
import scipy.signal
from statsmodels.tsa.stattools import acf
import dtw

from astropy.timeseries import LombScargle
import pywt
import scipy
import time

dataset_dir = './datasets/UCRArchive_2018'
output_dir = './tmp_sliding_individual'

def argsparser():
	parser = argparse.ArgumentParser("sliding dtw creator")
	parser.add_argument('--dataset', help='Dataset name', default='Coffee')
	parser.add_argument('--w', help='window size', type=int, default=-1)
	parser.add_argument('--j', help='j-window', type=int, default=-1)
	parser.add_argument('--d', help='divide', type=int, default=2)
	parser.add_argument('--f', help='function', default='ACF')

	return parser

def get_period_acf(X):  #AFC
	"""
	Estimate the optimal window size (w) based on periodicity analysis using the first time series (X[0]).

	Parameters:
	- X: Dataset array of shape (k, n) (k time series, length n)

	Returns:
	- w: Optimal sliding window size
	"""
	series = X[0] 
	series[np.isnan(series)] = 0  # Handle NaN values

	series_length = len(series)

	# Compute ACF (up to a maximum of 100 lags)
	acf_values = acf(series, nlags=min(series_length // 2, 100), fft=True)

	# Find the first peak (estimate periodicity)
	peaks, _ = scipy.signal.find_peaks(acf_values, distance=5)  # Minimum peak distance = 5

	if len(peaks) > 0:
		w = peaks[0]
	else:
		w = series_length/10  # Default to 10% of series length if no peak is found

	# Enforce minimum window size
	w = max(w, series_length/20) 
	return w


def get_period_fft(X):  #FFT
	"""
	Estimate periodicity of a time series using FFT.

	Parameters:
	- X: Dataset array

	Returns:
	- w: Estimated period (inverse of dominant frequency)
	"""
	series = X[0] 

	series = np.nan_to_num(series)
	series_length = len(series)

	fft_values = np.fft.rfft(series) 
	frequencies = np.fft.rfftfreq(series_length)

	peak_idx = np.argmax(np.abs(fft_values[1:])) + 1 # Skip DC component
	peak_freq = frequencies[peak_idx]

	w = 1 / peak_freq if peak_freq > 0 else series_length / 10

	return max(w, series_length / 20)


def get_period_lsp(X): #LSP
	"""
	Estimate periodicity using Lomb-Scargle Periodogram.

	Parameters:
	- X: Dataset array

	Returns:
	- w: Estimated period
	"""
	series = X[0]
	series = np.nan_to_num(series)
	series_length = len(series)

	time = np.arange(series_length)

	min_freq = 1 / (series_length / 2)
	max_freq = 1 / 2
	frequency = np.linspace(min_freq, max_freq, 1000)

	power = LombScargle(time, series).power(frequency)

	peak_freq = frequency[np.argmax(power)]
	w = 1 / peak_freq if peak_freq > 0 else series_length / 10

	return max(w, series_length / 20)


def get_period_wt(X): #WT
	"""
	Estimate periodicity using Wavelet Transform.

	Parameters:
	- X: Dataset array

	Returns:
	- w: Estimated period (scale with highest energy)
	"""
	series = X[0]
	series = np.nan_to_num(series)
	series_length = len(series)

	wavelet = 'morl'  # Morlet wavelet

	scales = np.arange(1, min(128, series_length // 2))

	coefficients, _ = pywt.cwt(series, scales, wavelet)

	energy = np.sum(np.abs(coefficients) ** 2, axis=1)

	best_scale = scales[np.argmax(energy)]
	w = best_scale

	return max(w, series_length / 20)


def get_dtw_sliding_window_stride_individual(X, Y, nw, nj, devide, function):
	"""
	Compute pairwise DTW distances using different window sizes (w) for each label.
	Also store the sliding windowed data.

	Parameters:
	- X: Dataset array of shape (k, n) (k time series, length n)
	- Y: Labels array of shape (k,) (label for each time series)
	- devide: Factor used to calculate stride (j) from w

	Returns:
	- distances_array: List of distance matrices for each time series (dtype=object)
	- windowed_data_array: List of sliding window data arrays (dtype=object)
	"""

	X = X.copy(order='C').astype(float)
	X[np.isnan(X)] = 0

	unique_labels = np.unique(Y)
	w_dict = {}

	if nw == -1:
		for label in unique_labels:
			label_indices = np.where(Y == label)[0]

			if function =='ACF':
				w_dict[label] = int(get_period_acf(X[label_indices]))
			elif function =='FFT':
				w_dict[label] = int(get_period_fft(X[label_indices]))
			elif function =='LSP':
				w_dict[label] = int(get_period_lsp(X[label_indices]))
			elif function =='WT':
				w_dict[label] = int(get_period_wt(X[label_indices]))
			else:
				w_dict[label] = int(get_optimal_window_size(X[label_indices])) # Default to another method if needed

	else:
		for label in unique_labels:
			w_dict[label] = nw

	distances_list = [] 
	windowed_data_list = [] 

	totalSubsequences = 0

	for k in range(len(X)):  
		label = Y[k]
		w = w_dict[label]
		j = max(1, w // devide) if nj == -1 else nj

		series_length = X.shape[1]
		num_windows = (series_length - w) // j + 1

		totalSubsequences += num_windows

		windows = [X[k, i:i + w] for i in range(0, series_length - w + 1, j)]

		windowed_data_list.append(np.array(windows, dtype=float)) 

		distances = np.zeros((num_windows, num_windows))

		for l in range(num_windows):
			for m in range(l + 1, num_windows): 
				r_value = min(len(windows[l])-1, len(windows[m])-1, 100)
				dtw_distance = dtw.query(windows[l], windows[m], r=r_value)['value']
				distances[l, m] = dtw_distance
				distances[m, l] = dtw_distance  

		distances_list.append(distances)

	distances_array = np.array(distances_list, dtype=object)

	windowed_data_array = np.empty(len(windowed_data_list), dtype=object)
	for i, item in enumerate(windowed_data_list):
		windowed_data_array[i] = item
	
	return distances_array, windowed_data_array


if __name__ == "__main__":
	# Get the arguments
	parser = argsparser()
	args = parser.parse_args()

	result_dir = os.path.join(output_dir, 'ucr_sliding_dtw_'+str(args.d)+'_'+args.f)
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)

	result_dir1 = os.path.join(output_dir, 'ucr_sliding_data_'+str(args.d)+'_'+args.f)
	if not os.path.exists(result_dir1):
		os.makedirs(result_dir1)

	X, Y = read_X_with_Y(dataset_dir, args.dataset)

	dtw_arr, data_arr = get_dtw_sliding_window_stride_individual(X, Y, args.w, args.j, args.d, args.f)
	

	np.save(os.path.join(result_dir, args.dataset), np.array(dtw_arr, dtype=object))
	np.save(os.path.join(result_dir1, args.dataset), np.array(data_arr, dtype=object))
