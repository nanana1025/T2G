import os
import argparse

import numpy as np

from src.utils import read_X,read_X_with_Y
import scipy.signal
from statsmodels.tsa.stattools import acf

import lc

dataset_dir = './datasets/UCRArchive_2018'
output_dir = './tmp_LC'

def argsparser():
	parser = argparse.ArgumentParser("Subsequence LC creator")
	parser.add_argument('--dataset', help='Dataset name', default='Coffee')
	parser.add_argument('--w', help='window size', type=int, default=-1)

	return parser

def compute_E(LC_ab, LC_ba):
	"""
	Compute the E(A,B) value based on lagged correlations LC(A,B) and LC(B,A).

	Parameters:
	LC_ab : float
		Lagged correlation from A to B.
	LC_ba : float
		Lagged correlation from B to A.

	Returns:
	float
		Computed E(A,B) value.
	"""
	s = LC_ab + LC_ba
	if s == 0:
		return 0.0  # 둘 다 0이면 E(A,B) = 0
	else:
		direction_term = LC_ab / s
		strength_term = np.log1p(s)  # log(1 + s)
		return direction_term * strength_term

def ACF(X):  #AFC
	"""
	Estimate the optimal window size (w) by analyzing the periodicity of the first time series in the dataset.

	Parameters:
	X : ndarray of shape (k, n)
		Time-series dataset containing k series of length n.

	Returns:
	float
		Estimated optimal window size for subsequence generation.
	"""
	series = X[0]  # Use the first time series for periodicity analysis
	series[np.isnan(series)] = 0  # Replace NaN with 0

	series_length = len(series)

	# Compute autocorrelation (ACF) up to 100 lags
	acf_values = acf(series, nlags=min(series_length // 2, 100), fft=True)

	# Find the first peak in ACF to determine periodicity
	peaks, _ = scipy.signal.find_peaks(acf_values, distance=5)  # Minimum distance between peaks = 5

	if len(peaks) > 0:
		w = peaks[0]  
	else:
		w = series_length/10  # Default to 10% of the series length if no peak is found

	# Enforce a minimum window size
	w = max(w, series_length/20)
	return w

def split_into_windows(series, window_size, stride=None):
	"""
	Split a time series into overlapping subsequences.

	Parameters:
	series : ndarray
		Input time series (1D array).
	window_size : int
		Length of each window.
	stride : int, optional
		Step size for window sliding. Defaults to 1.

	Returns:
	list of ndarrays
		List of subsequences generated from the time series.
	"""
	if stride is None:
		stride = 1
	if stride == 0:
		stride = 1
	subseqs = []
	for start in range(0, len(series) - window_size + 1, stride):
		subseqs.append(series[start:start + window_size])
	return subseqs

def get_LC(X, Y, nw):
	"""
	Compute the directional lagged correlation matrix (LC_V) based on subsequence windows.

	Parameters:
	X : ndarray of shape (k, n)
		Time-series dataset containing k series of length n.
	Y : ndarray of shape (k,)
		Labels corresponding to each time series.
	nw : int
		Fixed window size. If -1, window size is determined adaptively for each class.

	Returns:
	LC_V : ndarray of shape (k, k)
		Directional lagged correlation matrix.
	"""
	# Compute window sizes for each label
	XF = X.copy(order='C').astype(float)
	XF[np.isnan(XF)] = 0

	unique_labels = np.unique(Y)
	w_dict = {}

	if nw == -1:
		for label in unique_labels:
			label_indices = np.where(Y == label)[0]
			w_dict[label] = int(ACF(XF[label_indices])) # Determine optimal window size for this class
	else:
		for label in unique_labels:
			w_dict[label] = nw
	
	 # Compute pairwise lagged correlation values (LC)
	X = X.copy(order='C').astype(np.float64)
	X[np.isnan(X)] = 0
	LC = np.zeros((X.shape[0], X.shape[0]), dtype=np.float64)

	for i in range(len(X)):
		for j in range(len(X)):
			if i == j:
				LC[i][j] = 0.0
			else:
				data = X[i]
				query = X[j]

				label_data = Y[i]
				label_query = Y[j]

				w_data = w_dict[label_data]
				w_query = w_dict[label_query]

				window = min(w_data, w_query)
				stride = int(window/2)
				# Split into subsequences
				data_windows = split_into_windows(data, window, stride)
				query_windows = split_into_windows(query, window, stride)

				# Compute LC values for all subsequence pairs
				lc_values = []
				for sub_data in data_windows:
					for sub_query in query_windows:
						if len(sub_data) >= 3 and len(sub_query) >= 3:
							val = lc.compute_lc(sub_data, sub_query)
							lc_values.append(val)

				LC[i][j] = np.sum(lc_values) if lc_values else 0.0

	 # Compute directional weight matrix LC_V
	LC_V = np.zeros((X.shape[0], X.shape[0]), dtype=np.float64)

	for i in range(len(X)):
		for j in range(len(X)):
			if i!=j:
				LC_V[i][j] = compute_E(LC[i][j], LC[j][i])
		
	return LC_V

if __name__ == "__main__":
	# Get the arguments
	parser = argsparser()
	args = parser.parse_args()

	result_dir = os.path.join(output_dir, 'ucr_datasets_lc_Sub')

	if not os.path.exists(result_dir):
		os.makedirs(result_dir)


	X, Y = read_X_with_Y(dataset_dir, args.dataset)

	LC_V = get_LC(X, Y, args.w)
	np.save(os.path.join(result_dir, args.dataset), LC_V)
