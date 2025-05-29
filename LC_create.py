import os
import argparse

import numpy as np

from src.utils import read_X

import lc

dataset_dir = './datasets/UCRArchive_2018'
output_dir = './tmp_LC'

def argsparser():
	parser = argparse.ArgumentParser("LC creator")
	parser.add_argument('--dataset', help='Dataset name', default='Coffee')

	return parser

def compute_E(LC_ab, LC_ba):
	"""
	Compute the E(A,B) value based on LC(A,B) and LC(B,A).

	Parameters:
	LC_ab : float
		Lagged correlation value from A to B.
	LC_ba : float
		Lagged correlation value from B to A.

	Returns:
	float
		Computed E(A,B) value.
	"""
	s = LC_ab + LC_ba
	if s == 0:
		return 0.0  # Return 0 when both LC values are zero
	else:
		direction_term = LC_ab / s
		strength_term = np.log1p(s)  # log(1 + s)
		return direction_term * strength_term

def get_LC(X):
	"""
	Compute the directional weight matrix LC_V based on lagged correlation.

	Parameters:
	X : ndarray of shape (n_series, n_timestamps)
		Input time-series data matrix.

	Returns:
	ndarray of shape (n_series, n_series)
		Directional weight matrix LC_V.
	"""
	X = X.copy(order='C').astype(np.float64)
	X[np.isnan(X)] = 0
	LC = np.zeros((X.shape[0], X.shape[0]), dtype=np.float64)
	LC_V = np.zeros((X.shape[0], X.shape[0]), dtype=np.float64)


	for i in range(len(X)):
		for j in range(len(X)):
			if i==j:
				LC[i][j] = 0.0
			else:
				data = X[i]
				query = X[j]
				LC[i][j] = lc.compute_lc(data, query)
	
	for i in range(len(X)):
		for j in range(len(X)):
			if i!=j:
				LC_V[i][j] = compute_E(LC[i][j], LC[j][i])
			
	return LC_V

if __name__ == "__main__":
	# Get the arguments
	parser = argsparser()
	args = parser.parse_args()

	result_dir = os.path.join(output_dir, 'ucr_datasets_lc')

	if not os.path.exists(result_dir):
		os.makedirs(result_dir)

	X = read_X(dataset_dir, args.dataset)

	LC = get_LC(X)
	np.save(os.path.join(result_dir, args.dataset), LC)
