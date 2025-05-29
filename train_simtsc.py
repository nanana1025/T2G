import os
import argparse
import gc

import numpy as np
import torch
import time

from src.utils import read_dataset_from_npy, Logger
from src.simtsc.model import SimTSC, SimTSCTrainer

data_dir = './tmp'
log_dir = './logs'


def train(X, y, train_idx, test_idx, distances, device, logger, K, alpha, batch_size, epoches):
	nb_classes = len(np.unique(y, axis=0))

	input_size = X.shape[1]

	start = time.time()

	model = SimTSC(input_size, nb_classes)
	model = model.to(device)
	trainer = SimTSCTrainer(device, logger)

	model = trainer.fit(model, X, y, train_idx, distances, K, alpha, None, False, batch_size, epoches)

	acc = trainer.test(model, test_idx,batch_size)

	return acc


def argsparser():
	parser = argparse.ArgumentParser("SimTSC")
	parser.add_argument('--dataset', help='Dataset name', default='Coffee')
	parser.add_argument('--seed', help='Random seed', type=int, default=0)
	parser.add_argument('--gpu', type=str, default='0')
	parser.add_argument('--shot', help='shot', type=int, default=12)
	parser.add_argument('--K', help='K', type=int, default=3)
	parser.add_argument('--alpha', help='alpha', type=float, default=0.3)
	parser.add_argument('--batch', help='batch_size', type=int, default=128)
	parser.add_argument('--epoch', help='epochs', type=int, default=500)

	return parser

if __name__ == "__main__":
	# Get the arguments
	parser = argsparser()
	args = parser.parse_args()

	# Setup the gpu
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	if torch.cuda.is_available():
		device = torch.device("cuda:0")
	else:
		device = torch.device("cpu")

	# Seeding
	np.random.seed(args.seed)
	torch.manual_seed(int(time.time()))

	dtw_dir = os.path.join(data_dir, 'ucr_datasets_dtw') 
	distances = np.load(os.path.join(dtw_dir, args.dataset+'.npy'))

	out_dir = os.path.join(log_dir, 'simtsc_log_'+str(args.shot)+'_shot'+str(args.K)+'_'+str(args.alpha))
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	out_path = os.path.join(log_dir, args.dataset+'_'+str(args.seed)+'.txt')

	with open(out_path, 'w') as f:
		logger = Logger(f)
		# Read data
		X, y, train_idx, test_idx = read_dataset_from_npy(os.path.join(data_dir, 'ucr_datasets_'+str(args.shot)+'_shot', args.dataset+'.npy'))


		# Train the model
		acc = train(X, y, train_idx, test_idx, distances, device, logger, args.K, args.alpha,args.batch, args.epoch)

		logger.log('--> {} Test Accuracy: {:5.4f}'.format(args.dataset, acc))
		gc.collect()
		#logger.log(str(acc))
