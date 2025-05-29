import os
import argparse
import gc

import numpy as np
import torch
import time

from src.utils import read_dataset_from_npy, Logger
from src.simtsc.model3 import SimTSC, SimTSCTrainer, GCNEmbedding, GCNEmbedding2

data_dir = './tmp'
sliding_data_dir = './tmp_sliding_individual'
log_dir = './logs'

def train(X, y, train_idx, test_idx, distances, Oslidng_data, slidng_distances, device, logger, K, alpha, batch_size, epoches, resWeight, subWeight):
	nb_classes = len(np.unique(y, axis=0))

	normalized_data_list = []

	for arr in Oslidng_data:  
		
		arr_list = list(arr)  
		arr = np.array(arr_list, dtype=np.float32) 
		
		mean = np.mean(arr, axis=1, keepdims=True)  
		std = np.std(arr, axis=1, keepdims=True)  
		arr = (arr - mean) / (std + 1e-8)  

		arr = np.expand_dims(arr, axis=2).copy()
		tensor_arr = torch.tensor(arr, dtype=torch.float32).clone()
		normalized_data_list.append(arr)

	input_size = X.shape[1]
	
	adj_list = []

	for arr in slidng_distances:
		
		arr_list = list(arr) 
		tensor_arr = torch.tensor(np.array(arr_list, dtype=np.float32)).clone()
		adj_list.append(tensor_arr)

	sliding_adj = calculSlidingGraph3D(adj_list, K, alpha)

	model = SimTSC(input_size, nb_classes, normalized_data_list, sliding_adj, resWeight, subWeight)
	model = model.to(device)

	model_sliding = GCNEmbedding(nb_classes,1)
	model_sliding = model_sliding.to(device)

	model_sliding2 = GCNEmbedding2(normalized_data_list, sliding_adj, nb_classes,2)
	model_sliding2 = model_sliding2.to(device)
	
	trainer = SimTSCTrainer(device, logger)

	model, model_sliding, model_sliding2  = trainer.fit(model, model_sliding, model_sliding2, X, y, train_idx, distances, K, alpha, None, False, batch_size, epoches)
	acc = trainer.test(model, model_sliding, model_sliding2, test_idx, batch_size)
	
	return acc


def calculSlidingGraph3D(adj_list, K, alpha):
	all_sparse_graphs = []  

	for adj_np in adj_list:  
		adj = torch.tensor(adj_np, dtype=torch.float32)  
		W = adj.shape[0] 

		ranks = torch.argsort(adj, dim=1)  
		sparse_index = [[], []]
		sparse_value = []

		for i in range(W):
			_sparse_value = []
			K_actual = min(K, W) 

			for j in ranks[i][:K_actual]: 
				sparse_index[0].append(i)
				sparse_index[1].append(j)  
				_sparse_value.append(1 / np.exp(alpha * adj[i, j].item())) 

			_sparse_value = np.array(_sparse_value)
			_sparse_value /= _sparse_value.sum() 
			sparse_value.extend(_sparse_value.tolist())

		sparse_index = torch.LongTensor(sparse_index)
		sparse_value = torch.FloatTensor(sparse_value)

		sparse_adj = torch.sparse_coo_tensor(sparse_index, sparse_value, (W, W), dtype=torch.float32, device=adj.device)

		all_sparse_graphs.append(sparse_adj)

	return all_sparse_graphs 




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
	parser.add_argument('--res', help='resnet weight', type=float, default=0.4)
	parser.add_argument('--sub', help='sub weight', type=float, default=0.6)
	parser.add_argument('--d', help='divide', type=int, default=2)
	parser.add_argument('--f', help='function', default='ACF')
	
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


	slidng_dtw = os.path.join(sliding_data_dir, 'ucr_sliding_dtw_'+str(args.d)+'_'+args.f) 
	slidng_distances = np.load(os.path.join(slidng_dtw, args.dataset+'.npy'), allow_pickle=True)

	slidng_value = os.path.join(sliding_data_dir, 'ucr_sliding_data_'+str(args.d)+'_'+args.f) 
	slidng_data_list = np.load(os.path.join(slidng_value, args.dataset+'.npy'), allow_pickle=True)

	out_dir = os.path.join(log_dir, 'simtsc_log_'+str(args.shot)+'_shot'+str(args.K)+'_'+str(args.alpha))
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	out_path = os.path.join(log_dir, args.dataset+'_'+str(args.seed)+'.txt')

	with open(out_path, 'w') as f:
		logger = Logger(f)
		# Read data
		X, y, train_idx, test_idx = read_dataset_from_npy(os.path.join(data_dir, 'ucr_datasets_'+str(args.shot)+'_shot', args.dataset+'.npy'))

		# Train the model
		acc = train(X, y, train_idx, test_idx, distances, slidng_data_list, slidng_distances, device, logger, args.K, args.alpha, args.batch, args.epoch,args.res, args.sub)
		
		logger.log('>>>>> {} Test Accuracy: {:5.6f}'.format(args.dataset, acc))
		gc.collect()
		#logger.log(str(acc))
