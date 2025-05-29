import os
import uuid
import math
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.utils.data

warnings.simplefilter("ignore")

class SimTSCTrainer:
	def __init__(self, device, logger):
		self.device = device
		self.logger = logger
		self.tmp_dir = 'tmp'
		if not os.path.exists(self.tmp_dir):
			os.makedirs(self.tmp_dir)

	def fit(self, model, model_sliding,model_sliding2, X, y, train_idx, distances, K, alpha, test_idx=None, report_test=False, batch_size=128, epochs=500):
		self.K = K
		self.alpha = alpha

		model.apply(init_weights)  # 모델을 새롭게 초기화
		model_sliding.apply(init_weights)
		model_sliding2.apply(init_weights)

		train_batch_size = min(batch_size//2, len(train_idx))
		other_idx = np.array([i for i in range(len(X)) if i not in train_idx])
		other_batch_size = min(batch_size - train_batch_size, len(other_idx))
		train_dataset = Dataset(train_idx)
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=1)

		if report_test:
			test_batch_size = min(batch_size//2, len(test_idx))
			other_idx_test = np.array([i for i in range(len(X)) if i not in test_idx])
			other_batch_size_test = min(batch_size - test_batch_size, len(other_idx_test))
			test_dataset = Dataset(test_idx)
			test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=1)


		self.adj = torch.from_numpy(distances.astype(np.float32))
		self.X, self.y = torch.from_numpy(X), torch.from_numpy(y)
		#self.X = torch.from_numpy(X).contiguous().to(self.device)
		#self.y = torch.from_numpy(y).contiguous().to(self.device)
		#self.adj = torch.from_numpy(distances.astype(np.float32)).contiguous().to(self.device)

		file_path = os.path.join(self.tmp_dir, str(uuid.uuid4()))

		optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=4e-3)
		optimizer_sliding = optim.Adam(model_sliding.parameters(), lr=1e-4, weight_decay=4e-3)
		optimizer_sliding2= optim.Adam(model_sliding.parameters(), lr=1e-4, weight_decay=4e-3)

		best_acc = 0.0
		
		
		avg_acc = 0.0
		eCnt = 0
		# Training
		for epoch in range(epochs):
			eCnt = eCnt + 1

			#if eCnt % 20 == 0:
			#	print(f"epochs cnt = {eCnt}")

			model.train()
			optimizer.zero_grad()
			optimizer_sliding.zero_grad()
			optimizer_sliding2.zero_grad()

			for sampled_train_idx in train_loader:

				#print(f"sampled_train_idx: {sampled_train_idx}")

				sampled_other_idx = np.random.choice(other_idx, other_batch_size, replace=False)
				idx = np.concatenate((sampled_train_idx, sampled_other_idx))

				#print(f"idx[0] = {idx[0]}")
				#print(f"X[2] = {X[2]}")
				_X, _y, _adj = self.X[idx].to(self.device), self.y[sampled_train_idx].to(self.device), self.adj[idx][:,idx]

				#_X_sliding = model_sliding2(idx)

				outputs = model(model_sliding, idx, _X, _adj, K, alpha)
				#outputs = model(model_sliding, idx, _X_sliding, _adj, K, alpha)
				loss = F.nll_loss(outputs[:len(sampled_train_idx)], _y)

				loss.backward()
				optimizer.step()
				optimizer_sliding.step()
				optimizer_sliding2.step()

				#if eCnt==0:
				#	return model

			model.eval()
			model_sliding.eval()
			model_sliding2.eval()

			acc = compute_accuracy(model, model_sliding,model_sliding2, self.X, self.y, self.adj, self.K, self.alpha, train_loader, self.device, other_idx, other_batch_size)
			avg_acc += acc

			if acc >= best_acc:
				best_acc = acc
				torch.save(model.state_dict(), file_path)
			'''
			if eCnt % 20 == 0:
				if report_test:
					test_acc = compute_accuracy(model, self.X, self.y, self.adj, self.K, self.alpha, test_loader, self.device, other_idx_test, other_batch_size_test)
					self.logger.log('--> Epoch {}: loss {:5.4f}; accuracy: {:5.4f}; best accuracy: {:5.4f}; test accuracy: {:5.4f}'.format(epoch+1, loss.item(), acc, best_acc, test_acc))
				else:
					self.logger.log('--> Epoch {}: loss {:5.4f}; accuracy: {:5.4f}; avg_accuracy: {:5.4f}; best accuracy: {:5.4f}'.format(epoch+1, loss.item(), acc, avg_acc/eCnt, best_acc))
			'''
		# Load the best model
		#model.load_state_dict(torch.load(file_path))

		#model.load_state_dict(torch.load(file_path, weights_only=False))
		model.load_state_dict(torch.load(file_path, map_location=torch.device("cpu"), weights_only=True))
		#model.to(torch.device("cpu"))  # 모델을 CPU로 이동


		model.eval()
		model_sliding.eval()
		model_sliding2.eval()

		os.remove(file_path)
		
		return model, model_sliding, model_sliding2

	
	
	def test(self, model, model_sliding,model_sliding2, test_idx, batch_size=128):
		test_batch_size = min(batch_size//2, len(test_idx))
		other_idx_test = np.array([i for i in range(len(self.X)) if i not in test_idx])
		other_batch_size_test = min(batch_size - test_batch_size, len(other_idx_test))
		test_dataset = Dataset(test_idx)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=1)
		acc = compute_accuracy(model, model_sliding,model_sliding2,self.X, self.y, self.adj, self.K, self.alpha, test_loader, self.device, other_idx_test, other_batch_size_test)
		return acc.item()

def init_weights(m):
		if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
			torch.nn.init.xavier_uniform_(m.weight)
			if m.bias is not None:
				torch.nn.init.zeros_(m.bias)

def compute_accuracy(model, model_sliding,model_sliding2, X, y, adj, K, alpha, loader, device, other_idx, other_batch_size):
	correct = 0
	total = 0

	with torch.no_grad():
		for batch_idx in loader:
			sampled_other_idx = np.random.choice(other_idx, other_batch_size, replace=False)
			idx = np.concatenate((batch_idx, sampled_other_idx))
			_X, _y, _adj = X[idx].to(device), y[idx][:len(batch_idx)].to(device), adj[idx][:,idx]
			#_X_sliding = model_sliding2(idx)
			outputs = model(model_sliding, idx,_X, _adj, K, alpha)
			#outputs = model(model_sliding, idx,_X_sliding, _adj, K, alpha)
			preds = outputs[:len(batch_idx)].max(1)[1].type_as(_y)
			_correct = preds.eq(_y).double()
			correct += _correct.sum()
			total += len(batch_idx)
	acc = correct / total
	return acc

class GCNEmbedding2(nn.Module):
	def __init__(self, new_slidng_data, sliding_adj, input_size, nb_classes, num_layers=1, n_feature_maps=64, dropout=0.5):
		super(GCNEmbedding2, self).__init__()
		self.num_layers = num_layers
		self.n_feature_maps = n_feature_maps
		self.slidng_data = new_slidng_data
		self.sliding_adj = sliding_adj

		self.block_1 = ResNetBlock(input_size, n_feature_maps)
		self.block_2 = ResNetBlock(n_feature_maps, n_feature_maps)
		self.block_3 = ResNetBlock(n_feature_maps, n_feature_maps)
		self.block_4 = ResNetBlock(n_feature_maps, n_feature_maps)

		if self.num_layers == 1:
			self.gc1 = GraphConvolution(n_feature_maps, n_feature_maps)
		elif self.num_layers == 2:
			self.gc1 = GraphConvolution(n_feature_maps, n_feature_maps)
			self.gc2 = GraphConvolution(n_feature_maps, n_feature_maps)
			self.dropout = dropout
		elif self.num_layers == 3:
			self.gc1 = GraphConvolution(n_feature_maps, n_feature_maps)
			self.gc2 = GraphConvolution(n_feature_maps, n_feature_maps)
			self.gc3 = GraphConvolution(n_feature_maps, n_feature_maps)
			self.dropout = dropout

	def forward(self, idx):
		
		num = 0
		
		device = self.gc1.bias.device
		y = torch.zeros((len(idx), self.n_feature_maps)).to(device)

		for i in np.nditer(idx):
		
			x = self.slidng_data[i].to(device)  # idx에서 인덱스 추출
			adj = self.sliding_adj[i]

			adj = adj.to(device)

			x = self.block_1(x)
			x = self.block_2(x)
			x = self.block_3(x)
			x = self.block_4(x)
			x = F.avg_pool1d(x, x.shape[-1]).squeeze()

			if self.num_layers == 1:
				x = self.gc1(x, adj)
			elif self.num_layers == 2:
				x = F.relu(self.gc1(x, adj))
				x = F.dropout(x, self.dropout, training=self.training)
				x = self.gc2(x, adj)
			elif self.num_layers == 3:
				x = F.relu(self.gc1(x, adj))
				x = F.dropout(x, self.dropout, training=self.training)
				x = F.relu(self.gc2(x, adj))
				x = F.dropout(x, self.dropout, training=self.training)
				x = self.gc3(x, adj)

			graph_embedding = x.mean(dim=0)

			y[num] = graph_embedding
			num = num+1
			#print(f"graph_embedding = {graph_embedding}")

		return y


class GCNEmbedding(nn.Module):
	def __init__(self, input_size, nb_classes, num_layers=1, n_feature_maps=64, dropout=0.5):
		super(GCNEmbedding, self).__init__()
		self.num_layers = num_layers

		self.block_1 = ResNetBlock(input_size, n_feature_maps)
		self.block_2 = ResNetBlock(n_feature_maps, n_feature_maps)
		self.block_3 = ResNetBlock(n_feature_maps, n_feature_maps)
		self.block_4 = ResNetBlock(n_feature_maps, n_feature_maps)

		if self.num_layers == 1:
			self.gc1 = GraphConvolution(n_feature_maps, n_feature_maps)
		elif self.num_layers == 2:
			self.gc1 = GraphConvolution(n_feature_maps, n_feature_maps)
			self.gc2 = GraphConvolution(n_feature_maps, n_feature_maps)
			self.dropout = dropout
		elif self.num_layers == 3:
			self.gc1 = GraphConvolution(n_feature_maps, n_feature_maps)
			self.gc2 = GraphConvolution(n_feature_maps, n_feature_maps)
			self.gc3 = GraphConvolution(n_feature_maps, n_feature_maps)
			self.dropout = dropout

	def forward(self, x, adj):

		device = self.gc1.bias.device
		adj = adj.to(device)

		x = self.block_1(x)
		#x = self.block_2(x)
		#x = self.block_3(x)
		#x = self.block_4(x)
		x = F.avg_pool1d(x, x.shape[-1]).squeeze()

		if self.num_layers == 1:
			x = self.gc1(x, adj)
		elif self.num_layers == 2:
			x = F.relu(self.gc1(x, adj))
			x = F.dropout(x, self.dropout, training=self.training)
			x = self.gc2(x, adj)
		elif self.num_layers == 3:
			x = F.relu(self.gc1(x, adj))
			x = F.dropout(x, self.dropout, training=self.training)
			x = F.relu(self.gc2(x, adj))
			x = F.dropout(x, self.dropout, training=self.training)
			x = self.gc3(x, adj)

		graph_embedding = x.mean(dim=0)

		#print(f"graph_embedding = {graph_embedding}")

		return graph_embedding


class SimTSC(nn.Module):
	def __init__(self, input_size, sliding_input_size, nb_classes, slidng_data, sliding_adj, resWeight, subWeight, num_layers=1, n_feature_maps=64, dropout=0.5):
		super(SimTSC, self).__init__()
		self.num_layers = num_layers
		self.input_size = input_size
		self.sliding_input_size = sliding_input_size
		self.slidng_data = slidng_data
		self.sliding_adj = sliding_adj
		self.n_feature_maps = n_feature_maps

		self.resWeight = resWeight
		self.subWeight = subWeight


		self.block_1 = ResNetBlock(input_size, n_feature_maps)
		self.block_2 = ResNetBlock(n_feature_maps, n_feature_maps)
		self.block_3 = ResNetBlock(n_feature_maps, n_feature_maps)
		
		self.gcn_embedding = GCNEmbedding(sliding_input_size,nb_classes, 2)

		#self.gcn_embedding = model_sliding

		if self.num_layers == 1:
			self.gc1 = GraphConvolution(n_feature_maps, nb_classes)
		elif self.num_layers == 2:
			self.gc1 = GraphConvolution(n_feature_maps, n_feature_maps)
			self.gc2 = GraphConvolution(n_feature_maps, nb_classes)
			self.dropout = dropout
		elif self.num_layers == 3:
			self.gc1 = GraphConvolution(n_feature_maps, n_feature_maps)
			self.gc2 = GraphConvolution(n_feature_maps, n_feature_maps)
			self.gc3 = GraphConvolution(n_feature_maps, nb_classes)
			self.dropout = dropout

	def forward(self, gcn_embedding, idx, x, adj, K, alpha):
		ranks = torch.argsort(adj, dim=1)
		sparse_index = [[], []]
		sparse_value = []
		for i in range(len(adj)):
			_sparse_value = []
			for j in ranks[i][:K]:
				sparse_index[0].append(i)
				sparse_index[1].append(j)
				_sparse_value.append(1/np.exp(alpha*adj[i][j]))
				#_sparse_value.append(1/np.exp(alpha * adj[i][j].cpu().numpy()))
			_sparse_value = np.array(_sparse_value)
			_sparse_value /= _sparse_value.sum()
			sparse_value.extend(_sparse_value.tolist())
		sparse_index = torch.LongTensor(sparse_index)
		sparse_value = torch.FloatTensor(sparse_value)
		#adj = torch.sparse.FloatTensor(sparse_index, sparse_value, adj.size())
		adj = torch.sparse_coo_tensor(sparse_index, sparse_value, adj.size(), dtype=torch.float32, device=adj.device)
		device = self.gc1.bias.device
		adj = adj.to(device)
		#print(f"input_size = {self.input_size}")
		#print(f"sliding_input_size = {self.sliding_input_size}")
		#print(f"type(idx) = {type(idx)}")
		#print(f"type(x) = {type(x)}")
		#print(f"len(x) = {len(x)}")
		#print(f"x.shape = {x.shape}")
		#print(f"x[0] = {x[0]}")
		
		x = self.block_1(x)
		x = self.block_2(x)
		x = self.block_3(x)
		yy = F.avg_pool1d(x, x.shape[-1]).squeeze()
		
		#print(f"x = {x}")
		#print(f"idx = {idx}")

		#self.slidng_data = torch.from_numpy(slidng_data)
		num=0
		#print(f"len(idx) = {len(idx)}")
		#
		
		y = torch.zeros((len(idx), self.n_feature_maps)).to(device)
		for i in np.nditer(idx):
		
			sx = self.slidng_data[i].to(device)  # idx에서 인덱스 추출
			sadj = self.sliding_adj[i]
			'''
			print(f"i = {i}")

			print(f"type(sx) = {type(sx)}")
			print(f"type(sadj) = {type(sadj)}")

			print(f"sx.shape = {sx.shape}")
			print(f"sadj.shape = {sadj.shape}")

			print(f"sx = {sx}")
			print(f"sadj = {sadj}")
			'''
			y[num] = gcn_embedding(sx, sadj)
			num = num+1
			#print(f"self.slidng_data.shape = {self.slidng_data.shape}")
		#y=yy
		
		#print(f"type(y) = {type(y)}")
		#print(f"type(yy) = {type(yy)}")


		y = (self.resWeight * y + self.subWeight * yy)

		#y = (y+yy) / 2
		#y = (y+yy)

		if self.num_layers == 1:
			y = self.gc1(y, adj)
		elif self.num_layers == 2:
			y = F.relu(self.gc1(y, adj))
			y = F.dropout(y, self.dropout, training=self.training)
			y = self.gc2(y, adj)
		elif self.num_layers == 3:
			y = F.relu(self.gc1(y, adj))
			y = F.dropout(y, self.dropout, training=self.training)
			y = F.relu(self.gc2(y, adj))
			y = F.dropout(y, self.dropout, training=self.training)
			y = self.gc3(y, adj)

		y = F.log_softmax(y, dim=1)

		return y

class GraphConvolution(Module):
	"""
	Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
	"""

	def __init__(self, in_features, out_features, bias=True):
		super(GraphConvolution, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.FloatTensor(in_features, out_features))
		if bias:
			self.bias = Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(0))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input, adj):
		support = torch.mm(input, self.weight)
		output = torch.spmm(adj, support)
		if self.bias is not None:
			return output + self.bias
		else:
			return output

class ResNetBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(ResNetBlock, self).__init__()
		self.expand = True if in_channels < out_channels else False

		self.conv_x = nn.Conv1d(in_channels, out_channels, 7, padding=3)
		self.bn_x = nn.BatchNorm1d(out_channels)
		self.conv_y = nn.Conv1d(out_channels, out_channels, 5, padding=2)
		self.bn_y = nn.BatchNorm1d(out_channels)
		self.conv_z = nn.Conv1d(out_channels, out_channels, 3, padding=1)
		self.bn_z = nn.BatchNorm1d(out_channels)

		if self.expand:
			self.shortcut_y = nn.Conv1d(in_channels, out_channels, 1)
		self.bn_shortcut_y = nn.BatchNorm1d(out_channels)

	def forward(self, x):
		B, _, L = x.shape
		out = F.relu(self.bn_x(self.conv_x(x)))
		out = F.relu(self.bn_y(self.conv_y(out)))
		out = self.bn_z(self.conv_z(out))

		if self.expand:
			x = self.shortcut_y(x)
		x = self.bn_shortcut_y(x)
		out += x
		out = F.relu(out)
	   
		return out

class Dataset(torch.utils.data.Dataset):
	def __init__(self, idx):
		self.idx = idx

	def __getitem__(self, index):
		return self.idx[index]

	def __len__(self):
		return len(self.idx)

