import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
import utils
import pdb
from abc import abstractmethod
from torch.nn.utils.weight_norm import WeightNorm



# =========================== Few-shot learning method: ProtoNet =========================== #
class Prototype_Metric(nn.Module):
	'''
		The classifier module of ProtoNet by using the mean prototype and Euclidean distance,
		which is also Non-parametric.
		"Prototypical networks for few-shot learning. NeurIPS 2017."
	'''
	def __init__(self, way_num=5, shot_num=5, neighbor_k=3):
		super(Prototype_Metric, self).__init__()
		self.way_num = way_num
		self.avgpool = nn.AdaptiveAvgPool2d(1)


	# Calculate the Euclidean distance between the query and the mean prototype of the support class.
	def cal_EuclideanDis(self, input1, input2):
		'''
		 input1 (query images): 75 * 64 * 5 * 5
		 input2 (support set):  25 * 64 * 5 * 5
		'''
	
		# input1---query images
		# query = input1.view(input1.size(0), -1)                                    # 75 * 1600     (Conv64F)
		query = self.avgpool(input1).squeeze(3).squeeze(2)                           # 75 * 64
		query = query.unsqueeze(1)                                                   # 75 * 1 * 1600 (Conv64F)
   

		# input2--support set
		input2 = self.avgpool(input2).squeeze(3).squeeze(2)                          # 25 * 64
		# input2 = input2.view(input2.size(0), -1)                                   # 25 * 1600     
		support_set = input2.contiguous().view(self.way_num, -1, input2.size(1))     # 5 * 5 * 1600    
		support_set = torch.mean(support_set, 1)                                     # 5 * 1600


		# Euclidean distances between a query set and a support set
		proto_dis = -torch.pow(query-support_set, 2).sum(2)                          # 75 * 5 
		

		return proto_dis


	def forward(self, x1, x2):

		proto_dis = self.cal_EuclideanDis(x1, x2)

		return proto_dis



# =========================== Few-shot learning method: DN4 =========================== #
class ImgtoClass_Metric(nn.Module):
	'''
		Image-to-class classifier module for DN4, which is a Non-parametric classifier.
		"Revisiting local descriptor based image-to-class measure for few-shot learning. CVPR 2019."
	'''
	def __init__(self, way_num=5, shot_num=5, neighbor_k=3):
		super(ImgtoClass_Metric, self).__init__()
		self.neighbor_k = neighbor_k
		self.shot_num = shot_num


	# Calculate the Image-to-class similarity between the query and support class via k-NN.
	def cal_cosinesimilarity(self, input1, input2):
		'''
		 input1 (query images):  75 * 64 * 21 * 21
		 input2 (support set):   25 * 64 * 21 * 21
		'''

		# input1---query images
		input1 = input1.contiguous().view(input1.size(0), input1.size(1), -1)         # 75 * 64 * 441 (Conv64F_Local)
		input1 = input1.permute(0, 2, 1)                                              # 75 * 441 * 64 (Conv64F_Local)

		
		# input2--support set
		input2 = input2.contiguous().view(input2.size(0), input2.size(1), -1)         # 25 * 64 * 441
		input2 = input2.permute(0, 2, 1)                                              # 25 * 441 * 64


		# L2 Normalization
		input1_norm = torch.norm(input1, 2, 2, True)                                  # 75 * 441 * 1
		query = input1/input1_norm                                                    # 75 * 441 * 64
		query = query.unsqueeze(1)                                                    # 75 * 1 * 441 *64


		input2_norm = torch.norm(input2, 2, 2, True)                                  # 25 * 441 * 1 
		support_set = input2/input2_norm                                              # 25 * 441 * 64
		support_set = support_set.contiguous().view(-1,
				self.shot_num*support_set.size(1), support_set.size(2))               # 5 * 2205 * 64    
		support_set = support_set.permute(0, 2, 1)                                    # 5 * 64 * 2205     


		# cosine similarity between a query set and a support set
		innerproduct_matrix = torch.matmul(query, support_set)                        # 75 * 5 * 441 * 2205


		# choose the top-k nearest neighbors
		topk_value, topk_index = torch.topk(innerproduct_matrix, self.neighbor_k, 3)  # 75 * 5 * 441 * 3
		img2class_sim = torch.sum(torch.sum(topk_value, 3), 2)                        # 75 * 5 


		return img2class_sim


	def forward(self, x1, x2):

		img2class_sim = self.cal_cosinesimilarity(x1, x2)

		return img2class_sim



# =========================== Zero-shot Super-Class on top of DN4 =========================== #
class ImgtoSuperClass_Metric(nn.Module):
	'''
		Extends DN4 with one extra "super-class" score formed by pooling the local
		descriptors of the first `base_per_super` base classes of the episode. No
		training or supervision is added; the super-class score is computed from the
		exact same supports DN4 already uses.

		Output shape: (Q, way_num + 1). Columns [0..way_num) are the standard DN4
		base-class scores. Column `way_num` is the super-class score.

		Geometric rationale: the super-class descriptor bank is the union of its
		constituents' banks, so a query from a constituent class gets top-k matches
		that are a superset of that class's top-k — super_score >= constituent_score.
		A query from a non-constituent class gets top-k inside an unrelated bank, so
		the true class should still win.
	'''
	def __init__(self, way_num=5, shot_num=5, neighbor_k=3, base_per_super=2):
		super(ImgtoSuperClass_Metric, self).__init__()
		self.way_num = way_num
		self.shot_num = shot_num
		self.neighbor_k = neighbor_k
		self.base_per_super = base_per_super


	def cal_cosinesimilarity(self, input1, input2):
		# Queries: [Q, d, H, W] -> [Q, HW, d] normalized
		query = input1.contiguous().view(input1.size(0), input1.size(1), -1).permute(0, 2, 1)
		query = query / torch.norm(query, 2, 2, True)
		query = query.unsqueeze(1)                                                     # [Q, 1, HW, d]

		# Supports: [S, d, H, W] -> normalize -> group by class -> [C, d, shot*HW]
		support = input2.contiguous().view(input2.size(0), input2.size(1), -1).permute(0, 2, 1)
		support = support / torch.norm(support, 2, 2, True)
		support = support.contiguous().view(-1, self.shot_num * support.size(1), support.size(2))
		support = support.permute(0, 2, 1)                                             # [C, d, shot*HW]

		# Standard DN4 base-class scores
		base_sim = torch.matmul(query, support)                                        # [Q, C, HW, shot*HW]
		base_topk, _ = torch.topk(base_sim, self.neighbor_k, 3)                        # [Q, C, HW, k]

		# Super-class: pool the first `base_per_super` classes' descriptors into one bank
		d = support.size(1)
		super_bank = support[:self.base_per_super].permute(1, 0, 2).contiguous().view(d, -1).unsqueeze(0)
		# [1, d, base_per_super * shot * HW]
		super_sim = torch.matmul(query, super_bank)                                    # [Q, 1, HW, bps*shot*HW]
		super_topk, _ = torch.topk(super_sim, self.neighbor_k, 3)                      # [Q, 1, HW, k]

		all_topk = torch.cat([base_topk, super_topk], dim=1)                           # [Q, C+1, HW, k]
		scores = torch.sum(torch.sum(all_topk, 3), 2)                                  # [Q, C+1]
		return scores


	def forward(self, x1, x2):
		return self.cal_cosinesimilarity(x1, x2)
