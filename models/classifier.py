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
		Extends DN4 with one extra "super-class" score. The super column pools
		local descriptors from the first `base_per_super` base classes into a
		single bank, runs k-NN against it, then subtracts the spread of
		constituent scores.

		Output shape: (Q, way_num + 1). Columns [0..way_num) are standard DN4
		base scores. Column `way_num` is the adjusted super-class score:

		    super_raw  = k-NN score over the union of constituent descriptors
		    spread     = max(constituent_scores) - min(constituent_scores)
		    super_adj  = super_raw - spread

		Why subtract the spread:
		  * The union bank is a superset of each constituent bank, so
		    super_raw >= max(constituent_score) always. Using super_raw alone
		    (original design) made super tie-or-beat the winning constituent on
		    every query, i.e. super stole everything.
		  * For a pure-constituent query (e.g. pure lion) the top-k over the
		    union is dominated by that one constituent's descriptors, so
		    super_raw ~= winning constituent score, and the OTHER constituent's
		    score is much lower -> large spread -> super_adj collapses toward
		    the loser's score and the true class wins.
		  * For a balanced super-class query (e.g. liger) both constituent
		    scores are similar -> spread ~= 0 -> super_adj ~= super_raw. The
		    union also picks up cross-constituent matches the individual banks
		    miss, so super_raw typically exceeds max(constituent_score), letting
		    super win exactly when constituents are balanced.
	'''
	def __init__(self, way_num=5, shot_num=5, neighbor_k=3, base_per_super=2, super_alpha=1.0):
		super(ImgtoSuperClass_Metric, self).__init__()
		self.way_num = way_num
		self.shot_num = shot_num
		self.neighbor_k = neighbor_k
		self.base_per_super = base_per_super
		# Scales the spread penalty: 0.0 = no penalty, 1.0 = full subtraction.
		# Lower values help when the backbone produces asymmetric constituent scores
		# for a visually-unbalanced hybrid (e.g. ligers look more tiger than lion).
		self.super_alpha = super_alpha


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
		base_scores = torch.sum(torch.sum(base_topk, 3), 2)                            # [Q, C]

		# Super-class raw: k-NN over the union of the first `base_per_super` classes'
		# descriptor banks.
		d = support.size(1)
		super_bank = support[:self.base_per_super].permute(1, 0, 2).contiguous().view(d, -1).unsqueeze(0)
		# [1, d, bps * shot * HW]
		super_sim = torch.matmul(query, super_bank)                                    # [Q, 1, HW, bps*shot*HW]
		super_topk, _ = torch.topk(super_sim, self.neighbor_k, 3)                      # [Q, 1, HW, k]
		super_raw = torch.sum(torch.sum(super_topk, 3), 2)                             # [Q, 1]

		# Penalize the super score by the gap between the strongest and weakest
		# constituent: pure-constituent queries have a large gap and drop out;
		# balanced super queries have a small gap and stay near super_raw.
		constituent_scores = base_scores[:, :self.base_per_super]                      # [Q, bps]
		spread = (constituent_scores.max(dim=1, keepdim=True).values
				  - constituent_scores.min(dim=1, keepdim=True).values)                # [Q, 1]
		super_scores = super_raw - self.super_alpha * spread                          # [Q, 1]

		return torch.cat([base_scores, super_scores], dim=1)                           # [Q, C+1]


	def forward(self, x1, x2):
		return self.cal_cosinesimilarity(x1, x2)
