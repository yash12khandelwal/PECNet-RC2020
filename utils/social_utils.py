from IPython import embed
import glob
import pandas as pd
import pickle
import os
import torch
from torch import nn
from torch.utils import data
import random
import numpy as np

def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

"""for sanity check"""
def naive_social(p1_key, p2_key, all_data_dict):
	if abs(p1_key-p2_key)<4:
		return True
	else:
		return False

def find_min_time(t1, t2):
	"""given two time frame arrays, find then min dist (time)"""
	min_d = 9e4
	t1, t2 = t1[:8], t2[:8]

	for t in t2:
		if abs(t1[0]-t)<min_d:
			min_d = abs(t1[0]-t)

	for t in t1:
		if abs(t2[0]-t)<min_d:
			min_d = abs(t2[0]-t)

	return min_d

def find_min_dist(p1x, p1y, p2x, p2y):
	"""given two time frame arrays, find then min dist"""
	min_d = 9e4
	p1x, p1y = p1x[:8], p1y[:8]
	p2x, p2y = p2x[:8], p2y[:8]

	for i in range(len(p1x)):
		for j in range(len(p1x)):
			if ((p2x[i]-p1x[j])**2 + (p2y[i]-p1y[j])**2)**0.5 < min_d:
				min_d = ((p2x[i]-p1x[j])**2 + (p2y[i]-p1y[j])**2)**0.5

	return min_d

def social_and_temporal_filter(p1_key, p2_key, all_data_dict, time_thresh=48, dist_tresh=100):
	p1_traj, p2_traj = np.array(all_data_dict[p1_key]), np.array(all_data_dict[p2_key])
	p1_time, p2_time = p1_traj[:,1], p2_traj[:,1]
	p1_x, p2_x = p1_traj[:,2], p2_traj[:,2]
	p1_y, p2_y = p1_traj[:,3], p2_traj[:,3]

	if find_min_time(p1_time, p2_time)>time_thresh:
		return False
	if find_min_dist(p1_x, p1_y, p2_x, p2_y)>dist_tresh:
		return False

	return True

def mark_similar(mask, sim_list):
	for i in range(len(sim_list)):
		for j in range(len(sim_list)):
			mask[sim_list[i]][sim_list[j]] = 1


def collect_data(set_name, dataset_type = 'image', batch_size=32, time_thresh=48, dist_tresh=100, scene=None, verbose=True, root_path="./"):

	assert set_name in ['train','val','test']

	'''Please specify the parent directory of the dataset. In our case data was stored in:
		root_path/trajnet_image/train/scene_name.txt
		root_path/trajnet_image/test/scene_name.txt
	'''

	rel_path = './ETH/{0}/'.format(set_name)

	full_dataset = []
	full_masks = []

	current_batch = []
	mask_batch = [[0 for i in range(int(1000))] for j in range(int(1000))]

	current_size = 0
	social_id = 0
	part_file = '/{}.csv'.format('*' if scene == None else scene)

	for file in glob.glob(rel_path + part_file):
		#scene_name = file[len(rel_path)+1:-6] + file[-5]
		data = np.loadtxt(fname = file, delimiter = ',')
		print(data.shape)
		data_by_id = {}
		data = np.transpose(data)
		print(data[0,:])
		for frame_id, person_id, x, y in data:
			if person_id not in data_by_id.keys():
				data_by_id[person_id] = []
			data_by_id[person_id].append([frame_id, x, y])

		all_data_dict = data_by_id.copy()
		if verbose:
			print("Total People: ", len(list(data_by_id.keys())))
		for person in data_by_id.items():
			#print(person[1])
			data = person[1][::6]
			time = len(data)
			if time<20 : 
				continue
			for i in range(time-19):
				full_dataset.append(np.array(data[i:i+20]))
		# while len(list(data_by_id.keys()))>0:
		# 	related_list = []
		# 	curr_keys = list(data_by_id.keys())

		# 	if current_size<batch_size:
		# 		pass
		# 	else:
		# 		full_dataset.append(np.array(current_batch[:32],dtype=None).copy())
		# 		mask_batch = np.array(mask_batch)
		# 		full_masks.append(mask_batch[0:32, 0:32])
		# 		current_size = 0
		# 		social_id = 0
		# 		current_batch = []
		# 		mask_batch = [[0 for i in range(int(1000))] for j in range(int(1000))]

		# 	if len(data_by_id[curr_keys[0]]) < 16 :
		# 		del data_by_id[curr_keys[0]]
		# 		continue
		# 	current_batch.append(np.array(np.array(all_data_dict[curr_keys[0]],dtype=None)[:16]))
		# 	related_list.append(current_size)
		# 	del data_by_id[curr_keys[0]]

		# 	for i in range(1, len(curr_keys)):
		# 		if social_and_temporal_filter(curr_keys[0], curr_keys[i], all_data_dict, time_thresh, dist_tresh):
		# 			current_batch.append(np.array(np.array(all_data_dict[curr_keys[i]],dtype=None)[:16]))
		# 			related_list.append(current_size)
		# 			current_size+=1
		# 			del data_by_id[curr_keys[i]]

		# 	mark_similar(mask_batch, related_list)
		# 	social_id +=1


	#full_dataset.append(current_batch.copy())
	#mask_batch = np.array(mask_batch)
	#full_masks.append(mask_batch[0:len(current_batch),0:len(current_batch)])
	return np.array(full_dataset,dtype=None)

def generate_pooled_data(b_size, t_tresh, d_tresh, train=True, scene=None, verbose=True, dataset_type = "eth"):
	if train:
		full_train= collect_data("train", batch_size=32, time_thresh=t_tresh, dist_tresh=d_tresh, scene=scene, verbose=verbose)
		print(full_train.shape)
		train_name = "../social_pool_data/train_{0}_{1}_{2}_{3}_{4}.pickle".format('all' if scene is None else scene[:-2] + scene[-1], b_size, t_tresh, d_tresh, dataset_type)
		with open(train_name, 'wb') as f:
			pickle.dump(full_train, f)

	if not train:
		full_test= collect_data("test", batch_size=32, time_thresh=t_tresh, dist_tresh=d_tresh, scene=scene, verbose=verbose)
		print(full_test.shape)
		test_name = "../social_pool_data/test_{0}_{1}_{2}_{3}_{4}.pickle".format('all' if scene is None else scene[:-2] + scene[-1], b_size, t_tresh, d_tresh, dataset_type)
		with open(test_name, 'wb') as f:
			pickle.dump(full_test, f)

def initial_pos(traj_batches):
	batches = []
	for b in traj_batches:
		starting_pos = b[:,7,:].copy()/1000 #starting pos is end of past, start of future. scaled down.
		batches.append(starting_pos)

	return batches

def calculate_loss(criterion, x: torch.Tensor, reconstructed_x: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor, future: torch.Tensor, interpolated_future: torch.Tensor):
	batch_loss_dict = {}

	# Averate Endpoint Loss
	batch_loss_dict["ael"] = criterion(x, reconstructed_x)
	# Average Trajectory Loss
	batch_loss_dict["atl"] = criterion(future, interpolated_future)
	# KL Divergence Loss
	batch_loss_dict["kld"] = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

	return batch_loss_dict

class SocialDataset(data.Dataset):

	def __init__(self, set_name="train", b_size=4096, t_tresh=60, d_tresh=50, scene=None, id=False, verbose=True):
		"Initialization"
		load_name = f"../social_pool_data/{set_name}_{'all_' if scene is None else scene[:-2] + scene[-1] + '_'}{b_size}_{t_tresh}_{d_tresh}_drone.pickle"
		print(load_name)
		with open(load_name, "rb") as f:
			data = pickle.load(f)

		traj, masks = data
		traj_new = []

		if id==False:
			for t in traj:
				t = np.array(t)
				t = t[:,:,2:]
				traj_new.append(t)
				if set_name=="train":
					#augment training set with reversed tracklets...
					reverse_t = np.flip(t, axis=1).copy()
					traj_new.append(reverse_t)
		else:
			for t in traj:
				t = np.array(t)
				traj_new.append(t)

				if set_name=="train":
					#augment training set with reversed tracklets...
					reverse_t = np.flip(t, axis=1).copy()
					traj_new.append(reverse_t)


		masks_new = []
		for m in masks:
			masks_new.append(m)

			if set_name=="train":
				#add second time for the reversed tracklets...
				masks_new.append(m)

		traj_new = np.array(traj_new)
		masks_new = np.array(masks_new)
		self.trajectory_batches = traj_new.copy()
		self.mask_batches = masks_new.copy()
		self.initial_pos_batches = np.array(initial_pos(self.trajectory_batches)) #for relative positioning
		if verbose:
			print("Initialized social dataloader...")


class ETHDataset(data.Dataset):
	def __init__(self, dataset = "eth", set_name="train", b_size=4096, t_tresh=60, d_tresh=50, scene=None, id=False, verbose=True):
		load_name = "../social_pool_data/{0}_{1}{2}_{3}_{4}_{5}.pickle".format(set_name, 'all_' if scene is None else scene[:-2] + scene[-1] + '_', b_size, t_tresh, d_tresh, dataset)
		print(load_name)
		with open(load_name, 'rb') as f:
			self.data = np.array(pickle.load(f))
		print(self.data.shape)

	def __len__(self):
		return len(self.data)	

	def __getitem__(self, idx):
		return self.data[idx]
"""
We"ve provided pickle files, but to generate new files for different datasets or thresholds, please use a command like so:
Parameter1: batchsize, Parameter2: time_thresh, Param3: dist_thresh
"""

#generate_pooled_data(512, 0, 100, train=True, verbose=True, dataset_type="zara2")
