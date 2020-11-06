import sys
sys.path.append("../")
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import copy
import matplotlib.pyplot as plt
import numpy as np
import yaml
from torch.utils.data import DataLoader

from utils.models import PECNet
from utils.social_utils import SocialDataset
from utils.test_engine import test_engine

parser = argparse.ArgumentParser(description='PECNet')

parser.add_argument('--num_workers', '-nw', type=int, default=0)
parser.add_argument('--gpu_index', '-gi', type=int, default=0)
parser.add_argument('--load_file', '-lf', default="run7.pt")
parser.add_argument('--num_trajectories', '-nt', default=20) #number of trajectories to sample
parser.add_argument('--verbose', '-v', action='store_true')
parser.add_argument('--root_path', '-rp', default="./")

args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
	torch.cuda.set_device(args.gpu_index)
print(device)


checkpoint = torch.load('../saved_models/{}'.format(args.load_file), map_location=device)
hyper_params = checkpoint["hyper_params"]

print(hyper_params)

def main():
	N = args.num_trajectories #number of generated trajectories
	model = PECNet(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"], hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params['non_local_theta_size'], hyper_params['non_local_phi_size'], hyper_params['non_local_g_size'], hyper_params["fdim"], hyper_params["zdim"], hyper_params["nonlocal_pools"], hyper_params['non_local_dim'], hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], args.verbose)
	model = model.double().to(device)
	model.load_state_dict(checkpoint["model_state_dict"])
	test_dataset = SocialDataset(set_name="test", b_size=hyper_params["test_b_size"], t_tresh=hyper_params["time_thresh"], d_tresh=hyper_params["dist_thresh"], verbose=args.verbose)

	for traj in test_dataset.trajectory_batches:
		traj -= traj[:, :1, :]
		traj *= hyper_params["data_scale"]

	#average ade/fde for k=20 (to account for variance in sampling)
	num_samples = 150
	average_ade, average_fde = 0, 0
	for i in range(num_samples):
		test_loss, final_point_loss_best, final_point_loss_avg = test_engine(test_dataset, model, device, hyper_params, best_of_n = N)
		average_ade += test_loss
		average_fde += final_point_loss_best

	print()
	print("Average ADE:", average_ade/num_samples)
	print("Average FDE:", average_fde/num_samples)

main()
