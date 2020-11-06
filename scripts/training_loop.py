import sys
sys.path.append("../")
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import yaml
import numpy as np
from torch.utils.data import DataLoader

from utils.models import PECNet
from utils.social_utils import SocialDataset
from utils.train_engine import train_engine
from utils.test_engine import test_engine

parser = argparse.ArgumentParser(description='PECNet')

parser.add_argument('--num_workers', '-nw', type=int, default=0)
parser.add_argument('--gpu_index', '-gi', type=int, default=0)
parser.add_argument('--config_filename', '-cfn', type=str, default='optimal.yaml')
parser.add_argument('--save_file', '-sf', type=str, default='PECNET_social_model.pt')
parser.add_argument('--verbose', '-v', action='store_true')

args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
	torch.cuda.set_device(args.gpu_index)
print(device)

with open("../config/" + args.config_filename, 'r') as file:
	try:
		hyper_params = yaml.load(file, Loader = yaml.FullLoader)
	except:
		hyper_params = yaml.load(file)
file.close()
print(hyper_params)

model = PECNet(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"], hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params['non_local_theta_size'], hyper_params['non_local_phi_size'], hyper_params['non_local_g_size'], hyper_params["fdim"], hyper_params["zdim"], hyper_params["nonlocal_pools"], hyper_params['non_local_dim'], hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], args.verbose)
model = model.double().to(device)
optimizer = optim.Adam(model.parameters(), lr=  hyper_params["learning_rate"])

train_dataset = SocialDataset(set_name="train", b_size=hyper_params["train_b_size"], t_tresh=hyper_params["time_thresh"], d_tresh=hyper_params["dist_thresh"], verbose=args.verbose)
test_dataset = SocialDataset(set_name="test", b_size=hyper_params["test_b_size"], t_tresh=hyper_params["time_thresh"], d_tresh=hyper_params["dist_thresh"], verbose=args.verbose)

# shift origin and scale data
for traj in train_dataset.trajectory_batches:
	traj -= traj[:, :1, :]
	traj *= hyper_params["data_scale"]
for traj in test_dataset.trajectory_batches:
	traj -= traj[:, :1, :]
	traj *= hyper_params["data_scale"]


best_test_loss = 50 # start saving after this threshold
best_endpoint_loss = 50
N = hyper_params["n_values"]

for e in range(hyper_params['num_epochs']):
	train_loss, rcl, kld, adl = train_engine(train_dataset, model, device, hyper_params, optimizer)
	test_loss, final_point_loss_best, final_point_loss_avg = test_engine(test_dataset, model, device, hyper_params, best_of_n = N)


	if best_test_loss > test_loss:
		print("Epoch: ", e)
		print('################## BEST PERFORMANCE {:0.2f} ########'.format(test_loss))
		best_test_loss = test_loss
		if best_test_loss < 10.25:
			save_path = '../saved_models/' + args.save_file
			torch.save({
						'hyper_params': hyper_params,
						'model_state_dict': model.state_dict(),
						'optimizer_state_dict': optimizer.state_dict()
						}, save_path)
			print("Saved model to:\n{}".format(save_path))

	if final_point_loss_best < best_endpoint_loss:
		best_endpoint_loss = final_point_loss_best

	print("Train Loss", train_loss)
	print("RCL", rcl)
	print("KLD", kld)
	print("ADL", adl)
	print("Test ADE", test_loss)
	print("Test Average FDE (Across  all samples)", final_point_loss_avg)
	print("Test Min FDE", final_point_loss_best)
	print("Test Best ADE Loss So Far (N = {})".format(N), best_test_loss)
	print("Test Best Min FDE (N = {})".format(N), best_endpoint_loss)
