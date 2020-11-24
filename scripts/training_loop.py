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
from utils.social_utils import SocialDataset, set_seed
from utils.train_engine import train_engine
from utils.test_engine import test_engine
from visualization.wandb_utils import init_wandb, log_losses, save_model_wandb, log_metrics, log_summary

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="PECNet")
	parser.add_argument("--num_workers", "-nw", type=int, default=0)
	parser.add_argument("--gpu_index", "-gi", type=int, default=0)
	parser.add_argument("--config_filename", "-cfn", type=str, default="optimal.yaml")
	parser.add_argument("--version", "-v", type=str, default="PECNET_social_model")
	parser.add_argument("--verbose", action="store_true")
	parser.add_argument("-s", "--seed", default=42, help="Random seed")
	parser.add_argument("-w", "--wandb", action="store_true", help="Log to wandb or not")
	args = parser.parse_args()

	# setting seed system wide for proper reproducibility
	set_seed(args.seed)
	
	torch.set_default_dtype(torch.float64)
	
	# set device
	device = torch.device("cuda", index=args.gpu_index) if torch.cuda.is_available() else torch.device("cpu")
	if torch.cuda.is_available():
		torch.cuda.set_device(args.gpu_index)
	print(f"Using {device} device")

	# load hyperparams from config file and update the dict
	with open("../config/" + args.config_filename, "r") as file:
		try:
			hyperparams = yaml.load(file, Loader = yaml.FullLoader)
		except:
			hyperparams = yaml.load(file)
	file.close()
	print(hyperparams)

	# initialize model
	model = PECNet(hyperparams["enc_past_size"], hyperparams["enc_dest_size"], hyperparams["enc_latent_size"], hyperparams["dec_size"], hyperparams["predictor_hidden_size"], hyperparams["non_local_theta_size"], hyperparams["non_local_phi_size"], hyperparams["non_local_g_size"], hyperparams["fdim"], hyperparams["zdim"], hyperparams["nonlocal_pools"], hyperparams["non_local_dim"], hyperparams["sigma"], hyperparams["past_length"], hyperparams["future_length"], args.verbose)
	model = model.double().to(device)

	# initailize wandb, save the gradients and model information to wandb
	if args.wandb:
		init_wandb(hyperparams.copy(), model, args)

	# initialize optimizer
	optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

	# initialize dataloaders
	train_dataset = SocialDataset(set_name="train", b_size=hyperparams["train_b_size"], t_tresh=hyperparams["time_thresh"], d_tresh=hyperparams["dist_thresh"], verbose=args.verbose)
	test_dataset = SocialDataset(set_name="test", b_size=hyperparams["test_b_size"], t_tresh=hyperparams["time_thresh"], d_tresh=hyperparams["dist_thresh"], verbose=args.verbose)

	# shift origin and scale data
	for traj in train_dataset.trajectory_batches:
		traj -= traj[:, :1, :]
		traj *= hyperparams["data_scale"]
	for traj in test_dataset.trajectory_batches:
		traj -= traj[:, :1, :]
		traj *= hyperparams["data_scale"]

	best_ade = 50 # start saving after this threshold
	best_fde = 50
	best_metrics = {}
	N = hyperparams["n_values"]

	for e in range(hyperparams["num_epochs"]):
		# train_loss_dict = train_engine(train_dataset, model, device, hyperparams, optimizer, args.wandb)
		test_error_dict = test_engine(test_dataset, model, device, hyperparams, args.wandb, best_of_n = N)

		if test_error_dict["ade"] < best_ade:
			best_ade = test_error_dict["ade"]
			best_metrics["best_ade"] = (best_ade, e)
			if best_ade < 10.25:
				save_path = "../saved_models/" + args.version + ".pt"
				torch.save({
							"hyperparams": hyperparams,
							"model_state_dict": model.state_dict(),
							"optimizer_state_dict": optimizer.state_dict()
							}, save_path)
				if args.wandb:
					save_model_wandb(save_path)
				print(f"Saved model to: {save_path}")

		if test_error_dict["fde"] < best_fde:
			best_fde = test_error_dict["fde"]
			best_metrics["best_fde"] = (best_fde, e)
		
		if args.wandb:
			log_losses(losses=train_loss_dict, mode="train", epoch=e)
			log_metrics(metrics=test_error_dict, mode="test", epoch=e)
			log_summary(best_metrics=best_metrics)

		loss_str = ""
		for key in train_loss_dict:
			loss_str += f"{key} = {train_loss_dict[key]}   "
		error_str = ""
		for key in test_error_dict:
			error_str += f"{key} = {test_error_dict[key]}   "

		print(f"\r[Epoch {e}]   [Best ADE {best_ade}]   [Best FDE {best_fde}]   {loss_str}   {error_str}", end='')
