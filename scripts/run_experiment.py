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
from utils.social_utils import SocialDataset, ETHDataset, set_seed
from utils.train_engine import train_engine
from utils.test_engine import test_engine
from visualization.wandb_utils import init_wandb, log_losses, save_model_wandb, log_metrics, log_summary

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="PECNet")
	parser.add_argument("--num_workers", "-nw", type=int, default=0)
	parser.add_argument("--gpu_index", "-gi", type=int, default=0)
	# parser.add_argument("--config_filename", "-cfn", type=str, default="optimal_drone.yaml")
	parser.add_argument("--version", "-v", type=str, default="PECNET_social_model")
	parser.add_argument("--verbose", action="store_true")
	parser.add_argument("-s", "--seed", default=42, help="Random seed")
	parser.add_argument("-w", "--wandb", action="store_true", help="Log to wandb or not")
	parser.add_argument("-d", "--dataset", default="drone", help="The datset to train the model on (ETH_UCY or drone)")
	parser.add_argument("-e", "--experiment", default="default", help="Which experiment to perform : none (To train the model as per cfg), default, k_variation, waypoint_conditioning, waypoint_conditioning_oracle or design_choice_for_VAE")
	parser.add_argument("-k", "--k_val", default=20, help="Valid only for k_variation experiment")
	parser.add_argument("-TT", "--truncation_trick", action="store_true", help="Use this option to not use truncation trick")
	parser.add_argument("-S", "--social_pooling", action="store_true", help="Use this option to not use social pooling")
	parser.add_argument("-n", "--conditioned_waypoint", default=5, help="Valid only for waypoint_conditioning OR waypoint_conditioning_oracle experiment, value should be between 1 to 11")
	parser.add_argument("-vis", "--visualize", action="store_true")
	args = parser.parse_args()
	args.k_val = int(args.k_val)
	args.conditioned_waypoint = int(args.conditioned_waypoint)
	# setting seed system wide for proper reproducibility
	set_seed(int(args.seed))
	if args.dataset=="drone" :
		config_filename = "optimal_drone.yaml"
	else :
		config_filename = "optimal_eth.yaml"
	torch.set_default_dtype(torch.float64)
	
	# set device
	device = torch.device("cuda", index=args.gpu_index) if torch.cuda.is_available() else torch.device("cpu")
	if torch.cuda.is_available():
		torch.cuda.set_device(args.gpu_index)
	print(f"Using {device} device")

	# load hyperparams from config file and update the dict
	with open("../config/" + config_filename, "r") as file:
		try:
			hyperparams = yaml.load(file, Loader = yaml.FullLoader)
		except:
			hyperparams = yaml.load(file)
	file.close()
	hyperparams['conditioned_waypoint'] = args.conditioned_waypoint
	# Change params for truncation trick
	hyperparams['visualize'] = args.visualize
	# Change params if social pooling is not applicable
	if args.social_pooling:
		hyperparams['nonlocal_pools']=0

	if args.experiment=="k_variation":
		hyperparams['n_values'] = args.k_val
	
	if args.truncation_trick==False:
		if hyperparams['n_values']>3 :
			hyperparams['sigma'] = 1.3
		else:
			hyperparams['sigma'] = 1
	else:
		hyperparams['sigma'] = 1
	print(hyperparams)
	model = PECNet(args.experiment, args.dataset, hyperparams["enc_past_size"], hyperparams["enc_dest_size"], hyperparams["enc_latent_size"], hyperparams["dec_size"], hyperparams["predictor_hidden_size"], hyperparams["non_local_theta_size"], hyperparams["non_local_phi_size"], hyperparams["non_local_g_size"], hyperparams["fdim"], hyperparams["zdim"], hyperparams["nonlocal_pools"], hyperparams["non_local_dim"], hyperparams["sigma"], hyperparams["past_length"], hyperparams["future_length"], args.verbose)
	model = model.double().to(device)
	# initailize wandb, save the gradients and model information to wandb
	if args.wandb:
		init_wandb(hyperparams.copy(), model, args)
	
	# Only for k_variation experiment load the pretrained model and run it
	if args.experiment == "k_variation":
		checkpoint = torch.load(f"../saved_models/PECNET_social_model1.pt", map_location=device)
		model.load_state_dict(checkpoint["model_state_dict"])
		train_dataset = SocialDataset(set_name="train", b_size=hyperparams["train_b_size"], t_tresh=hyperparams["time_thresh"], d_tresh=hyperparams["dist_thresh"], verbose=args.verbose)
		test_dataset = SocialDataset(set_name="test", b_size=hyperparams["test_b_size"], t_tresh=hyperparams["time_thresh"], d_tresh=hyperparams["dist_thresh"], verbose=args.verbose)
		# shift origin and scale data
		for traj in train_dataset.trajectory_batches:
			traj -= traj[:, 0:1, :]
			traj *= hyperparams["data_scale"]
		for traj in test_dataset.trajectory_batches:
			traj -= traj[:, 0:1, :]
			traj *= hyperparams["data_scale"]
		test_error_dict = test_engine(args.dataset, test_dataset, model, device, hyperparams, best_of_n = hyperparams['n_values'], experiment = args.experiment)
		print("Best ADE :" + str(test_error_dict['ade']))
		print("Best FDE :" + str(test_error_dict['fde']))
		exit()


	# Only for k_variation experiment load the pretrained model and run it
	if args.experiment == "compare_with_without_s":
		checkpoint = torch.load(f"../saved_models/PECNET_social_model1.pt", map_location=device)
		model.load_state_dict(checkpoint["model_state_dict"])
		model_ws = PECNet(args.experiment, args.dataset, hyperparams["enc_past_size"], hyperparams["enc_dest_size"], hyperparams["enc_latent_size"], hyperparams["dec_size"], hyperparams["predictor_hidden_size"], hyperparams["non_local_theta_size"], hyperparams["non_local_phi_size"], hyperparams["non_local_g_size"], hyperparams["fdim"], hyperparams["zdim"], 0, hyperparams["non_local_dim"], hyperparams["sigma"], hyperparams["past_length"], hyperparams["future_length"], args.verbose)
		model_ws = model_ws.double().to(device)
		checkpoint_ws = torch.load(f"../saved_models/PECNET_ws.pt", map_location=device)
		model_ws.load_state_dict(checkpoint_ws["model_state_dict"])
		train_dataset = SocialDataset(set_name="train", b_size=hyperparams["train_b_size"], t_tresh=hyperparams["time_thresh"], d_tresh=hyperparams["dist_thresh"], verbose=args.verbose)
		test_dataset = SocialDataset(set_name="test", b_size=hyperparams["test_b_size"], t_tresh=hyperparams["time_thresh"], d_tresh=hyperparams["dist_thresh"], verbose=args.verbose)
		# shift origin and scale data
		for traj in train_dataset.trajectory_batches:
			traj -= traj[:, 0:1, :]
			traj *= hyperparams["data_scale"]
		for traj in test_dataset.trajectory_batches:
			traj -= traj[:, 0:1, :]
			traj *= hyperparams["data_scale"]
		test_error_dict = test_engine(args.dataset, test_dataset, model, device, hyperparams, best_of_n = hyperparams['n_values'], experiment = args.experiment, model_ws = model_ws)
		print("Best ADE :" + str(test_error_dict['ade']))
		print("Best FDE :" + str(test_error_dict['fde']))
		exit()

	# initialize optimizer
	# optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
	optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

	# initialize dataloaders
	if args.dataset == "drone":
		train_dataset = SocialDataset(set_name="train", b_size=hyperparams["train_b_size"], t_tresh=hyperparams["time_thresh"], d_tresh=hyperparams["dist_thresh"], verbose=args.verbose)
		test_dataset = SocialDataset(set_name="test", b_size=hyperparams["test_b_size"], t_tresh=hyperparams["time_thresh"], d_tresh=hyperparams["dist_thresh"], verbose=args.verbose)
		# shift origin and scale data
		for traj in train_dataset.trajectory_batches:
			traj -= traj[:, :1, :]
			traj *= hyperparams["data_scale"]
		for traj in test_dataset.trajectory_batches:
			traj -= traj[:, :1, :]
			traj *= hyperparams["data_scale"]
	else:
		train_dataset = ETHDataset(dataset=args.dataset, set_name="train", b_size=hyperparams["train_b_size"], t_tresh=hyperparams["time_thresh"], d_tresh=hyperparams["dist_thresh"], verbose=args.verbose)
		test_dataset = ETHDataset(dataset=args.dataset, set_name="test", b_size=hyperparams["test_b_size"], t_tresh=hyperparams["time_thresh"], d_tresh=hyperparams["dist_thresh"], verbose=args.verbose)
	

	best_ade = 50 # start saving after this threshold
	best_fde = 50
	best_wpe = 50
	best_metrics = {}
	N = hyperparams["n_values"]

	for e in range(hyperparams["num_epochs"]):
		train_loss_dict = train_engine(args.dataset, train_dataset, model, device, hyperparams, optimizer, experiment = args.experiment)
		test_error_dict = test_engine(args.dataset, test_dataset, model, device, hyperparams, best_of_n = N, experiment = args.experiment)

		if test_error_dict["ade"] < best_ade:
			best_ade = test_error_dict["ade"]
			best_metrics["best_ade"] = (best_ade, e)
			if best_ade < 14.25:
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

		if args.experiment == "waypoint_conditioning":
			if test_error_dict["wpe"] < best_wpe:
				best_wpe = test_error_dict["wpe"]
				best_metrics["best_wpe"] = (best_wpe, e)
			print(f"\r[Epoch {e}]   [Best ADE {best_ade}]   [Best FDE {best_fde}]   [Best WPE {best_wpe} ] {loss_str}   {error_str}", end='')
		else:
			print(f"\r[Epoch {e}]   [Best ADE {best_ade}]   [Best FDE {best_fde}]   {loss_str}   {error_str}", end='')
