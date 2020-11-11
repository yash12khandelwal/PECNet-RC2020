import torch
import torch.nn as nn
from collections import defaultdict

from utils.social_utils import calculate_loss

def train_engine(train_dataset, model, device, hyperparams: dict, optimizer) -> dict:
	"""General training function

	Arguments:
		train_dataset -- Training dataset
		model -- PECNet object
		device -- Device to use (GPU or CPU)
		hyperparams {dict} -- Dictionary stores all the hyperparams that are used while training
		optimizer -- Optimizer to be used

	Returns:
		dict -- Training loss dictionary
				4 keys -> total_train_loss - weighted sum of all the component losses for the epoch
						  total_ael_loss - Average Endpoint Loss for the epoch
						  total_kld_loss - KL Divergence Loss for the epoch
						  total_atl_loss - Average Trajectory Loss for the epoch
	"""
	model.train()
	loss_dict = defaultdict(lambda : 0)
	criterion = nn.MSELoss()
	dataloader = data.DataLoader(
			train_dataset, batch_size=128, shuffle=True, num_workers=0)
	for i, traj in enumerate(dataloader):
		traj = torch.DoubleTensor(traj).to(device)
		traj = traj - traj[:,:1,:]
		x = traj[:, :hyper_params['past_length'], 1:]
		y = traj[:, hyper_params['past_length']:, 1:]
		x = x.contiguous().view(-1, x.shape[1]*x.shape[2]) # (x,y,x,y ... )
		x = x.to(device)
		dest = y[:, -1, :].to(device)
		future = y[:, :-1, :].contiguous().view(y.size(0),-1).to(device)
		
		dest_recon, mu, var, interpolated_future = model.forward(x, dest=dest, device=device)
		optimizer.zero_grad()
		batch_loss_dict = calculate_loss(criterion, dest, pred_dest, mu, var, gt_future, interpolated_future)
		loss = batch_loss_dict["ael"] + batch_loss_dict["kld"]*hyperparams["kld_reg"] + batch_loss_dict["atl"]*hyperparams["adl_reg"]
		loss.backward()

		loss_dict["total_train_loss"] += loss.item()
		loss_dict["total_ael_loss"] += batch_loss_dict["ael"].item()
		loss_dict["total_kld_loss"] += batch_loss_dict["kld"].item()
		loss_dict["total_atl_loss"] += batch_loss_dict["atl"].item()
		optimizer.step()

	return loss_dict
