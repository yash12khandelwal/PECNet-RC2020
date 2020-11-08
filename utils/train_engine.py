import torch
import torch.nn as nn
from collections import defaultdict

from utils.social_utils import calculate_loss

def train_engine(train_dataset, model, device, hyperparams: dict, optimizer):

	model.train()
	loss_dict = defaultdict(0)
	criterion = nn.MSELoss()

	for _, (traj, mask, initial_pos) in enumerate(zip(train_dataset.trajectory_batches, train_dataset.mask_batches, train_dataset.initial_pos_batches)):
		traj, mask, initial_pos = torch.DoubleTensor(traj).to(device), torch.DoubleTensor(mask).to(device), torch.DoubleTensor(initial_pos).to(device)
		x = traj[:, :hyperparams["past_length"], :]
		y = traj[:, hyperparams["past_length"]:, :]

		x = x.contiguous().view(-1, x.shape[1]*x.shape[2]) # (x,y,x,y ... )
		x = x.to(device)
		dest = y[:, -1, :].to(device)
		future = y[:, :-1, :].contiguous().view(y.size(0),-1).to(device)

		dest_recon, mu, var, interpolated_future = model.forward(x, initial_pos, dest=dest, mask=mask, device=device)

		optimizer.zero_grad()
		batch_loss_dict = calculate_loss(dest, dest_recon, mu, var, criterion, future, interpolated_future)
		loss = batch_loss_dict["ael"] + batch_loss_dict["kld"]*hyperparams["kld_reg"] + batch_loss_dict["atl"]*hyperparams["adl_reg"]
		loss.backward()

		loss_dict["total_train_loss"] += loss.item()
		loss_dict["total_ael_loss"] += batch_loss_dict["ael"].item()
		loss_dict["total_kld_loss"] += batch_loss_dict["kld"].item()
		loss_dict["total_atl_loss"] += batch_loss_dict["atl"].item()
		optimizer.step()

	return loss_dict
