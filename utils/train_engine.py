import torch
import torch.nn as nn

from utils.social_utils import calculate_loss

def train_engine(train_dataset, model, device, hyperparams, optimizer):

	model.train()
	train_loss = 0
	total_rcl, total_kld, total_adl = 0, 0, 0
	criterion = nn.MSELoss()

	for _, (traj, mask, initial_pos) in enumerate(zip(train_dataset.trajectory_batches, train_dataset.mask_batches, train_dataset.initial_pos_batches)):
		traj, mask, initial_pos = torch.DoubleTensor(traj).to(device), torch.DoubleTensor(mask).to(device), torch.DoubleTensor(initial_pos).to(device)
		x = traj[:, :hyperparams['past_length'], :]
		y = traj[:, hyperparams['past_length']:, :]

		x = x.contiguous().view(-1, x.shape[1]*x.shape[2]) # (x,y,x,y ... )
		x = x.to(device)
		dest = y[:, -1, :].to(device)
		future = y[:, :-1, :].contiguous().view(y.size(0),-1).to(device)

		dest_recon, mu, var, interpolated_future = model.forward(x, initial_pos, dest=dest, mask=mask, device=device)

		optimizer.zero_grad()
		rcl, kld, adl = calculate_loss(dest, dest_recon, mu, var, criterion, future, interpolated_future)
		loss = rcl + kld*hyperparams['kld_reg'] + adl*hyperparams['adl_reg']
		loss.backward()

		train_loss += loss.item()
		total_rcl += rcl.item()
		total_kld += kld.item()
		total_adl += adl.item()
		optimizer.step()

	return {
		"total_train_loss": train_loss,
		"total_rcl_loss": total_rcl,
		"total_kld_loss": total_kld,
		"total_adl_loss": total_adl
	}