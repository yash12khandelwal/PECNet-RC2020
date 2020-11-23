import torch
import numpy as np

from collections import defaultdict
from torch.utils.data import DataLoader

def test_engine(dataset_type, test_dataset, model, device, hyperparams: dict, best_of_n: int = 1) -> dict:
	"""[summary]

	Arguments:
		dataset_type {string} -- Dataset type (ETH_UCY or drone)
		test_dataset -- Test Dataset
		model -- PECNet object
		device -- Device to use (GPU or CPU)
		hyperparams {dict} -- Dictionary stores all the hyperparams that are used while training

	Keyword Arguments:
		best_of_n {int} -- [description] (default: {1})

	Returns:
		dict -- Dictionary of the test error
	"""
	model.eval()
	assert best_of_n >= 1 and type(best_of_n) == int
	error_dict = defaultdict(lambda: 0)
	if dataset_type=="drone":
		with torch.no_grad():
			for i, (traj, mask, initial_pos) in enumerate(zip(test_dataset.trajectory_batches, test_dataset.mask_batches, test_dataset.initial_pos_batches)):
				
				traj = torch.DoubleTensor(traj).to(device)
				mask = torch.DoubleTensor(mask).to(device)
				initial_pos = torch.DoubleTensor(initial_pos).to(device)

				past_traj = traj[:, :hyperparams["past_length"], :]
				future_traj = traj[:, hyperparams["past_length"]:, :]
				future_traj = future_traj.cpu().numpy()

				past_traj = past_traj.view(-1, past_traj.shape[1]*past_traj.shape[2]).to(device)
				dest = future_traj[:, -1, :]

				all_fde = []
				all_guesses = []
				for _ in range(best_of_n):
					pred_dest = model.forward(past_traj, initial_pos, device=device, mask=None)
					pred_dest = pred_dest.cpu().numpy()
					all_guesses.append(pred_dest)

					fde = np.linalg.norm(pred_dest - dest, axis = 1)
					all_fde.append(fde)

				all_fde = np.array(all_fde)
				all_guesses = np.array(all_guesses)

				# final displacement error
				fde = np.mean(all_fde)
				# minimum final displacement error
				avg_min_fde = np.mean(np.min(all_fde, axis = 0))
				
				# choosing the best guess
				indices = np.argmin(all_fde, axis = 0)
				# TODO didn't understood this
				best_guess_dest = all_guesses[indices, np.arange(past_traj.shape[0]),  :]
				best_guess_dest = torch.DoubleTensor(best_guess_dest).to(device)

				# using the best guess for interpolation
				interpolated_future = model.predict(past_traj, best_guess_dest, mask, initial_pos)

				# average displacement error
				interpolated_future = interpolated_future.cpu().numpy()
				best_guess_dest = best_guess_dest.cpu().numpy()
				predicted_future = np.concatenate((interpolated_future, best_guess_dest), axis = 1)
				predicted_future = np.reshape(predicted_future, (-1, hyperparams["future_length"], 2))
				ade = np.mean(np.linalg.norm(future_traj - predicted_future, axis = 2))

				error_dict["ade"] += ade /hyperparams["data_scale"]
				error_dict["fde"] += avg_min_fde / hyperparams["data_scale"]
				error_dict["fde_avg"] = fde / hyperparams["data_scale"]

		for key in error_dict:
			error_dict[key] /= (i + 1)

		return error_dict
	else :
		dataloader = DataLoader(
			test_dataset, batch_size=128, shuffle=True, num_workers=0)
		with torch.no_grad():
			for i, traj in enumerate(dataloader):	
				traj = torch.DoubleTensor(traj).to(device)
				traj = traj - traj[:,:1,:]
				past_traj = traj[:, :hyperparams["past_length"], 1:]
				future_traj = traj[:, hyperparams["past_length"]:, 1:]
				future_traj = future_traj.cpu().numpy()

				past_traj = past_traj.contiguous().view(-1, past_traj.shape[1]*past_traj.shape[2]).to(device)
				dest = future_traj[:, -1, :]

				all_fde = []
				all_guesses = []
				for _ in range(best_of_n):
					pred_dest = model.forward(past_traj, None, device=device, mask=None)
					pred_dest = pred_dest.cpu().numpy()
					all_guesses.append(pred_dest)

					fde = np.linalg.norm(pred_dest - dest, axis = 1)
					all_fde.append(fde)

				all_fde = np.array(all_fde)
				all_guesses = np.array(all_guesses)

				# final displacement error
				fde = np.mean(all_fde)
				# minimum final displacement error
				avg_min_fde = np.mean(np.min(all_fde, axis = 0))
				
				# choosing the best guess
				indices = np.argmin(all_fde, axis = 0)
				# TODO didn't understood this
				best_guess_dest = all_guesses[indices, np.arange(past_traj.shape[0]),  :]
				best_guess_dest = torch.DoubleTensor(best_guess_dest).to(device)

				# using the best guess for interpolation
				interpolated_future = model.predict(past_traj, best_guess_dest, None, None)

				# average displacement error
				interpolated_future = interpolated_future.cpu().numpy()
				best_guess_dest = best_guess_dest.cpu().numpy()
				predicted_future = np.concatenate((interpolated_future, best_guess_dest), axis = 1)
				predicted_future = np.reshape(predicted_future, (-1, hyperparams["future_length"], 2))
				ade = np.mean(np.linalg.norm(future_traj - predicted_future, axis = 2))

				error_dict["ade"] += ade /hyperparams["data_scale"]
				error_dict["fde"] += avg_min_fde / hyperparams["data_scale"]
				error_dict["fde_avg"] = fde / hyperparams["data_scale"]

		for key in error_dict:
			error_dict[key] /= (i + 1)

		return error_dict