import wandb
import os
import numpy as np
import matplotlib.pyplot as plt

thres = 2
def init_wandb(cfg: dict, model, args=None) -> None:
    """Initialize project on Weights & Biases
    Args:
        cfg (dict): Configuration dictionary
    """
    wandb.init(
        name=args.version,
        project="Trajectory Prediction",
        config=cfg,
        dir="~/",
    )
    if args:
        wandb.config.update(args)

    wandb.watch(model, log="all")

def save_model_wandb(save_path: str):
    """Save model weights to wandb
    """
    wandb.save(os.path.abspath(save_path))

def log_losses(losses: dict, mode: str, epoch: int):
    """
    Log the losses
    Args:
        losses (dict): all the losses should be of type float
        mode (str): "train" or "val"
        epoch (int): epoch number
    """

    for k, v in losses.items():
        wandb.log({f"{mode}/{k}": v}, step=epoch)


def log_metrics(metrics: dict, mode: str, epoch: int):
    """
    Log the metrics
    Args:
        metrics (dict): all the metrics should be of type float
        mode (str): "train" or "val"
        epoch (int): epoch number
    """

    for k, v in metrics.items():
        wandb.log({f"{mode}/{k}": v}, step=epoch)

def log_summary(best_metrics: dict):
    
    for key in best_metrics:
        wandb.run.summary[f"{key}"] = best_metrics[key]

def visualize(past_traj, initial_pos, future_traj, predicted_future_traj, predicted_destinations, mask, filename, pred_future_ws = None):
    """Visualize the predictions and grund truth
    Arguments:
        past_traj <tensor> : Past trajectory of size (batch, past_length, 2)
        initial_pos <tensor> : Initial position of all the samples of size (batch, 2)
        future_traj <tensor> : Ground truth future trajectory of size (batch, future_length, 2)
        predicted_future_traj <tensor> : Predicted future trajectory of size (batch, future_length, 2)
        predicted_destinations <tensor> : Sampled destinations (k, batch_size, 2)
        mask <tensor> : Social mask of size (batch_size, batch_size)
        filename <string> : File name
    """
    batch_size = past_traj.shape[0]
    past_traj = np.array(past_traj)
    initial_pos = np.array(initial_pos.reshape(batch_size,1,2))
    print(initial_pos)
    past_traj = past_traj + initial_pos*1000
    future_traj = future_traj + initial_pos*1000
    predicted_future_traj = predicted_future_traj + initial_pos*1000
    predicted_destinations = predicted_destinations + initial_pos.transpose((1,0,2))*1000
    if pred_future_ws is not None :
        pred_future_ws = pred_future_ws + initial_pos*1000

    for i in range(batch_size):
        print(i)
        #fig = plt.figure()
        neighbors_list = []
        #ax = plt.gca()
        for j in range(batch_size):
            if mask[i][j]==1 :
                neighbors_list.append(j)
        
        print('No of people : ', len(neighbors_list))
        if len(neighbors_list) > thres :
            filename = '../visualization/multiple/' + str(i)+'a'+str(len(neighbors_list)) + '.png'
            plt.plot(past_traj[neighbors_list,:,0].T, past_traj[neighbors_list,:,1].T, label = 'Past trajectory', color = 'b')
            #print(past_traj[neighbors_list,:,0].T)
            plt.plot(future_traj[neighbors_list,:,0].T, future_traj[neighbors_list,:,1].T, label = 'Ground truth future trajectory', color = 'g')
            plt.plot(predicted_future_traj[neighbors_list,:,0].T, predicted_future_traj[neighbors_list,:,1].T, label = 'Predicted future trajectory', color = 'r')
            if pred_future_ws is not None :
                plt.plot(pred_future_ws[neighbors_list,:,0].T, pred_future_ws[neighbors_list,:,1].T, label = 'Predicted future trajectory without SP', color = 'r')
            # plt.scatter(predicted_destinations[:, neighbors_list, 0], predicted_destinations[:, neighbors_list, 1], label = 'Predicted destinations', color = 'k')
            handles, labels = plt.gca().get_legend_handles_labels()
            labels, ids = np.unique(labels, return_index=True)
            handles = [handles[i] for i in ids]
            plt.legend(handles, labels, loc='best')
            #plt.legend(loc='best')
            plt.savefig(filename)
            plt.close()


                    