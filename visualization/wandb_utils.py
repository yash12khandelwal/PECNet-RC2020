import wandb
import os

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

def log_prediction_images(past_traj_original_shape, future_traj, predicted_future):

    print(predicted_future[0], future_traj[0], past_traj_original_shape[0])
    print(predicted_future.shape, future_traj.shape, past_traj_original_shape.shape)
    data = []
    data.extend(predicted_future[0].tolist())
    data.extend(future_traj[0].tolist())
    data.extend(past_traj_original_shape[0].tolist())
    print(len(data))
    # data = [[x, y] for (x, y) in zip(class_x_prediction_scores, class_y_prediction_scores)]
    print(data)
    table = wandb.Table(data=data, columns = ["y coordinate", "x coordinate"])
    wandb.log({"my_custom_id" : wandb.plot.scatter(table, "y coordinate", "y coordinate")})

    # fig = plt.figure()
    # plt.scatter(predicted_future[23][:, 0], predicted_future[23][:, 1], label='predicted_future')
    # plt.scatter(future_traj[23][:, 0], future_traj[23][:, 1], label='ground_truth_future')
    # plt.scatter(past_traj_original_shape[23][:, 0], past_traj_original_shape[23][:, 1], label='past')
    # plt.legend()
    # plt.savefig('a.png')
    # fig = plt.figure()
    # plt.scatter(predicted_future[21][:, 0], predicted_future[21][:, 1], label='predicted_future')
    # plt.scatter(future_traj[21][:, 0], future_traj[21][:, 1], label='ground_truth_future')
    # plt.scatter(past_traj_original_shape[21][:, 0], past_traj_original_shape[21][:, 1], label='past')
    # plt.legend()
    # plt.savefig('b.png')
    # exit()
