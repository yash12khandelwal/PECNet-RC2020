import wandb

def init_wandb(cfg: dict, args=None) -> None:
    '''Initialize project on Weights & Biases
    Args:
        cfg (dict): Configuration dictionary
    '''
    wandb.init(
        name=args.version,
        project='Trajectory Prediction',
        config=cfg,
        dir='~/',
    )
    if args:
        wandb.config.update(args)
