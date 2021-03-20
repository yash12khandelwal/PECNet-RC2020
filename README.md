## Introduction 
This repository is the reproducibility report of 'It Is Not the Journey but the Destination: Endpoint Conditioned Trajectory Prediction'. You can find the original code from the [repository](https://github.com/HarshayuGirase/PECNet). 

This repository is an extension which contains code for executing all the experiments as well as the additional experiment that we performed. Also, the code has been well commented and organized for easy understanding and use.

## Setup
All code was developed and tested on Ubuntu 16.04.6 with Python 3.6.6 and PyTorch 1.4.0 with CUDA 10.0.

### Steps:
```
1. pip install -r requirements.txt
2. wandb login 0b5f978ab6f86d54b1ee4e485a8e477bfedcb491
```

## Pretrained Models
Pretrained models are available in the `saved_models` folder.

## Configuration File
Configuration files (or config files) are used to load parameters such as hidden layer dimensions or learning rates into a model to be trained. To do this, first edit any of the parameters in the contents dictionary in config_gen.py in the config folder. Next, run config_gen.py using the following commands:
```bash
# Start in the project root directory
cd config
python config_gen.py -fn <config_save_name>
```
where config_save_name is the name that the config file should be saved with ending in .yaml.

## Running Models
You can run the commmands:
```bash
# Start in the project root directory
cd scripts
python test_pretrained_model.py -lf <file_to_load>
```
to easily run any of the pretrained models. The `file_to_load` is the name of the model ending in `.pt` in the `saved_models` folder. For example you can replicate our Table 1 results like this:

```bash
# Start in the project root directory
cd scripts
python test_pretrained_model.py -lf PECNET_social_model1.pt
```

## Training new models
To train a new model, you can run the command:
```bash
# Start in the project root directory
cd scripts
python training_loop.py -cfn <config_file_name> -sf <model_save_name>
```
where `config_file_name` is the name of the config file used to load the configuration parameters ending in `.yaml` and `model_save_name` is the name that is used when saving the model ending in `.pt`. You can use our optimal parameters as given by `optimal.yaml` or create your own config file by changing parameters in and running `config_gen.py` in the config folder.

## Experiment Results
Reproduce the following experiment results using the given specific instructions:-

## Drone/ETH/UNIV/ZARA1/ZARA2/HOTEL results (Table 1 and 2)
Run run_experiment.py with the dataset argument set to the corresponding dataset (drone, eth, hotel, zara1 or zara2). Use the -S option to not use social pooling and -TT option to not use truncation trick. For eg to reproduce results for drone dataset without social pooling and truncation trick, run the following command from scripts directory :-
```bash
python run_experiment.py --experiment default --dataset drone -S -TT
```

## K_variation results
Execute the following command to get results on drone dataset for some value of k, use -TT if required to not use truncation trick
```bash
python run_experiment.py --experiment k_variation --dataset drone -k <k value> <-TT>
```

## Waypoint conditioning and oracle results
Execute the following command with waypoint_no between 1 and 11 to get results for waypoint conditioning error without oracle
```bash
python run_experiment.py --experiment waypoint_conditioning --dataset drone -n <Conditioned waypoint no.>
```
Execute the following command to get the results with oracle
```bash
python run_experiment.py --experiment waypoint_conditioning_oracle --dataset drone -n <Conditioned waypoint no.>
```

## Design choice for VAE experiment results
Execute the following command :-
```bash
python run_experiment.py --experiment design_choice_for_VAE
```
