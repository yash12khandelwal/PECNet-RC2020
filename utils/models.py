import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import pdb
from torch.nn import functional as F
from torch.distributions.normal import Normal
import math
import numpy as np
import yaml

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: tuple = (1024, 512), activation: str = "relu", discrim: bool = False, dropout: float = -1):
        """Constructor of MLP Model Class
        Takes in input about all the information required to create a Multi Layer Fully Connected Neural Network

        Arguments:
            input_dim {int} -- Input dimension of the MLP
            output_dim {int} -- Output dimension of the MLP

        Keyword Arguments:
            hidden_size {tuple} -- Dimensions of the hidden layer (default: {(1024, 512)})
            activation {str} -- Activation function to be used between layers (default: {"relu"})
            discrim {bool} -- True if use Sigmoid after the last layer else False (default: {False})
            dropout {float} -- Dropout value to be used between layers (default: {-1})
        """
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of the MLP network

        Arguments:
            x {torch.Tensor} -- Input to the MLP network

        Returns:
            torch.Tensor -- Output after forward pass to the MLP network
        """
        for i in range(len(self.layers)):
            x = self.layers[i](x)

            # if not last layer, then use the activation function provided
            # if last layer then use sigmoid activation function available else no activation function for last layer
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)

        return x

class PECNet(nn.Module):

    def __init__(self, enc_past_size: list, enc_dest_size: list, enc_latent_size: list, dec_size: list, predictor_size: list, non_local_theta_size: list, non_local_phi_size: list, non_local_g_size: list, fdim: int, zdim: int, nonlocal_pools: int, non_local_dim: int, sigma: float, past_length: int, future_length: int, verbose: bool):
        """PECNet Model Construction
        Constructed sub-modules of the PECNet model on the basis of the input dimension

        Arguments:
            enc_past_size {list} -- Dimension of hidden layer of past trajectory encoder
            enc_dest_size {list} -- Dimension of hidden layer of destination encoder
            enc_latent_size {list} -- Dimension of hidden layer of latent encoder
            dec_size {list} -- Dimensions of hidden layer of CVAE Decoder
            predictor_size {list} -- Dimension of hidden layer of final prediction layer
            non_local_theta_size {list} -- Dimensions of hidden layer of theta network of Social pooling module
            non_local_phi_size {list} -- Dimensions of hidden layer of phi network of Social pooling module
            non_local_g_size {list} -- Dimensions of hidden layer of g network of Social pooling module
            fdim {int} -- Output dimension of the past trajectory and destination position encoder
            zdim {int} -- Dimension of the latent space
            nonlocal_pools {int} -- No. of iterations of regression through social pooling layer
            non_local_dim {int} -- Output dimension of subnetwork (theta, phi) of Social Pooling Module
            sigma {float} -- Variance of the normal distribution from which to sample the z vector
            past_length {int} -- No. of points taken to encode past trajectory
            future_length {int} -- No. of points to be predicted of the future trajectory
            verbose {bool} -- True if want to print the architecture information, else False
        """
        super(PECNet, self).__init__()

        self.zdim = zdim   # Dimension of the latent variable
        self.nonlocal_pools = nonlocal_pools  
        self.sigma = sigma  # For testing the latent variable is sampled as mu=0 and variance=sigma

        # past trajectory information encoder
        # input -> trajectory points -> (x_coordinate, y_coordinate)
        self.encoder_past = MLP(input_dim = past_length*2, output_dim = fdim, hidden_size=enc_past_size)

        # destination position encoder
        # input -> (x_coordinate, y_coordinate)
        self.encoder_dest = MLP(input_dim = 2, output_dim = fdim, hidden_size=enc_dest_size)

        # CVAE encoder - decoder
        # input -> concatenated encoded information from the past and dest encoder
        self.encoder_latent = MLP(input_dim = 2*fdim, output_dim = 2*zdim, hidden_size=enc_latent_size)
        # input -> concatenated past traj encoded info and encoded latent vector
        # output -> destination coordinate -> (x_coordinate, y_coordinate)
        self.decoder = MLP(input_dim = fdim + zdim, output_dim = 2, hidden_size=dec_size)

        # These 3 layers are used to establish social pooling among vehicles
        self.non_local_theta = MLP(input_dim = 2*fdim + 2, output_dim = non_local_dim, hidden_size=non_local_theta_size)
        self.non_local_phi = MLP(input_dim = 2*fdim + 2, output_dim = non_local_dim, hidden_size=non_local_phi_size)
        self.non_local_g = MLP(input_dim = 2*fdim + 2, output_dim = 2*fdim + 2, hidden_size=non_local_g_size)

        # This layer is used to make the final trajectory points prediction except the final point (which is already predicted)
        self.predictor = MLP(input_dim = 2*fdim + 2, output_dim = 2*(future_length-1), hidden_size=predictor_size)

        architecture = lambda net: [l.in_features for l in net.layers] + [net.layers[-1].out_features]

        if verbose:
            print(f"Past Encoder architecture : {architecture(self.encoder_past)}")
            print(f"Dest Encoder architecture : {architecture(self.encoder_dest)}")
            print(f"Latent Encoder architecture : {architecture(self.encoder_latent)}")
            print(f"Decoder architecture : {architecture(self.decoder)}")
            print(f"Predictor architecture : {architecture(self.predictor)}")

            print(f"Non Local Theta architecture : {architecture(self.non_local_theta)}")
            print(f"Non Local Phi architecture : {architecture(self.non_local_phi)}")
            print(f"Non Local g architecture : {architecture(self.non_local_g)}")

    def non_local_social_pooling(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Social Pooling Module forward function

        Arguments:
            feat {torch.Tensor} -- Predicted features (past_encoder + generated_dest + initial_pos)
            mask {torch.Tensor} -- Social Mask (batch_size, batch_size)

        Returns:
            torch.Tensor -- socially pooled features + predicted features (past_encoder + generated_dest + initial_pos)
        """
        theta_x = self.non_local_theta(feat)
        phi_x = self.non_local_phi(feat).transpose(1,0)

        f = torch.matmul(theta_x, phi_x)
        f_weights = F.softmax(f, dim = -1)
        f_weights = f_weights * mask # setting weights of non neighbours to zero
        f_weights = F.normalize(f_weights, p=1, dim=1)

        pooled_f = torch.matmul(f_weights, self.non_local_g(feat))

        return pooled_f + feat

    def forward(self, x: torch.Tensor, initial_pos: torch.Tensor, dest: torch.Tensor = None, mask: torch.Tensor = None, device=torch.device("cpu")) -> tuple:
        """Forward function of the PECNet model.
        This function gets called to do the forward pass through the network.

        Arguments:
            x {torch.Tensor} -- Past Trajectory points -> (batch_size, No. of points * 2)
            initial_pos {torch.Tensor} -- Initial position of the people -> (batch_size, 2)

        Keyword Arguments:
            dest {torch.Tensor} -- Ground Truth destination of the people (default: {None})
            mask {torch.Tensor} -- Social Mask (default: {None})
            device -- (default: {torch.device("cpu")})

        Returns:
            tuple -- If training, the tuple returns the destination point, mean, logvar, future trajectory points
                     If validation, the tuple only returns the destination point
        """
        # if model is in training mode, dest & mask should not be None
        # if model is in validation mode, dest & mask should be None
        assert self.training ^ (dest is None)
        assert self.training ^ (mask is None)

        # encode the past trajectory
        # output -> (batch_size, fdim)
        ftraj = self.encoder_past(x)  #initially x, now put as initial_pos (1), dimension mismatch

        # if training, then use the ground truth destination position to encode them and then concatenate with ftraj to get the latent features
        # using the VAE Reparametrization trick, get the z vector
        # if validation, then sample the latent vector from a normal with mean -> 0 and variance -> sigma
        if not self.training:
            z = torch.Tensor(x.size(0), self.zdim)
            z.normal_(0, self.sigma)

        else:
            # encode the ground truth destination positions to get dest_features
            # output -> (batch_size, fdim)
            dest_features = self.encoder_dest(dest)  #changed to initial_pos, was dest earlier (2)
            features = torch.cat((ftraj, dest_features), dim = 1)
            latent =  self.encoder_latent(features)

            mu = latent[:, 0:self.zdim] # 2-d array
            logvar = latent[:, self.zdim:] # 2-d array

            # sample z from the predicted mean and logvar of the normal distribution
            var = logvar.mul(0.5).exp_()
            eps = torch.DoubleTensor(var.size()).normal_()
            eps = eps.to(device)
            z = eps.mul(var).add_(mu)

        z = z.double().to(device)
        decoder_input = torch.cat((ftraj, z), dim = 1)
        # predicted destination point
        # output -> (batch_size, 2)
        generated_dest = self.decoder(decoder_input)  #=dest (3) self.decoder(decoder_input)

        # prediction of trajectory points only during training only
        # during val/test the best generated_dest is chosen
        if self.training:
            generated_dest_features = self.encoder_dest(generated_dest) #dest, was generated_dest(4)
            prediction_features = torch.cat((ftraj, generated_dest_features, initial_pos), dim = 1)

            for i in range(self.nonlocal_pools):
                prediction_features = self.non_local_social_pooling(prediction_features, mask)

            pred_future = self.predictor(prediction_features)
            return generated_dest, mu, logvar, pred_future

        return generated_dest

    def predict(self, past: torch.Tensor, generated_dest: torch.Tensor, mask: torch.Tensor, initial_pos: torch.Tensor) -> torch.Tensor:
        """This function is used be test engine to predict the best destination
        Similar computation is done in the forward function too but only for train, as during the validation best generated_dest
        is chosen outside the function. 

        Arguments:
            past {torch.Tensor} -- Past Trajectory points -> (batch_size, No. of points * 2)
            generated_dest {torch.Tensor} -- Generated destination by the model -> (batch_size, 2)
            mask {torch.Tensor} -- Social Mask -> (batch_size, batch_size)
            initial_pos {torch.Tensor} -- Initial position of the people -> (batch_size, 2)

        Returns:
            torch.Tensor -- Predicted trajectory points except the end point -> (batch_size, 2*(future_length - 1))
        """
        ftraj = self.encoder_past(past)
        generated_dest_features = self.encoder_dest(generated_dest)
        prediction_features = torch.cat((ftraj, generated_dest_features, initial_pos), dim = 1)

        for i in range(self.nonlocal_pools):
            prediction_features = self.non_local_social_pooling(prediction_features, mask)

        interpolated_future = self.predictor(prediction_features)
        return interpolated_future
