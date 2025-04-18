import gpytorch
import torch
import os
import tqdm
import urllib.request
import warnings
import numpy as np
import time
from math import floor
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from scipy.cluster.vq import kmeans2
from scipy.stats import norm
from matplotlib import pyplot as plt
from gpytorch.mlls import DeepPredictiveLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.variational import VariationalStrategy, BatchDecoupledVariationalStrategy
from gpytorch.variational import MeanFieldVariationalDistribution
from gpytorch.models.deep_gps.dspp import DSPPLayer, DSPP
import gpytorch.settings as settings
import scipy.io as scio

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class DSPPHiddenLayer(DSPPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=300, inducing_points=None, mean_type='constant', Q=8):
        if inducing_points is not None and output_dims is not None and inducing_points.dim() == 2:
            # The inducing points were passed in, but the shape doesn't match the number of GPs in this layer.
            # Let's assume we wanted to use the same inducing point initialization for each GP in the layer,
            # and expand the inducing points to match this.
            inducing_points = inducing_points.unsqueeze(0).expand((output_dims,) + inducing_points.shape)
            inducing_points = inducing_points.clone() + 0.01 * torch.randn_like(inducing_points)
        if inducing_points is None:
            # No inducing points were specified, let's just initialize them randomly.
            if output_dims is None:
                # An output_dims of None implies there is only one GP in this layer
                # (e.g., the last layer for univariate regression).
                inducing_points = torch.randn(num_inducing, input_dims)
            else:
                inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        else:
            # Get the number of inducing points from the ones passed in.
            num_inducing = inducing_points.size(-2)

        # Let's use mean field / diagonal covariance structure.
        variational_distribution = MeanFieldVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=torch.Size([output_dims]) if output_dims is not None else torch.Size([])
        )

        # Standard variational inference.
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        batch_shape = torch.Size([]) if output_dims is None else torch.Size([output_dims])

        super(DSPPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims, Q)

        if mean_type == 'constant':
            # We'll use a constant mean for the final output layer.
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        elif mean_type == 'linear':
            # As in Salimbeni et al. 2017, we find that using a linear mean for the hidden layer improves performance.
            self.mean_module = LinearMean(input_dims, batch_shape=batch_shape)

        self.covar_module = ScaleKernel(MaternKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
                                        batch_shape=batch_shape, ard_num_dims=None)

    def forward(self, x, mean_input=None, **kwargs):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class TwoLayerDSPP(DSPP):
    def __init__(self, train_x_shape, inducing_points, num_inducing, hidden_dim=3, Q=3):
        hidden_layer = DSPPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=hidden_dim,
            mean_type='linear',
            inducing_points=inducing_points,
            Q=Q,
        )
        last_layer = DSPPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            mean_type='constant',
            inducing_points=None,
            num_inducing=num_inducing,
            Q=Q,
        )

        likelihood = GaussianLikelihood()

        super().__init__(Q)
        self.likelihood = likelihood
        self.last_layer = last_layer
        self.hidden_layer = hidden_layer

    def forward(self, inputs, **kwargs):
        hidden_rep1 = self.hidden_layer(inputs, **kwargs)
        output = self.last_layer(hidden_rep1, **kwargs)
        return output

    def predict(self, loader):
        with settings.fast_computations(log_prob=False, solves=False), torch.no_grad():
            mus, variances, lls = [], [], []
            for x_batch, y_batch in loader:
                preds = self.likelihood(self(x_batch, mean_input=x_batch))
                mus.append(preds.mean.cpu())
                variances.append(preds.variance.cpu())

                # Compute test log probability. The output of a DSPP is a weighted mixture of Q Gaussians,
                # with the Q weights specified by self.quad_weight_grid. The below code computes the log probability of each
                # test point under this mixture.

                # Step 1: Get log marginal for each Gaussian in the output mixture.
                base_batch_ll = self.likelihood.log_marginal(y_batch, self(x_batch))

                # Step 2: Weight each log marginal by its quadrature weight in log space.
                deep_batch_ll = self.quad_weights.unsqueeze(-1) + base_batch_ll

                # Step 3: Take logsumexp over the mixture dimension, getting test log prob for each datapoint in the batch.
                batch_log_prob = deep_batch_ll.logsumexp(dim=0)
                lls.append(batch_log_prob.cpu())

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)


def load_GPmodel(Model_Path, data_record):
    batch_size = 500  # Size of minibatch
    milestones = [20, 150, 300]  # Epochs at which we will lower the learning rate by a factor of 0.1
    num_inducing_pts = 300  # Number of inducing points in each hidden layer 97.8, 8.0
    num_epochs = 400  # Number of epochs to train for
    initial_lr = 0.01  # Initial learning rate
    hidden_dim = 6  # Number of GPs (i.e., the width) in the hidden layer. 6:97.7, 8.8
    num_quadrature_sites = 8  # Number of quadrature sites (see paper for a description of this. 5-10 generally works well). S

    PATHcwd = os.getcwd()
    warnings.filterwarnings("ignore")
    data = scio.loadmat(data_record)
    data = data['Data']
    TSI = data[:, -1].reshape(-1, 1)
    TSI = torch.from_numpy(TSI).float()
    data = torch.from_numpy(data[:, :-1]).float()

    X = data.detach().clone()
    X_min = X.min(0)[0]
    X = X - X_min
    X_max = X.max(0)[0]
    X = 2.0 * (X / X_max) - 1.0
    y = TSI.detach().clone()
    y = y.min(1)[0]
    y_mean = y.mean()
    y -= y.mean()
    y_std = y.std()
    y /= y.std()

    shuffled_indices = torch.randperm(X.size(0))
    X = X[shuffled_indices, :]
    y = y[shuffled_indices]

    train_n = int(floor(0.8 * X.size(0)))

    train_x = X[:train_n, :].contiguous()
    train_y = y[:train_n].contiguous()
    test_x = X[train_n:, :].contiguous()
    test_y = y[train_n:].contiguous()

    if torch.cuda.is_available():
        train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

   # print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    # Use k-means to initialize inducing points (only helpful for the first layer)
    inducing_points = (train_x[torch.randperm(min(1000 * 100, train_n))[0:num_inducing_pts], :])
    inducing_points = inducing_points.clone().data.cpu().numpy()
    inducing_points = torch.tensor(kmeans2(train_x.data.cpu().numpy(), inducing_points, minit='matrix')[0])

    if torch.cuda.is_available():
        inducing_points = inducing_points.cuda()

    model = TwoLayerDSPP(
        train_x.shape,
        inducing_points,
        num_inducing=num_inducing_pts,
        hidden_dim=hidden_dim,
        Q=num_quadrature_sites
    )

    state_dict = torch.load(Model_Path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    GPmodel = {}
    GPmodel['model'] = model
    GPmodel['X_max'] = X_max / 100
    GPmodel['X_min'] = X_min / 100
    GPmodel['y_mean'] = y_mean
    GPmodel['y_std'] = y_std

    data = data.numpy()
    TSI = TSI.numpy()
    TSI = TSI.min(1)

    return GPmodel, data, TSI
