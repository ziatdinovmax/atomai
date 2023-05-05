"""
mGPorch.py
=========
Multi-output Gaussian Process Regression with Hyper-parameter optimization.
Developed in Pytorch environment
Created by Arpan Biswas (email: arpanbiswas52@gmail.com)
"""
from abc import ABC

import atomai as aoi
import gpytorch as gpt
import numpy as np
import torch
from botorch.models import ModelListGP
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, PeriodicKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from torch.optim import Adam


class mGPtrainer:

    def __init__(self):
        pass

    @classmethod
    # Currently implemented to train two objectives, will be extended to train n objectives
    def train_GP(self, train_X, train_Y, mfun="zero", kfun="periodic", lr=0.01):
        """
        Hyperparameter training of >1 GP models using MLE loss function, backpropagation technique and ADAM optimizer
        Args:
            train_X: input data (mxn) matrix where m is the number of training data and n is the dimension of X
            train_Y: output data (mxn) matrix where m is the number of training data and n is the number of outputs
            mfun: Set the mean function of GP, either "zero" for Constant Mean or "linear" for Linear Mean.
            Default:"zero"
            kfun: Set the kernel function of GP, Options: "periodic", "rbf" and "matern.
            Default:"periodic"
            lr: learning rate of ADAM optimizer.
            Default:"0.01"
        :return a list of torch models
        """
        # Gp model fit
        gp_surro1 = SimpleCustomGP(train_X, train_Y[:, 0], mfun, kfun)
        gp_surro2 = SimpleCustomGP(train_X, train_Y[:, 1], mfun, kfun)
        gp_surro = ModelListGP(gp_surro1, gp_surro2)
        gp_surro = gp_surro.double()
        gp_surro.likelihood.likelihoods[0].noise_covar.register_constraint("raw_noise", GreaterThan(1e-1))
        gp_surro.likelihood.likelihoods[1].noise_covar.register_constraint("raw_noise", GreaterThan(1e-1))
        mll1 = ExactMarginalLogLikelihood(gp_surro.likelihood.likelihoods[0], gp_surro.models[0])
        mll2 = ExactMarginalLogLikelihood(gp_surro.likelihood.likelihoods[1], gp_surro.models[1])
        mll1 = mll1.to(train_X)
        mll2 = mll2.to(train_X)
        gp_surro.models[0].train()
        gp_surro.likelihood.likelihoods[0].train()
        gp_surro.models[1].train()
        gp_surro.likelihood.likelihoods[1].train()
        optimizer1 = Adam([{'params': gp_surro.models[0].parameters()}], lr=lr)
        optimizer2 = Adam([{'params': gp_surro.models[1].parameters()}], lr=lr)

        NUM_EPOCHS = 150

        for epoch in range(NUM_EPOCHS):
            # clear gradients
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            # forward pass through the model to obtain the output MultivariateNormal
            output1 = gp_surro.models[0](train_X)
            output2 = gp_surro.models[1](train_X)
            # Compute negative marginal log likelihood
            loss1 = - mll1(output1, gp_surro.models[0].train_targets)
            loss2 = - mll2(output2, gp_surro.models[1].train_targets)
            # back prop gradients
            loss1.backward(retain_graph=True)
            # back prop gradients
            loss2.backward(retain_graph=True)
            # print last iterations
            if (epoch + 1) > NUM_EPOCHS:  # Stopping the print for now
                print("GP Model trained:")
                print("Iteration:" + str(epoch + 1))
                print("Loss for ob1:" + str(loss1.item()))
                print("Loss for ob2:" + str(loss2.item()))
                print("noise for obj1:" + str(gp_surro.likelihood.likelihoods[0].noise.item()))
                print("noise for obj2:" + str(gp_surro.likelihood.likelihoods[1].noise.item()))

            optimizer1.step()
            optimizer2.step()

        gp_surro.models[0].eval()
        gp_surro.likelihood.likelihoods[0].eval()
        gp_surro.models[1].eval()
        gp_surro.likelihood.likelihoods[1].eval()
        return gp_surro

    @classmethod
    def train_dKLgp(self, train_X, train_Y):
        """
        Fitting deep kernel learning GP, developed in ATOMAI package
        Args:
            train_X: input data (mxn) matrix where m is the number of training data and n is the dimension of X
            train_Y: output data (mxn) matrix where m is the number of training data and n is the number of outputs
        :return a list of torch models
        """
        data_dim = train_X.shape[-1]
        dklgp_surro1 = aoi.models.dklGPR(data_dim, embedim=2, precision="single")
        dklgp_surro1.fit(train_X, train_Y[:, 0], training_cycles=200)
        dklgp_surro2 = aoi.models.dklGPR(data_dim, embedim=2, precision="single")
        dklgp_surro2.fit(train_X, train_Y[:, 1], training_cycles=200)

        gp_surro = [dklgp_surro1, dklgp_surro2]

        return gp_surro

    @classmethod
    def cal_posteriorGP(self, model, test_x, n_obj=2):
        """
        # Calculate posterior distribution, given the model
        Args:
            model: a list of torch models
            test_x: input data (mxn) matrix where m is the number of test data and n is the dimension of X
            n_obj: number of outputs
            Default: 2

        :return: Posterior prediction of test data
            y_pred_means: (mxn) matrix of posterior mean where m is the number of test data and and n is the number of
            outputs
            y_pred_vars: (mxn) matrix of posterior variance where m is the number of test data and and n is the number of
            outputs
        """
        self.model = model
        y_pred_means = torch.empty(len(test_x), n_obj)
        y_pred_vars = torch.empty(len(test_x), n_obj)
        t_X = torch.empty(1, test_x.shape[1])
        # print(t_X)
        for t in range(0, len(test_x)):
            with torch.no_grad(), gpt.settings.max_lanczos_quadrature_iterations(32), \
                    gpt.settings.fast_computations(covar_root_decomposition=False, log_prob=False,
                                                   solves=True), \
                    gpt.settings.max_cg_iterations(100), \
                    gpt.settings.max_preconditioner_size(80), \
                    gpt.settings.num_trace_samples(128):
                t_X[0, :] = test_x[t, :]
                # t_X[:, 1] = test_X[t, 1]
                # t_X = test_X.double()
                y1_pred_surro, y2_pred_surro = self.model.models[0].posterior(t_X), self.model.models[1].posterior(t_X)
                y_pred_means[t, 0] = y1_pred_surro.mean
                y_pred_vars[t, 0] = y1_pred_surro.variance
                y_pred_means[t, 1] = y2_pred_surro.mean
                y_pred_vars[t, 1] = y2_pred_surro.variance

        return y_pred_means, y_pred_vars

    @classmethod
    def cal_posteriordKLGP(self, model, test_x, n_obj=2):
        """
        # Calculate posterior distribution, given the model
        Args:
            model: a list of torch models
            test_x: input data (mxn) matrix where m is the number of test data and n is the dimension of X
            n_obj: number of outputs
            Default: 2

        :return: Posterior prediction of test data
            y_pred_means: (mxn) matrix of posterior mean where m is the number of test data and and n is the number of
            outputs
            y_pred_vars: (mxn) matrix of posterior variance where m is the number of test data and and n is the number of
            outputs
        """
        self.model = model
        y_pred_means = np.empty((len(test_x), n_obj))
        y_pred_vars = np.empty((len(test_x), n_obj))
        t_X = test_x.detach().numpy()
        for i in range(0, n_obj):
            mean, var = self.model[i].predict(t_X, batch_size=len(t_X))
            y_pred_means[:, i] = mean
            y_pred_vars[:, i] = var

        y_pred_means = torch.from_numpy(y_pred_means)
        y_pred_vars = torch.from_numpy(y_pred_vars)
        return y_pred_means, y_pred_vars


class SimpleCustomGP(ExactGP, GPyTorchModel, ABC):
    _num_outputs = 1  # to inform GPyTorchModel API
    """
    Gaussian process regression (GPR)
    Args:
        train_X: input data (mxn) matrix where m is the number of training data and n is the dimension of X
        train_Y: output data (mx1) array where m is the number of training data
        mfun: Set the mean function of GP, either "zero" for Constant Mean or "linear" for Linear Mean.
        Default:"zero"
        kfun: Set the kernel function of GP, Options: "periodic", "rbf" and "matern".
        Default:"periodic"
    :returns torch model
    """

    def __init__(self, train_X, train_Y, mfun="zero", kfun="periodic"):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mfun = mfun
        self.kfun = kfun
        if self.mfun == "zero":
            self.mean_module = ConstantMean()
        elif self.mfun == "linear":
            self.mean_module = LinearMean(train_X.shape[-1])
        else:
            print("ERROR: Provide correct GP mean functions: 'zero' or 'linear'")

        if self.kfun == "rbf":
            base_kernel = RBFKernel(ard_num_dims=train_X.shape[-1])
        elif self.kfun == "periodic":
            base_kernel = PeriodicKernel(ard_num_dims=train_X.shape[-1])
        elif self.kfun == "matern":
            base_kernel = MaternKernel(nu=2.5, ard_num_dims=train_X.shape[-1])
        else:
            print("ERROR: Provide correct GP kernel functions: 'periodic', 'rbf' and 'matern'")

        self.covar_module = ScaleKernel(
            base_kernel=base_kernel,
        )
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
