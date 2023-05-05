import sys
import gdown
import torch
from torchvision import datasets, transforms
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pickle
import numpy as np
import random

# Import GP and BoTorch functions
import gpytorch as gpt
from botorch.models import SingleTaskGP
#from botorch.models import gpytorch
from botorch.fit import fit_gpytorch_model
from botorch.models.gpytorch import GPyTorchModel
from botorch.utils import standardize
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, PeriodicKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition import ExpectedImprovement
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.constraints import GreaterThan

from gpytorch.models import ExactGP
from mpl_toolkits.axes_grid1 import make_axes_locatable

from torch.optim import SGD
from torch.optim import Adam
from scipy.stats import norm
import time

class SimpleCustomGP(ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        #self.mean_module = LinearMean(train_X.shape[-1])
        self.covar_module = ScaleKernel(
            base_kernel=MaternKernel(nu=2.5, ard_num_dims=train_X.shape[-1]),
            #base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1]),
            #base_kernel=PeriodicKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def optimize_hyperparam_trainGP(train_X, train_Y):
    # Gp model fit

    gp_surro = SimpleCustomGP(train_X, train_Y)
    gp_surro = gp_surro.double()
    gp_surro.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-1))
    mll1 = ExactMarginalLogLikelihood(gp_surro.likelihood, gp_surro)
    # fit_gpytorch_model(mll)
    mll1 = mll1.to(train_X)
    gp_surro.train()
    gp_surro.likelihood.train()
    ## Here we use Adam optimizer
    optimizer1 = Adam([{'params': gp_surro.parameters()}], lr=0.01)
    # optimizer1 = SGD([{'params': gp_surro.parameters()}], lr=0.0001)

    NUM_EPOCHS = 150

    for epoch in range(NUM_EPOCHS):
        # clear gradients
        optimizer1.zero_grad()
        # forward pass through the model to obtain the output MultivariateNormal
        output1 = gp_surro(train_X)
        # Compute negative marginal log likelihood
        loss1 = - mll1(output1, gp_surro.train_targets)
        # back prop gradients
        loss1.backward(retain_graph=True)
        # print last iterations
        if (epoch + 1) > NUM_EPOCHS:  # Stopping the print for now
            print("GP Model trained:")
            print("Iteration:" + str(epoch + 1))
            print("Loss:" + str(loss1.item()))
            # print("Length Scale:" +str(gp_PZO.covar_module.base_kernel.lengthscale.item()))
            print("noise:" + str(gp_surro.likelihood.noise.item()))

        optimizer1.step()

    gp_surro.eval()
    gp_surro.likelihood.eval()
    return gp_surro


def cal_posterior(gp_surro, test_X):
    y_pred_means = torch.empty(len(test_X))
    y_pred_vars = torch.empty(len(test_X))
    t_X = torch.empty(1)
    for t in range(0, len(test_X)):
        with torch.no_grad(), gpt.settings.max_lanczos_quadrature_iterations(32), \
                gpt.settings.fast_computations(covar_root_decomposition=False, log_prob=False,
                                               solves=True), \
                gpt.settings.max_cg_iterations(100), \
                gpt.settings.max_preconditioner_size(80), \
                gpt.settings.num_trace_samples(128):
            t_X[0] = test_X[t]
            # t_X = test_X.double()
            y_pred_surro = gp_surro.posterior(t_X)
            y_pred_means[t] = y_pred_surro.mean
            y_pred_vars[t] = y_pred_surro.variance

    return y_pred_means, y_pred_vars


def acqmanEI(y_means, y_vars, train_Y):
    y_means = y_means.detach().numpy()
    y_vars = y_vars.detach().numpy()
    y_std = np.sqrt(y_vars)
    fmax = train_Y.max()
    fmax = fmax.detach().numpy()
    best_value = fmax
    EI_val = np.zeros(len(y_vars))
    Z = np.zeros(len(y_vars))
    eta = 0.01

    for i in range(0, len(y_std)):
        if (y_std[i] <= 0):
            EI_val[i] = 0
        else:
            Z[i] = (y_means[i] - best_value - eta) / y_std[i]
            EI_val[i] = (y_means[i] - best_value - eta) * norm.cdf(Z[i]) + y_std[i] * norm.pdf(Z[i])

    # EI_val[ieval] = -1
    acq_val = np.max(EI_val)
    acq_cand = [k for k, j in enumerate(EI_val) if j == acq_val]
    return acq_cand, acq_val, EI_val


def batch_LP(y_means, y_vars, train_Y, weight):
    y_means = y_means.detach().numpy()
    y_vars = y_vars.detach().numpy()
    y_std = np.sqrt(y_vars)
    fmax = train_Y.max()
    fmax = fmax.detach().numpy()
    best_value = fmax
    EI_val = np.zeros(len(y_vars))
    Z = np.zeros(len(y_vars))
    eta = 0.01

    for i in range(0, len(y_std)):
        if (y_std[i] <= 0):
            EI_val[i] = 0
        else:
            Z[i] = (y_means[i] - best_value - eta) / y_std[i]
            EI_val[i] = (y_means[i] - best_value - eta) * norm.cdf(Z[i]) + y_std[i] * norm.pdf(Z[i])

    # Normalize
    EI_val_norm = (EI_val - EI_val.min()) / (EI_val.max() - EI_val.min())
    y_vars_norm = (y_vars - y_vars.min()) / (y_vars.max() - y_vars.min())

    # Local Penalization with EI acq func
    EI_LP_val = ((1 - weight) * EI_val_norm) + (weight * y_vars_norm)

    # EI_val[ieval] = -1
    acq_val = np.max(EI_LP_val)
    acq_cand = [k for k, j in enumerate(EI_LP_val) if j == acq_val]
    return acq_cand, acq_val, EI_LP_val


def batch_DE(train_data, test_data, y_means, y_vars, train_Y, weight, b_n):
    y_means = y_means.detach().numpy()
    y_vars = y_vars.detach().numpy()
    fmax = train_Y.max()
    fmax = fmax.detach().numpy()
    best_value = fmax
    train_data = train_data.detach().numpy()
    test_data = test_data.detach().numpy()
    batch_data = train_data[-b_n:]
    dist_X = np.zeros((len(test_data), len(batch_data)))
    min_dist_X = np.zeros(len(test_data))
    dist_Y = np.zeros(len(y_means))

    for i in range(0, len(test_data)):
        x = test_X[i]  # Select candidate
        for j in range(0, len(batch_data)):  # Explored data
            dist_X[i, j] = (x - batch_data[j]) ** 2
            # print(x, train_data[j], dist_X[i, j])
        min_dist_X[i] = np.min(dist_X[i, :])

    for i in range(0, len(y_means)):
        y = y_means[i]  # Select candidate
        dist_Y[i] = y - best_value

    # Normalize
    min_dist_X_norm = (min_dist_X - np.nanmax(min_dist_X)) / (np.nanmax(min_dist_X) - np.nanmin(min_dist_X))
    dist_Y_norm = (dist_Y - np.nanmax(dist_Y)) / (np.nanmax(dist_Y) - np.nanmin(dist_Y))

    # Distance exploration with value improvment
    # DE_VI_val = ((1-weight)*dist_Y_norm) + (weight*dist_X_norm)
    DE_VI_val = min_dist_X

    # EI_val[ieval] = -1
    # DE_VI_val = DE_VI_val.detach().numpy()
    acq_val = np.max(DE_VI_val)
    acq_cand = [k for k, j in enumerate(DE_VI_val) if j == acq_val]
    print(acq_val)
    return acq_cand, acq_val, DE_VI_val


def plot_results(train_X, train_Y, test_X, y_pred_means, y_pred_vars, img_space, i):
    pen = 10 ** 0
    y_pred_sd = np.sqrt(y_pred_vars)
    # Objective map
    plt.plot(test_X, img_space, c="r", label="ground truth")
    plt.plot(test_X, y_pred_means, c="b", label="mean")
    plt.fill_between(
        test_X.flatten(),
        y_pred_means - (1.96 * y_pred_sd),
        y_pred_means + (1.96 * y_pred_sd),
        alpha=0.3,
        label="95% CI",
    )
    plt.scatter(train_X[-1], train_Y[-1], c="g", marker="o", label="new obs", s=50)
    plt.scatter(train_X[:-1], train_Y[:-1], c="k", marker="X", label="prior obs", s=20)

    # plt.plot(test_X, y_pred_means+(1.96*y_pred_sd), "b--", label="upper 95% CI")
    # plt.plot(test_X, y_pred_means-(1.96*y_pred_sd), "b--", label="upper 95% CI")
    plt.legend(loc="best", fontsize=10)
    plt.xlabel("X")
    plt.ylabel("obj")
    plt.title("Evaluation: " + str(i))
    plt.show()


def plot_results_batch(train_X, train_Y, test_X, y_pred_means, y_pred_vars, img_space, i, b):
    pen = 10 ** 0
    y_pred_sd = np.sqrt(y_pred_vars)
    # Objective map
    plt.plot(test_X, img_space, c="r", label="ground truth")
    plt.plot(test_X, y_pred_means, c="b", label="mean")
    plt.fill_between(
        test_X.flatten(),
        y_pred_means - (1.96 * y_pred_sd),
        y_pred_means + (1.96 * y_pred_sd),
        alpha=0.3,
        label="95% CI",
    )
    plt.scatter(train_X[-batch:], train_Y[-batch:], c="g", marker="o", label="batch obs", s=50)
    plt.scatter(train_X[:-batch], train_Y[:-batch], c="k", marker="X", label="prior obs", s=20)

    # plt.plot(test_X, y_pred_means+(1.96*y_pred_sd), "b--", label="upper 95% CI")
    # plt.plot(test_X, y_pred_means-(1.96*y_pred_sd), "b--", label="upper 95% CI")
    plt.legend(loc="best", fontsize=10)
    plt.xlabel("X")
    plt.ylabel("obj")
    plt.title("Evaluation: " + str(i))
    plt.show()

def objective(x):
    # a modification of https://www.sfu.ca/~ssurjano/forretal08.html
    y = -((x + 1) ** 2) * torch.sin(2 * x + 2) / 5 + 1 + x / 3
    return y


lb = -10
ub = 10
grid= 1000
X_feas = torch.linspace(lb, ub, grid)

y_full = objective(X_feas)
#print(ys.shape)

plt.plot(X_feas, y_full, c="r")
plt.show()

# initial train location
np.random.seed(10) #10
num_start = 10
idx = np.random.randint(0, len(X_feas), num_start)
train_X = X_feas[idx]

train_Y = objective(train_X)

plt.plot(X_feas, y_full, c="r", label = "ground truth")
plt.scatter(train_X, train_Y, c="k", marker="X", label="obs", s=20)
plt.legend(loc="best", fontsize =15)
plt.show()

test_X = X_feas
N = 10
e = int(0.7 * N)
batch = 5
wts = np.linspace(0.9, 0.1, e)
# GP fit
gp_surro = optimize_hyperparam_trainGP(train_X, train_Y)
for i in range(1, N + 1):
    print("step: ", i)
    # Calculate posterior for analysis for intermidiate iterations

    y_pred_means_full, y_pred_vars_full = cal_posterior(gp_surro, X_feas)
    y_pred_means, y_pred_vars = cal_posterior(gp_surro, test_X)
    # weight for local penalization-- the weight has a cooldown trajectory as the iteration progress
    if i > e:
        w = wts[-1]
    else:
        w = wts[i - 1]

    for j in range(0, batch):
        print("batch: ", j + 1)

        # calculate acquisition function
        if j == 0:
            acq_cand, acq_val, EI_val = acqmanEI(y_pred_means, y_pred_vars,
                                                 train_Y)  # 1st cand in the batch is same as single BO sampling
        else:
            acq_cand, acq_val, EI_val = batch_DE(train_X, test_X, y_pred_means, y_pred_vars, train_Y, w,
                                                 j)  # Batch sampling with KB(EI)+LP acq function
        val = acq_val
        ind = np.random.choice(acq_cand)  # When multiple points have same acq values
        idx = np.hstack((idx, ind))

        ## Find next point which maximizes the learning through exploration-exploitation
        nextX = test_X[ind]
        train_X = torch.hstack((train_X, nextX))

        # Remove evaluated samples from test and predicted data (Avoid repeated sampling)
        test_X = torch.cat([test_X[:ind], test_X[ind + 1:]])
        y_pred_means = torch.cat([y_pred_means[:ind], y_pred_means[ind + 1:]])
        y_pred_vars = torch.cat([y_pred_vars[:ind], y_pred_vars[ind + 1:]])

    # Batch evaluation
    next_Y = objective(train_X[-batch:])
    train_Y = torch.hstack((train_Y, next_Y))

    # Plot each iterations
    plot_results_batch(train_X, train_Y, X_feas, y_pred_means_full, y_pred_vars_full, y_full, i, batch)

    # Updating GP after batch evaluation
    gp_surro = optimize_hyperparam_trainGP(train_X, train_Y)

    ## Final posterior prediction after all the sampling done

if (i == N):
    print("\n\n#####----Max. sampling reached, model stopped----#####")

# Optimal GP learning
gp_opt = gp_surro
# Posterior calculation with converged GP model
y_pred_means_full, y_pred_vars_full = cal_posterior(gp_opt, X_feas)
# Plotting functions to check final iteration
plot_results_batch(train_X, train_Y, X_feas, y_pred_means_full, y_pred_vars_full, y_full, i, batch)