"""
mAForch.py
=========
List of acquistion functions to handle multi-objective Bayesian optimization
Developed in Pytorch environment
Created by Arpan Biswas (email: arpanbiswas52@gmail.com)
"""
import torch
from botorch.acquisition import GenericMCObjective, qExpectedImprovement
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf_discrete, optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.sampling import sample_simplex
from scipy.stats import norm


class mAcqFunc:

    def __init__(self):
        pass

    @classmethod
    def wtbei(self, y_means, y_vars, train_Y, n_obj=2, BATCH_SIZE=1):
        """
        # Weighted Tchebycheff-Expected Improvement acquisition function
        Args:
            y_means: (mxn) matrix of posterior mean where m is the number of test data and and n is the number of
            outputs
            y_vars: (mxn) matrix of posterior variance where m is the number of test data and and n is the number of
            outputs
            train_Y: output data (mxn) matrix where m is the number of training data and n is the number of outputs
            n_obj: number of outputs
            Default:"2"
            BATCH_SIZE: # of new sampled data to be evaluated in next BO iteration. The parallel sampling for this
            acquisition function will be implemented in future. Currently should be set only 1

        :return:
            new_x: new sample as indices, 1D tensor array where D = batch size, of input data matrix to be evaluated next.
            acq_val: a scalar maximum acquisition function value
        """
        weights = sample_simplex(n_obj).squeeze().cpu()
        weights = weights.detach().numpy()
        u = torch.tensor((10, 10))
        y_weighted_train = torch.zeros_like(train_Y)
        y_weighted_pred = torch.zeros_like(y_means)
        y_weighted_pred_var = torch.zeros_like(y_vars)
        for i in range(0, n_obj):
            y_weighted_train[:, i] = weights[i] * train_Y[:, i] - u[i]
            y_weighted_pred[:, i] = weights[i] * y_means[:, i] - u[i]
            y_weighted_pred_var[:, i] = weights[i] ** 2 * y_vars[:, i]

        y_multi_train = -1 * torch.max(y_weighted_train, axis=1).values  # Transforming to maximization prob
        y_multi_pred = -1 * torch.max(y_weighted_pred, axis=1).values  # Transforming to maximization prob
        y_multi_pred_var = torch.max(y_weighted_pred_var, axis=1).values

        y_std = torch.sqrt(y_multi_pred_var)
        fmax = y_multi_train.max()
        best_value = fmax
        EI_val = torch.zeros(len(y_vars))
        Z = torch.zeros(len(y_vars))
        eta = 0.01

        for i in range(0, len(y_std)):
            if y_std[i] <= 0:
                EI_val[i] = 0
            else:
                Z[i] = (y_multi_pred[i] - best_value - eta) / y_std[i]
                EI_val[i] = (y_multi_pred[i] - best_value - eta) * norm.cdf(Z[i]) + y_std[i] * norm.pdf(Z[i])

        acq_val = torch.max(EI_val)
        new_x = [k for k, j in enumerate(EI_val) if j == acq_val]
        return new_x, acq_val

    @classmethod
    def qehvi(self, model, test_x, train_Y, bound=None, rf=None, n_obj=2, BATCH_SIZE=1, optim='discrete'):
        """
        # qExpected Hypervolume Improvement acquisition function
        Args:
            model: a list of torch models
            test_x: input data (mxn) matrix where m is the number of test data and n is the dimension of X
            train_Y: output data (mxn) matrix where m is the number of training data and n is the number of outputs
            bound: the bound of the parameter space to be optimized.
            Default: None, if optim is discrete. Need to provide a torch tensor if optim is continuous
            Eg for 2D case: bound = torch.tensor([[-3.0, -2.0], [3.0, 2.0]]) or torch.tensor([[0.0] * 2, [1.0] * 2])
            BATCH_SIZE: # of new sampled data to be evaluated in next BO iteration.
            Default: 1
            optim: Methods to optimize acquisition function, "discrete" or "continuous"
            Default: "discrete"

        :return:
            new sample as indices, 1D tensor array where D = batch size, of input data matrix to be evaluated next if
            optim = discrete.
            If optim is continuous, returns new sample as input X as tensor matrix (mxn) where m is the batch size and
            m is the dimension of X. In this case, the new sample may or may not present in grid matrix test_x.
            Use the continuous option only if the input are feasible to measure at any values within two adjacent
            discrete points in test data
            acq_val: a scalar maximum acquisition function value
        """
        if rf is None:
            self.rf = torch.tensor([0.0] * n_obj) - 0.001  # rf is the worst solution of the objectives.
        else:
            self.rf = rf

        # partition non-dominated space into disjoint rectangles
        sampler = SobolQMCNormalSampler(500, seed=0, resample=False)
        partitioning = NondominatedPartitioning(ref_point=self.rf, Y=train_Y)
        acq_func = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.rf.tolist(),  # use known reference point
            partitioning=partitioning,
            sampler=sampler,
        )
        self.acq_func = acq_func
        self.test_x = test_x
        self.BATCH_SIZE = BATCH_SIZE
        self.bound = bound
        # discrete optimization
        new_cand, value = self.optimize_acq(optim)
        acq_val = torch.mean(value)

        if optim == "discrete":
            _, new_x = torch.min(torch.sum(torch.absolute(test_x - new_cand), axis=1), axis=0)
        else:
            new_x = new_cand

        return new_x, acq_val

    @classmethod
    def qparego(self, model, test_x, train_Y, bound=None, BATCH_SIZE=1, optim='discrete'):
        """
        # qPareto Efficient Global optimization acquisition function
        Args:
            model: a list of torch models
            test_x: input data (mxn) matrix where m is the number of test data and n is the dimension of X
            train_Y: output data (mxn) matrix where m is the number of training data and n is the number of outputs
            bound: the bound of the parameter space to be optimized.
            Default: None, if optim is discrete. Need to provide a torch tensor if optim is continuous
            Eg for 2D case: bound = torch.tensor([[-3.0, -2.0], [3.0, 2.0]]) or torch.tensor([[0.0] * 2, [1.0] * 2])
            BATCH_SIZE: # of new sampled data to be evaluated in next BO iteration.
            Default: 1
            optim: Methods to optimize acquisition function, "discrete" or "continuous"
            Default: "discrete"

        :return:
            new sample as indices, 1D tensor array where D = batch size, of input data matrix to be evaluated next if
            optim = discrete.
            If optim is continuous, returns new sample as input X as tensor matrix (mxn) where m is the batch size and
            m is the dimension of X. In this case, the new sample may or may not present in grid matrix test_x.
            Use the continuous option only if the input are feasible to measure at any values within two adjacent
            discrete points in test data
            acq_val: a scalar maximum acquisition function value
        """
        sampler = SobolQMCNormalSampler(500, seed=0, resample=False)
        acq_func_list = []
        for _ in range(BATCH_SIZE):
            n_obj = train_Y.dim()
            weights = sample_simplex(n_obj).squeeze()
            objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights.cpu(), Y=train_Y.cpu()))

            acq_func = qExpectedImprovement(  # pyre-ignore: [28]
                model=model,
                objective=objective,
                best_f=objective(train_Y).max().cpu(),
                sampler=sampler,
            )
            acq_func_list.append(acq_func)

        self.acq_func = acq_func
        self.test_x = test_x
        self.BATCH_SIZE = BATCH_SIZE
        self.bound = bound
        # discrete optimization
        new_cand, value = self.optimize_acq(optim)

        acq_val = torch.mean(value)

        if optim == "discrete":
            _, new_x = torch.min(torch.sum(torch.absolute(test_x - new_cand), axis=1), axis=0)
        else:
            new_x = new_cand
        return new_x, acq_val

    @classmethod
    def optimize_acq(self, args=None):
        """
        # optimize acquisition function: Continuous or discrete optimization
        Args:
            None
        :return:
            new_cand: new input X data as (mxn) matrix where m is the batch size and n is the dimension of X
            acq_val: a scalar or torch tensor of batch size for maximum acquisition function value
        """
        acq_func = self.acq_func
        BATCH_SIZE = self.BATCH_SIZE
        test_x = self.test_x
        bound = self.bound
        self.optim = args

        if self.optim == 'discrete':
            # discrete optimization
            new_cand, value = optimize_acqf_discrete(
                acq_function=acq_func,
                q=BATCH_SIZE,
                choices=test_x
            )

        else:  # continuous optimization
            new_cand, value = optimize_acqf(
                acq_function=acq_func,
                # bounds=torch.tensor([[0.0] * 2, [1.0] * 2]),
                bounds=bound,
                q=BATCH_SIZE,
                num_restarts=20,
                raw_samples=1000,  # used for initialization heuristic
                options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
                sequential=True,
            )
        return new_cand, value
