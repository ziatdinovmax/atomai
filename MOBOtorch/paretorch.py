"""
paretorch.py
=========
Compute Pareto solutions and hypervolume functions, given the pareto in Multi-objective optimization settings
Developed in Pytorch environment
Created by Arpan Biswas (email: arpanbiswas52@gmail.com)
"""
import torch
from botorch.utils.multi_objective import is_non_dominated, Hypervolume


class Pareto:

    def __init__(self):
        pass

    @classmethod
    def estimate(self, train_samples, train_Y, hvs, rf=None, n_obj=2):
        """
        estimate pareto front and hypervolume over the objective space
        Args:
            train_samples: sample grid points (mxn) matrix where m is the number of training data and n is the number of outputs
            train_Y: output data (mxn) matrix where m is the number of training data and n is the number of outputs
            hvs: a list of scalar values
            rf:a torch tensor of worst possible solution for each objectives in the multi-objective setting
            n_obj: number of outputs
            Default: 2
        :return: Pareto front and the respective hypervolume
            pareto_y: pareto front over objective space
            pareto_s: pareto front over input grid space
            hvs: a list of scalar hypervolume values
        """

        # compute pareto front
        pareto_y, pareto_s = self.select_Pareto(train_samples, train_Y)

        # compute hypervolume
        if rf is None:
            self.rf = torch.tensor([0.0] * n_obj) - 0.001  # rf is the worst solution of the objectives.
        else:
            self.rf = rf
        volume = self.calHV(pareto_y)
        hvs.append(volume)

        return pareto_y, pareto_s, hvs

    @classmethod
    def select_Pareto(self, train_samples, train_Y):
        """
        compute Pareto front
        Args:
            train_samples: sample grid points (mxn) matrix where m is the number of training data and n is the number of outputs
            train_Y: output data (mxn) matrix where m is the number of training data and n is the number of outputs
        :return: Pareto front and the respective hypervolume
            pareto_y: pareto front over objective space
            pareto_s: pareto front over input grid space
        """
        pareto_mask = is_non_dominated(train_Y)
        pareto_s = train_samples[pareto_mask]
        pareto_y = train_Y[pareto_mask]
        return pareto_y, pareto_s

    @classmethod
    def calHV(self, pareto):
        """
        compute Pareto front
        Args:
            rf:a torch tensor of worst possible solution for each objectives in the multi-objective setting
            pareto: pareto front in torch matrix over the objective space
        :return a scalar value
        """
        hv = Hypervolume(ref_point=self.rf)
        volume = hv.compute(pareto)
        return volume
