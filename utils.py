"""
utils.py
=========
Miscellaneous functions like plot functions for MOO problems
Developed in Pytorch environment
Created by Arpan Biswas (email: arpanbiswas52@gmail.com)
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plotPareto(data):
    """
    plot GP explorations over each objective space, pareto front over parameter and objective space
    Args:
        data: a list of data required to plot figures
    :return: Null
    """

    pareto_y, pareto_ind, train_indices, train_Y, indices, y_pred_means, y_pred_vars, img_space, iter = \
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]
    # Plotting exploration and GP maps for individual objectives
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    for i in range(0, 2):
        ax[i, 0].imshow(img_space, origin="lower")
        # ax[i, 0].scatter(indices[:, 1], indices[:, 0], c=img_space[:, i], cmap='viridis', linewidth=0.1)
        ax[i, 0].scatter(train_indices[:, 1], train_indices[:, 0], marker='s', c=train_Y[:, i], cmap='jet',
                         linewidth=0.1)
        ax[i, 0].axes.xaxis.set_visible(False)
        ax[i, 0].axes.yaxis.set_visible(False)

        a = ax[i, 1].scatter(indices[:, 1], indices[:, 0], c=y_pred_means[:, i], cmap='viridis', linewidth=0.2)
        ax[i, 1].scatter(train_indices[:, 1], train_indices[:, 0], marker='s', c=train_Y[:, i], cmap='jet',
                         linewidth=0.1)
        divider = make_axes_locatable(ax[i, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(a, cax=cax, orientation='vertical')
        ax[i, 1].set_title('GP mean: Obj ' + str(i + 1) + ' iter= ' + str(iter), fontsize=10)
        ax[i, 1].axes.xaxis.set_visible(False)
        ax[i, 1].axes.yaxis.set_visible(False)
        # ax[1].colorbar(a)

        b = ax[i, 2].scatter(indices[:, 1], indices[:, 0], c=y_pred_vars[:, i], cmap='viridis', linewidth=0.1)
        divider = make_axes_locatable(ax[i, 2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(b, cax=cax, orientation='vertical')
        ax[i, 2].set_title('GP var: Obj ' + str(i + 1) + ' iter= ' + str(iter), fontsize=10)
        ax[i, 2].axes.xaxis.set_visible(False)
        ax[i, 2].axes.yaxis.set_visible(False)
        # ax[2].colorbar(b)

    plt.savefig('acquisition_results_step=' + str(iter) + '.png', dpi=300, bbox_inches='tight', pad_inches=1.0)
    plt.show()

    # Plotting Pareto front

    fig, ax = plt.subplots(ncols=2, figsize=(12, 5))

    # Pareto in parameter space
    ax[0].imshow(img_space, origin="lower")
    ax[0].scatter(pareto_ind[:, 0], pareto_ind[:, 1], c=pareto_y[:, 0], cmap='gist_rainbow', linewidth=0.2)
    ax[0].set_xlabel('X1')
    ax[0].set_ylabel('X2')
    ax[0].set_title('Pareto frontier over parameter space, iter=' + str(iter))

    # Pareto in objective space

    ax[1].scatter(pareto_y[:, 0], pareto_y[:, 1], c=pareto_y[:, 0], cmap='gist_rainbow', linewidth=0.2)
    ax[1].set_xlabel('Target 1')
    ax[1].set_ylabel('Target 2')
    # ax[1].set_xlim((-5, 165))
    # ax[1].set_ylim((-9.5, -2.5))
    ax[1].set_title('Pareto frontier over objective space, iter=' + str(iter))
    plt.savefig('pareto_results_step=' + str(iter) + '.png', dpi=300, bbox_inches='tight', pad_inches=1.0)
    plt.show()
