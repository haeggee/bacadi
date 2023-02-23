import os
import matplotlib.pyplot as plt
import imageio
import seaborn as sns


def visualize_ground_truth(mat, size=4.0, interventions=False):
    """    
    `mat`: (k, d) 
    """
    plt.rcParams['figure.figsize'] = [size, size]
    fig, ax = plt.subplots(1, 1)
    ax.matshow(mat, vmin=0, vmax=1, cmap='cividis')
    # sns.heatmap(mat,
    #             cmap="YlGnBu",
    #             cbar=False,
    #             ax=ax,
    #             linewidths=1.,
    #             center=0.3,
    #             square=True,
    #             linecolor='white')
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_title(r'Ground truth $G^*$', pad=10)
    if interventions:
        ax.set_title(r'Ground truth $I^*$', pad=10)
    plt.show()
    return

def visualize_obs_interv(mat_gt,
                         exp_g_obs,
                         exp_g_intv,
                         mat_obs,
                         mat_intv,
                         size=4.0):
    """    
    `mat`: (d, d) 
    """
    plt.rcParams['figure.figsize'] = [size * 3, size * 2]
    fig, axes = plt.subplots(2, 3)
    axes[0, 0].matshow(mat_gt, vmin=0, vmax=1)
    axes[0, 1].matshow(exp_g_obs, vmin=0, vmax=1)
    axes[0, 2].matshow(exp_g_intv, vmin=0, vmax=1)
    axes[1, 1].matshow(mat_obs, vmin=0, vmax=1)
    axes[1, 2].matshow(mat_intv, vmin=0, vmax=1)
    for i in range(2):
        for j in range(3):
            plt.setp(axes[i, j].get_xticklabels(), visible=False)
            plt.setp(axes[i, j].get_yticklabels(), visible=False)
            axes[i, j].tick_params(axis='both', which='both', length=0)
            axes[i, j].axis('off')
    axes[0, 0].set_title(r'Ground truth $G^*$', pad=10)
    axes[0, 1].set_title(r'Expected $G^O$ with obs. data', pad=10)
    axes[0, 2].set_title(r'Expected $G^I$ with interv data', pad=10)
    axes[1, 1].set_title(r'MAP $G^O$ with obs. data', pad=10)
    axes[1, 2].set_title(r'MAP $G^I$ with interv data', pad=10)
    plt.show()
    return


def visualize(mats, t, save_path=None, n_cols=7, size=2.5, show=False):
    """
    Based on visualization by https://github.com/JannerM/gamma-models/blob/main/gamma/visualization/pendulum.py
    
    `mats` should have shape (N, d, k) and take values in [0,1]
    """

    N = mats.shape[0]
    n_rows = N // n_cols
    if N % n_cols:
        n_rows += 1

    plt.rcParams['figure.figsize'] = [size * n_cols, size * n_rows]
    fig, axes = plt.subplots(n_rows, n_cols)
    axes = axes.flatten()

    # for j, (ax, mat) in enumerate(zip(axes[:len(mats)], mats)):
    for j, ax in enumerate(axes):
        if j < len(mats):
            # plot matrix of edge probabilities
            ax.matshow(mats[j], vmin=0, vmax=1, cmap='cividis')
            # sns.heatmap(mats[j],
            #             cmap="YlGnBu",
            #             cbar=False,
            #             ax=ax,
            #             linewidths=1.,
            #             center=0.3,
            #             square=True,
            #             linecolor='white')
            ax.tick_params(axis='both', which='both', length=0)
            ax.set_title(r'Particle$^{('f'{j}'r')}$', pad=3)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.axis('off')


    # save
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + f'/img{t}.png')
        img = imageio.imread(save_path + f'/img{t}.png')
    else:
        img = None
    if show:
        plt.show()
    plt.close()
    return img
