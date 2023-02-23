from util import collect_exp_results, ucb, lcb, median, count
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from config import BASE_DIR, PLOT_DICTS, PLOT_DICTS_APPENDIX
import os
from plot_specs import metric_to_tex, NEURIPS_RCPARAMS, method_to_tex, method_to_color
from tueplots import bundles
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import seaborn as sns

palette = sns.color_palette("colorblind")

rc = bundles.neurips2022(usetex=True, ncols=4, nrows=2)
rc['text.latex.preamble'] = (r'\usepackage{amsmath}'
                             r'\usepackage{amssymb}'
                             r'\usepackage{amsfonts}')
sns.set_theme(style="whitegrid", palette=palette, rc=NEURIPS_RCPARAMS)
# sns.set_theme(style="whitegrid", palette="husl", rc=NEURIPS_RCPARAMS)

QUANTILE_BASED_CI = True
BOXENPLOT = False
NTICKS = 4
DOTSIZE = 6
CAPSIZE = 3
PLOT_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

METHODS = ["JCI-PC", "IGSP", "DCDI-G", "bacadi"]

if not BOXENPLOT:
    scale_down = 1.6
else:
    scale_down = 1.2
figsize_x = 6
figsize_y = 2

for plot_dict in PLOT_DICTS:
    n_metrics = len(plot_dict['metrics'])
    title = plot_dict['title']
    df_all = pd.DataFrame()
    nrows = len(plot_dict['plot_dirs'])
    if nrows == 1:
        figsize_y /= 1.6
        gridspec_kw = {'hspace': 0.05, 'wspace': 0.4}
    else:
        gridspec_kw = {'hspace': 0.05, 'wspace': 0.8}
    fig, axes = plt.subplots(ncols=n_metrics,
                             gridspec_kw=gridspec_kw,
                             nrows=nrows,
                             figsize=(figsize_x / scale_down,
                                      figsize_y / scale_down))
    for row_num, (graph, result_dirs) in enumerate(plot_dict['plot_dirs']):
        num_methods = len(result_dirs)
        methods = [result_dir[0] for result_dir in result_dirs]
        for k, (method, result_dir) in enumerate(result_dirs):

            df_full, param_names, _ = collect_exp_results(exp_name=result_dir,
                                                          dir_tree_depth=1)

            # remove columns that only contain nans
            nan_col_mask = df_full.isna().apply(lambda col: np.all(col),
                                                axis=0)
            nan_cols = list(df_full.columns[nan_col_mask])
            df_full = df_full.drop(nan_cols, axis=1)

            if BOXENPLOT:  # group over everything except seeds and aggregate over the seeds
                df_all = pd.concat([df_all, df_full], ignore_index=True)
                continue

            groupby_names = list(
                set(param_names) - set(nan_cols) - {'seed', 'model_seed'})
            # df_grouped = df_full.groupby(by=groupby_names, axis=0)

            df_agg = df_full.groupby(by=groupby_names, axis=0).aggregate(
                ['mean', 'std', ucb, lcb, median, count], axis=0)
            df_agg.reset_index(drop=False, inplace=True)

            for col_num, metric in enumerate(plot_dict['metrics']):
                for j in range(1):
                    row = df_agg.iloc[j]
                    num_seeds = row[(metric, 'count')]
                    ci_factor = 1 / np.sqrt(num_seeds)

                    if QUANTILE_BASED_CI:
                        metric_median = row[(metric, 'median')]
                        label = method_to_tex.get(method) or method
                        lower_err = -(row[
                            (metric, 'lcb')] - metric_median) * ci_factor
                        upper_err = (row[(metric, 'ucb')] -
                                     metric_median) * ci_factor
                        if nrows > 1:
                            axes[row_num, col_num].scatter(
                                k,
                                metric_median,
                                s=DOTSIZE,
                                c=method_to_color.get(method),
                                label=label)
                            axes[row_num, col_num].errorbar(
                                k,
                                metric_median,
                                c=method_to_color.get(method),
                                yerr=np.array([[lower_err, upper_err]]).T,
                                capsize=CAPSIZE)
                        else:
                            axes[col_num].scatter(
                                k,
                                metric_median,
                                s=DOTSIZE,
                                c=method_to_color.get(method),
                                label=label)
                            axes[col_num].errorbar(
                                k,
                                metric_median,
                                c=method_to_color.get(method),
                                yerr=np.array([[lower_err, upper_err]]).T,
                                capsize=CAPSIZE)

                    else:
                        # NOT USED RN
                        metric_mean = row[(metric, 'mean')]
                        metric_std = row[(metric, 'std')]
                        if nrows > 1:
                            axes[row_num, col_num].scatter(k,
                                                           metric_mean,
                                                           s=DOTSIZE,
                                                           label=f'{method}')
                            axes[row_num, col_num].errorbar(
                                k,
                                metric_mean,
                                yerr=2 * metric_std * ci_factor,
                                capsize=CAPSIZE)
                        else:
                            axes[col_num].scatter(k,
                                                  metric_mean,
                                                  s=DOTSIZE,
                                                  label=f'{method}')
                            axes[col_num].errorbar(k,
                                                   metric_mean,
                                                   yerr=2 * metric_std *
                                                   ci_factor,
                                                   capsize=CAPSIZE)

                    if row_num == 0:
                        print(f'{method}_{j}', row['exp_result_folder'][0])
                ###
                # if row_num == 0 and col_num == 0:
                #     axes[row_num, col_num].set_ylabel('') # TODO
                if row_num == 1 and nrows > 1:
                    pass
                    # names at the bottom of bottom row?
                    # axes[row_num, col_num].set_xticklabels(methods)
                else:
                    if nrows == 1:
                        axes[col_num].set_xticks(np.arange(num_methods))
                        axes[col_num].title.set_text(metric_to_tex[metric])
                    else:

                        # axes[row_num, col_num].set_yticks(axes[row_num, col_num].get_yticks(), axes[row_num, col_num].get_yticklabels(), rotation=45, ha='right')
                        axes[row_num,
                             col_num].title.set_text(metric_to_tex[metric])

                if nrows > 1:
                    if metric == 'mixture_negll_interv' and title != 'BGe' and not (
                            "Obs" in title):
                        axes[row_num, col_num].set_yscale('log')
                        # axes[row_num, col_num].set_yticklabels([])
                        if '50' in title:
                            axes[row_num, col_num].set_ylim((101, 8000))
                    axes[row_num, col_num].set_xticks(np.arange(num_methods))
                    axes[row_num, col_num].set_xticklabels([])
                    axes[row_num, col_num].set_xlim((-0.5, num_methods - 0.5))
                else:
                    # if metric == 'mixture_negll_interv':
                    #     axes[col_num].set_yscale('log')
                    axes[col_num].set_xticklabels([])
                    axes[col_num].set_xlim((-0.5, num_methods - 0.5))

        if BOXENPLOT:
            for col_num, metric in enumerate(plot_dict['metrics']):
                if metric == 'mixture_negll_interv':
                    axes[row_num, col_num].set_yscale('log')
                sns.boxenplot(x='method',
                              y=metric,
                              data=df_all,
                              ax=axes[row_num, col_num],
                              k_depth='trustworthy',
                              showfliers=False)
                axes[row_num, col_num].set_ylabel('')
                axes[row_num, col_num].set_xlabel('')
                if row_num == 0:
                    axes[row_num,
                         col_num].title.set_text(metric_to_tex[metric])
                    axes[row_num, col_num].set_xticklabels([])
                else:
                    axes[row_num, col_num].set_xticklabels(
                        [method_to_tex[m] for m in METHODS])
    if nrows == 1:
        axes[0].set_ylabel(r'SERGIO', labelpad=2)
    else:
        # axes[0, 0].set_ylabel(r'$\text{Erd{\H{o}}s-R{\'e}nyi}$')
        axes[0, 0].set_ylabel(r'ER', labelpad=2)
        axes[1, 0].set_ylabel(r'SF', labelpad=2)
    # for axi in axes.flat:
    #     axi.xaxis.set_major_locator(plt.MaxNLocator(NTICKS))
    #     axi.yaxis.set_major_locator(plt.MaxNLocator(NTICKS))
    plt.tight_layout()
    if not BOXENPLOT:
        if nrows > 1:
            lines_labels = [
                ax.get_legend_handles_labels() for ax in [axes[1][2]]
            ]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            fig.legend(lines,
                       labels,
                       loc='center',
                       bbox_to_anchor=(0.5, 0.02),
                       fancybox=True,
                       columnspacing=0.8,
                       markerscale=1.2,
                       handlelength=.5,
                       ncol=4)
        else:
            lines_labels = [ax.get_legend_handles_labels() for ax in [axes[0]]]

            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            # only for sergio result with all 5
            if len(result_dirs) == 5:
                order = [0,3,1,4,2]
                # lines, labels = [sum(lol, []) for lol in zip(*lines_labels[order])]
                lines = [lines[i] for i in order]
                labels = [labels[i] for i in order]
                
            fig.legend(lines,
                       labels,
                       loc='center',
                       bbox_to_anchor=(0.5, -0.2),
                       fancybox=True,
                       columnspacing=0.8,
                       markerscale=1.2,
                       handlelength=.5,
                       ncol=3)

    plt.show()
    # fig.show()
    fig_path = os.path.join(PLOT_DIR, plot_dict['file_name'] + '.png')
    fig.savefig(fig_path)
    fig_path = os.path.join(PLOT_DIR, plot_dict['file_name'] + '.pdf')
    fig.savefig(fig_path, bbox_inches='tight')
    # fig_path = os.path.join(PLOT_DIR, plot_dict['file_name'] + '.svg')
    # fig.savefig(fig_path, bbox_inches='tight')
    print(f'Saved figure to {fig_path}')
