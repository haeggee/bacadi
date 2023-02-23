import os
from jax import jit, vmap
import jax.numpy as jnp
import pandas as pd
from bacadi.eval.metrics import expected_shd, expected_sid, kl_divergence, threshold_metrics, neg_ave_log_likelihood, neg_ave_log_marginal_likelihood
from bacadi.utils.func import joint_dist_to_marginal
from bacadi.eval.target import create_data_interv_tuples, save_pickle

STORE_ROOT = ['results']


def get_metrics(descr,
                dist,
                target,
                config,
                bacadi=None,
                use_cpdag=False,
                final=False):
    n_vars = target.n_vars
    metrics = {}

    # get targets used for inference to get interventional cpdag

    # negll only for bacadi
    if bacadi is not None:
        if config.joint:
            # evaluates log likelihood of all (G, theta) particles in batch on held-out data
            eltwise_log_likelihood = jit(
                vmap(
                    lambda w_, theta_, x_:
                    (bacadi.log_joint_likelihood(single_theta=theta_,
                                               single_w=w_,
                                               single_data=x_,
                                               single_interv_targets=None,
                                               envs=None)), (0, 0, None), 0))

            eltwise_log_interv_likelihood = jit(
                vmap(
                    lambda w_, theta_, x_: (bacadi.log_joint_likelihood(
                        single_theta=theta_,
                        single_w=w_,
                        single_data=x_,
                        single_interv_targets=target.interv_targets_ho,
                        envs=target.envs_ho)), (0, 0, None), 0))
            negll_method = neg_ave_log_likelihood

        else:
            # evaluates log likelihood of all (G, theta) particles in batch on held-out data
            eltwise_log_likelihood = jit(
                vmap(
                    lambda w_, x_: (bacadi.log_marginal_prob(
                        single_w=w_, data=x_, interv_targets=None, envs=None)),
                    (0, None), 0))

            eltwise_log_interv_likelihood = jit(
                vmap(
                    lambda w_, x_: (bacadi.log_marginal_prob(
                        single_w=w_,
                        data=x_,
                        interv_targets=target.interv_targets_ho,
                        envs=target.envs_ho)), (0, None), 0))

            negll_method = neg_ave_log_marginal_likelihood
        negll = negll_method(dist=dist,
                             eltwise_log_target=eltwise_log_likelihood,
                             x=target.x_ho,
                             unknown_interv=config.infer_interv)
        negll_interv = negll_method(
            dist=dist,
            eltwise_log_target=eltwise_log_interv_likelihood,
            x=target.x_ho_interv_data,
            unknown_interv=config.infer_interv)
        metrics[descr + 'negll'] = negll
        metrics[descr + 'negll_interv'] = negll_interv

    dist_marginal = joint_dist_to_marginal(dist) if len(dist) >= 3 else dist

    gt_posterior = target.gt_posterior_interv if config.interv_data else target.gt_posterior_obs
    if gt_posterior is not None:
        kl_div = kl_divergence(n_vars, dist_marginal,
                            gt_posterior) if gt_posterior is not None else -1.        
        metrics[descr + 'kl_div'] = kl_div
    if config.interv_data or config.infer_interv:
        interv_targets_train = target.interv_targets
        if (interv_targets_train[0] == 0).all():
            interv_targets_train = interv_targets_train[1:]
    else:
        interv_targets_train = jnp.zeros((1, n_vars))
    interv_target_list = [
        set([i_ for i_ in range(n_vars) if t[i_]])
        for t in interv_targets_train
    ]

    if final:
        esid = expected_sid(dist=dist_marginal,
                            g=target.g,
                            use_cpdag=use_cpdag,
                            interv_targets=interv_target_list)
        metrics[descr + 'esid'] = esid

    eshd = expected_shd(dist=dist_marginal,
                        g=target.g,
                        use_cpdag=use_cpdag,
                        interv_targets=interv_target_list)

    if final:
        thresh_metr = threshold_metrics(
            dist=dist_marginal,
            g=target.g,
            undirected_cpdag_oriented_correctly=use_cpdag)
    else:
        thresh_metr = threshold_metrics(dist=dist_marginal, g=target.g)

    metrics[descr + 'eshd'] = eshd
    metrics[descr + 'auroc'] = thresh_metr['roc_auc']
    metrics[descr + 'auprc'] = thresh_metr['prc_auc']
    metrics[descr + 'avgprec'] = thresh_metr['ave_prec']
    if config.infer_interv:
        interv_targets = target.interv_targets
        if (interv_targets[0] == 0).all():
            interv_targets = interv_targets[1:]
        metrics_i = threshold_metrics(dist=(dist[-2], dist[-1]),
                                      g=interv_targets,
                                      is_graph_distr=False)

        metrics[descr + 'intv_auroc'] = metrics_i['roc_auc']
        metrics[descr + 'intv_auprc'] = metrics_i['prc_auc']
        metrics[descr + 'intv_avgprec'] = metrics_i['ave_prec']
    return metrics


def process_incoming_result(result_df, incoming_dfs, args):
    """
    Appends new pd.DataFrame and saves the result in the existing .csv file  
    """

    # concatenate existing and new results
    result_df = pd.concat([result_df, incoming_dfs], ignore_index=True)

    save_path = os.path.join(*STORE_ROOT)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # save to csv
    save_path = os.path.abspath(os.path.join(save_path, args.descr + '.csv'))
    result_df.to_csv(save_path)

    return result_df