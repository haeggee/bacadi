import os
import glob
import json
import pandas as pd
import numpy as np
from config import RESULT_DIR
import shutil

def ucb(row, q=0.95):
    assert row.shape[0] > 1
    return np.quantile(row, q=q, axis=0)


def lcb(row, q=0.05):
    assert row.shape[0] > 1
    return np.quantile(row, q=q, axis=0)


def median(row):
    assert row.shape[0] > 1
    return np.quantile(row, q=0.5, axis=0)


def count(row):
    return row.shape[0]


def ece(preds, g=True):
    """
        y_target: [N, pred]
        y_pred: [N, pred]
    """

    bins = np.linspace(0.0, 1.0 + 1e-8, N_BINS + 1)
    if g:
        y_prob = preds['g_empirical'].reshape(-1)
        y_true = preds['g_gt'].reshape(-1)
    else:
        y_prob = preds['I_empirical'].reshape(-1)
        y_true = preds['interv_targets_gt'].reshape(-1)
    binids = np.digitize(y_prob, bins) - 1
    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    ece = (bin_total[nonzero] * np.abs(prob_true - prob_pred)).sum() / y_prob.shape[0]
    return ece
    


def collect_exp_results(exp_name: str,
                        dir_tree_depth: int = 3,
                        verbose: bool = True):
    exp_dir = os.path.join(RESULT_DIR, exp_name)
    print(exp_dir)
    no_results_counter = 0
    success_counter = 0
    exp_dicts = []
    param_names = set()
    g_gt = []
    interv_targets_gt = []
    g_empirical = []
    g_mixture = []
    I_empirical = []
    I_mixture = []
    search_path = os.path.join(
        exp_dir, '/'.join(['*' for _ in range(dir_tree_depth)]) + '.json')
    results_jsons = glob.glob(search_path)
    for results_file in results_jsons:

        if os.path.isfile(results_file):
            try:
                with open(results_file, 'r') as f:
                    exp_dict = json.load(f)
                if isinstance(exp_dict, dict):
                    if 'dcdi_mu_mul_factor' in exp_dict['params'].keys():
                        exp_dict['params']['dcdi_mu_mult_factor'] = exp_dict[
                            'params'].pop('dcdi_mu_mul_factor')
                    exp_dicts.append({
                        **exp_dict['evals'],
                        **exp_dict['params'],
                        'duration': exp_dict['duration']
                    })
                    param_names = param_names.union(
                        set(exp_dict['params'].keys()))
                    g_gt.append(exp_dict['g_gt'])
                    interv_targets_gt.append(exp_dict['interv_targets_gt'][1:])
                    g_empirical.append(exp_dict['g_empirical'])
                    g_mixture.append(exp_dict['g_mixture'])
                    I_empirical.append(exp_dict['I_empirical'])
                    I_mixture.append(exp_dict['I_mixture'])
                elif isinstance(exp_dict, list):
                    exp_dicts.extend([{
                        **d['evals'],
                        **d['params']
                    } for d in exp_dict])
                    for d in exp_dict:
                        param_names = param_names.union(set(
                            d['params'].keys()))
                else:
                    raise ValueError
                success_counter += 1
            except json.decoder.JSONDecodeError as e:
                print(f'Failed to load {results_file}', e)
        else:
            no_results_counter += 1

    assert success_counter + no_results_counter == len(results_jsons)
    if verbose:
        print(
            f'Parsed results in {search_path} - found {success_counter} folders with results'
            f' and {no_results_counter} folders without results')

    g_gt = np.array(g_gt)
    interv_targets_gt = np.array(interv_targets_gt)
    g_empirical = np.array(g_empirical)
    g_mixture = np.array(g_mixture)
    I_empirical = np.array(I_empirical)
    I_mixture = np.array(I_mixture)

    pred_dict = dict(
        g_gt=g_gt,
        interv_targets_gt=interv_targets_gt,
        g_empirical=g_empirical,
        g_mixture=g_mixture,
        I_empirical=I_empirical,
        I_mixture=I_mixture,
    )
    return pd.DataFrame(data=exp_dicts), list(param_names), pred_dict





