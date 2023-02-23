import torch
import tqdm
import time
from datetime import datetime

import numpy as onp

from jax import random
from jax.scipy.special import logsumexp
import jax.numpy as jnp
from baselines.learners import DCDILearner, IGSPLearner, JCI_PC_Learner

from bacadi.utils.graph import *
from bacadi.utils.func import bit2id, id2bit

from bacadi.exceptions import InvalidCPDAGError, ContinualInvalidCPDAGError, BaselineTimeoutError


class NonparametricDAGBootstrap:
    """
    Nonparametric DAG Bootstrap as proposed by
        https://arxiv.org/abs/1608.04471
    
    and e.g. used in 
        https://arxiv.org/abs/1902.10347 (Algorithm 1)
    
    but adapted for data from different environments.
    When doing the bootstrapping samples, sampling is done individually
    for each environment.

    Arguments: 

        learner :    
            DAG learning algorithm satisfying the following signature
                in: 
                    x :   [n_data, n_vars] a number of observations used to learn a DAG
                    intv_targets: [n_envs, n_vars] gt mask of intervention targets
                    envs: [n_data] indicating which env a sample is from
                out:
                    mat : [n_vars, n_vars] adjacency matrix of a DAG
                    intv_targets: [n_envs, n_vars] either gt returned or what mask was inferred
                    nll: negloglik of the method on x

        n_restarts: int
            number of restarts with lower confidence thresholds in case an invalid CPDAG is returned

        no_bootstrap: bool
            if true, does not do any bootstrapping and is run once on the full dataset
    """
    def __init__(self, *, verbose, n_restarts=0, no_bootstrap=False):
        super(NonparametricDAGBootstrap, self).__init__()

        self.verbose = verbose
        self.n_restarts = n_restarts
        self.no_bootstrap = no_bootstrap

    def sample_bootstrap(self, *, key, x, envs):
        boot_samples = []
        boot_envs = []
        for e in onp.unique(envs):
            # correct sampling for each environment individually
            x_e = x[envs == e]
            n_observations = x_e.shape[0]
            n = n_observations

            key, subk = random.split(key)
            idxs = random.choice(subk,
                                 n_observations,
                                 shape=(n, ),
                                 replace=True)
            boot_samples.append(x_e[idxs, :])
            boot_envs.append(jnp.array([e] * n_observations, dtype=jnp.int32))
        boot_samples = jnp.concatenate(boot_samples)
        boot_envs = jnp.concatenate(boot_envs)
        return boot_samples, boot_envs

    def sample_particles(self,
                         *,
                         learner,
                         key,
                         n_samples,
                         x,
                         envs,
                         verbose_indication=0):
        """
        Generates `n_samples` DAGs by bootstrapping (sampling with replacement) `n_data` points from x
        and learning a single DAG using an external DAG learning algorithm, in total `n_samples` times

            key
            x : [n_observations, n_vars]
            n_samples : int

        Returns 
            results: list of (dags, intv_targets, nll, nll_ho, nll_ho_intv)
        """
        last_verbose_indication = 1
        t_start = time.time()

        results = []
        for l in tqdm.tqdm(range(n_samples),
                           desc='NonparametricDAGBootstrap',
                           disable=not self.verbose):

            learner.reinit(ci_alpha=learner.ci_alpha_init)
            # sample bootstrap dataset
            if self.no_bootstrap:
                boot_samples = x
                boot_envs = envs
            else:
                key, subk = random.split(key)
                boot_samples, boot_envs = self.sample_bootstrap(key=subk,
                                                                x=x,
                                                                envs=envs)
            # learn DAG
            key, subk = random.split(key)

            attempts = 0
            while True:
                try:
                    mat, intv_targets, loglik, nll_ho, nll_ho_intv = learner.learn_dag(
                        key=subk, x=boot_samples, envs=boot_envs)
                    results.append((jnp.array(mat), jnp.array(intv_targets),
                                    loglik, nll_ho, nll_ho_intv))
                    learner.final_try = False
                    break

                except (onp.linalg.LinAlgError, InvalidCPDAGError) as e:
                    attempts += 1
                    if attempts > self.n_restarts:
                        if learner.final_try: # it just was the final try
                            if self.verbose:
                                print(
                                    f'{type(learner).__name__} did not return an extendable CPDAG '
                                    'likely due to an undirected chain '
                                    'OR singular matrices for IGSP. \n'
                                    'Skipping this bootstrap sample.')
                            break
                        else:
                            learner.final_try = True # next one is final try
                    else:
                        if self.verbose:
                            print(
                                f'{type(learner).__name__} threw a LinAlgError (mostly due to singular '
                                'matrices) or threw a InvalidCPDAG. Restarting with harder threshold'
                            )
                        learner.reinit(ci_alpha=learner.ci_alpha / 2.0)
            if self.no_bootstrap:
                break

            # verbose progress
            if verbose_indication > 0:
                if (l + 1) >= (last_verbose_indication * n_samples //
                               verbose_indication):
                    print(
                        f'DAGBootstrap {type(learner).__name__}    {l + 1} / {n_samples} [{(100 * (l + 1) / n_samples):3.1f} % '
                        +
                        f'| {((time.time() - t_start)/60):.0f} min | {datetime.now().strftime("%d/%m %H:%M")}]',
                        flush=True)
                    last_verbose_indication += 1

        if not results:
            if self.verbose:
                print(
                    'Could not find a valid DAG for any of the boostrap datasets.\n'
                    'Will try no_bootstrap')
            try:
                learner.final_try = True
                boot_samples = x
                boot_envs = envs
                mat, intv_targets, nll, nll_ho, nll_ho_intv = learner.learn_dag(
                    key=subk, x=boot_samples, envs=boot_envs)
                results.append((jnp.array(mat), jnp.array(intv_targets), nll,
                                nll_ho, nll_ho_intv))
                learner.final_try = False
            except Exception as e:
                if self.verbose:
                    print('Still did not receive a valid DAG. Exception:', e)
                raise ContinualInvalidCPDAGError(
                    'Could not find a valid DAG for any of the boostrap datasets.'
                )
        if self.verbose:
            print(f"Successful bootstrap samples: {len(results)}")
        return results


def run_bootstrap(key, config, target, learner_str):
    interv_setting = config.interv_data or config.infer_interv
    if interv_setting:
        x = target.x_interv_data
        envs = target.envs
    else:
        x = target.x
        envs = jnp.array([0] * x.shape[0], dtype=jnp.int64)

    n_bootstrap_samples = config.n_bootstrap_samples
    boot = NonparametricDAGBootstrap(
        verbose=config.verbose,
        n_restarts=config.bootstrap_n_error_restarts,
        no_bootstrap=config.no_bootstrap)

    if learner_str == 'dcdi':
        learner = DCDILearner(target, config)
        n_bootstrap_samples = config.dcdi_n_bootstrap_samples
        if config.dcdi_gpu:
            torch.cuda.empty_cache()
    elif learner_str == 'jci-pc':
        learner = JCI_PC_Learner(target, config)
    elif learner_str == 'igsp':
        learner = IGSPLearner(target, config)
    else:
        raise NotImplementedError(f"{learner_str} not implemented")

    # list of (dags/cpdags, intv_targets, loglik, nll_ho, nll_ho_intv)
    boot_samples = boot.sample_particles(learner=learner,
                                         key=key,
                                         n_samples=n_bootstrap_samples,
                                         x=x,
                                         envs=envs)
    n_particles = len(boot_samples)

    # get same distr tuple just like in the other methods
    ids = bit2id(jnp.array([res[0] for res in boot_samples], dtype=jnp.int32))
    particles_I = jnp.array([res[1] for res in boot_samples])
    log_probs_empirical = -jnp.log(n_particles) * jnp.ones(n_particles)
    dist_empirical = (ids, particles_I, log_probs_empirical)

    # mixture weighted by negll on train
    log_probs_mixture = jnp.array([res[2] for res in boot_samples])
    log_probs_mixture -= logsumexp(log_probs_mixture)
    dist_mixture = (ids, particles_I, log_probs_mixture)

    # weighted average of loglik metrics
    neglls_ho = jnp.array([res[3] for res in boot_samples])
    neglls_ho_interv = jnp.array([res[4] for res in boot_samples])

    metrics = {}
    for descr1, log_weights in [('empirical_', log_probs_empirical),
                                ('mixture_', log_probs_mixture)]:
        for descr2, log_scores in [('negll', neglls_ho),
                                   ('negll_interv', neglls_ho_interv)]:
            log_score, log_score_sgn = logsumexp(log_weights,
                                                 b=log_scores,
                                                 axis=0,
                                                 return_sign=True)
            score = log_score_sgn * jnp.exp(log_score)
            metrics[descr1 + descr2] = score
    return dist_empirical, dist_mixture, metrics
