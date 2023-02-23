from copy import deepcopy
import os
import pickle
from collections import namedtuple
from warnings import warn
import numpy as onp
import jax.numpy as jnp
from jax import random, vmap, jit
from jax.scipy.special import logsumexp

from bacadi.graph.graph import ErdosReniDAGDistribution, ScaleFreeDAGDistribution, UniformDAGDistributionRejection
from bacadi.kernel.joint import JointAdditiveFrobeniusSEKernel, JointMultiplicativeFrobeniusSEKernel
from bacadi.kernel.interv import JointAdditiveFrobeniusSEKernel as JointAdditiveInterv
from bacadi.kernel.interv import MarginalAdditiveFrobeniusSEKernel as MarginalAdditiveInterv
from bacadi.kernel.marginal import FrobeniusSquaredExponentialKernel
from bacadi.utils.graph import graph_to_mat, adjmat_to_str, make_all_dags, mat_to_graph

from bacadi.models.linearGaussian import LinearGaussian, LinearGaussianJAX
from bacadi.models.linearGaussianEquivalent import BGe, BGeJAX, NewBGe
from bacadi.models.nonlinearGaussian import DenseNonlinearGaussianJAX
from bacadi.models.sobolev import SobolevGaussianJAX

from bacadi.utils.func import bit2id

from sergio.sergio_sampler import sergio_clean

STORE_ROOT = ['store'] 

if not os.path.exists(os.path.join(*STORE_ROOT)):
    os.makedirs(os.path.join(*STORE_ROOT))


Target = namedtuple('Target', (
    'passed_key',               # jax.random key passed _into_ the function generating this object
    'graph_model',
    'n_vars',
    'n_observations',
    'n_ho_observations',
    'g',                        # [n_vars, n_vars]
    'theta',                    # PyTree
    'x',                        # [n_observation, n_vars]    data
    'x_ho',                     # [n_ho_observation, n_vars] held-out data
    'x_interv',                 # list of (interv dict, interventional data) 
    'x_interv_data',            # (n_obs stacked, n_vars) interventional data (same data as above)
    'interv_targets',           # (n_env, n_vars)
    'envs',                     # (n_obs) giving env that sample i came from
    'x_ho_interv',              # a tuple (interv dict, held-out interventional data) 
    'x_ho_interv_data',         # (n_obs stacked, n_vars) heldout interventional data (same data as above)
    'envs_ho',                  # (n_obs) giving env that sample i came from
    'interv_targets_ho',        # (n_env, n_vars)
    'gt_posterior_obs',         # ground-truth posterior with observational data
                                # (log distribution tuple as e.g. given by `particle_marginal_empirical`), or None
    'gt_posterior_interv'       # ground-truth posterior with interventional data
                                # (log distribution tuple as e.g. given by `particle_marginal_empirical`), or None
))


def save_pickle(obj, relpath):
    """Saves `obj` to `path` using pickle"""
    save_path = os.path.abspath(os.path.join(
        *STORE_ROOT, relpath + '.pk'
    ))
    with open(save_path, 'wb') as fp:
        pickle.dump(obj, fp)

def load_pickle(relpath):
    """Loads object from `path` using pickle"""
    load_path = os.path.abspath(os.path.join(
        *STORE_ROOT, relpath + '.pk'
    ))
    with open(load_path, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def options_to_str(**options):
    return '__'.join(['{}={}'.format(k, v) for k, v in options.items()])


def hparam_dict_to_str(d):
    """
    Converts hyperparameter dictionary into human-readable string
    """
    strg = '_'.join([k + '=' + str(v) for k, v, in d.items()
                     if type(v) in [bool, int, float, str, dict]])
    return strg

def interv_list_to_tuple(x_interv, n_vars):
    """
    Transform list of pairs (intervention dict, samples) to a 
    tuple of jnp.arrays (data, intervention_targets)

    Args:
        x_interv: list of (intervention dict, x_) of length k,
                  where the dict is of node indices to values, and the x_ correspond
                  to samples from the interventional distribution
        n_vars (int): number of variables

    Returns:
        tuple (data, interv_targets)
        where data is a jnp.array of shape [sum_{n_obs}, n_vars]
        and interv_targets is a boolean mask of shape [sum_{n_obs}, n_vars]
    """
    data = jnp.concatenate([interv_set[1] for interv_set in x_interv])
    interv_targets = jnp.concatenate([
        jnp.array([[i in interv_set[0].keys() for i in range(n_vars)]
                   for _ in interv_set[1]]) for interv_set in x_interv
    ])
    env_size = jnp.array([interv_set[1].shape[0] for interv_set in x_interv])
    envs = jnp.repeat(jnp.arange(len(x_interv)), env_size)
    return (data, interv_targets, envs)


def create_data_interv_tuples(target):
    """
        Create tuples of the form (data, interv_targets, environments) for the 
        train and held-out datasamples of the target.
    
    Args:
        target (Target)
    
    Returns: 
        tuple (x, x_ho, X_interv, X_ho_interv)
    """
    # train, held-out and interventional samples
    n_vars = target.n_vars
    x = jnp.array(target.x)
    x_ho = jnp.array(target.x_ho)
    x_interv = target.x_interv
    x_ho_interv = target.x_ho_interv

    X_interv = interv_list_to_tuple(x_interv, n_vars)

    X_ho_interv = interv_list_to_tuple(x_ho_interv, n_vars)

    return (x, x_ho, X_interv, X_ho_interv)


def make_synthetic_bayes_net(*,
    key,
    n_vars,
    graph_model,
    generative_model,
    inference_model,
    n_observations=100,
    n_interv_obs=10,
    n_ho_observations=100,
    n_intervention_sets=10,
    n_ho_intervention_sets=10,
    perc_intervened=0.1,
    theta=None,
    g_gt_mat=None,
    interventions=None,
    intervention_val=0,
    intervention_noise=0,
    verbose=False,
    get_posteriors=False
):
    """
    Returns an instance of `Target` for evaluation of a method against 
    a ground truth synthetic Bayesian network

    Args:
        key: rng key
        c (int): seed
        graph_model (GraphDistribution): graph model object 
        generative_model (BasicModel): BN model object for generating the observations
        inference_model (BasicModel): JAX-BN model object for inference
        n_observations (int): number of observations generated for posterior inference
        n_ho_observations (int): number of held-out observations generated for validation
        n_intervention_sets (int): number of different interventions considered overall
            for generating interventional data
        perc_intervened (float): percentage of nodes intervened upon (set to 0) in 
            an intervention.
        theta (single parameter PyTree): parameters that can be predefined to e.g. 
            have a specific BN
        g_gt_mat (jnp.array of shape [n,n]): possibility to define a specific graph using
            an adjacency matrix
        interventions ("string" or list of dict(int, float)): a string or dictionary to specify what interventions
            to perform. if dict, overrides n_intervention_sets and perc_intervened

    Returns:
        `Target` 
    """

    # remember random key
    passed_key = key.copy()

    if g_gt_mat is None:
        # generate ground truth observations
        key, subk = random.split(key)
        g_gt = graph_model.sample_G(subk)
        g_gt_mat = jnp.array(graph_to_mat(g_gt))
    else:
        g_gt = mat_to_graph(g_gt_mat)

    if theta is None:
        key, subk = random.split(key)
        theta = generative_model.sample_parameters(key=subk, g=g_gt)

    n_interv = jnp.ceil(n_vars * perc_intervened).astype(jnp.int32)

    intervention_dict = None
    if type(interventions) is list:
        intervention_dict = interventions
    else:
        if interventions == "perfect":
            intervention_dict = [{
                "n_samples": n_observations # observational samples
            }] + [{
                i: {
                    'mean': intervention_val,
                    'noise': intervention_noise
                },
                "n_samples": n_interv_obs, # interventional samples
            } for i in range(n_vars)]
        elif interventions == "all_nodes":
            intervention_dict = [{
                i: {
                    'mean': intervention_val,
                    'noise': intervention_noise
                }
            } for i in range(n_vars)]
        elif interventions == "random":
            intervention_dict = [{
                "n_samples": n_observations  # observational samples
            }]
            for idx in range(n_intervention_sets):
                num_obs = n_interv_obs
                # random intervention
                key, subk = random.split(key)
                interv_val = random.normal(subk) * 2
                interv_val = jnp.sign(interv_val) * 5 + interv_val

                key, subk = random.split(key)
                interv_targets = random.choice(subk,
                                               n_vars,
                                               shape=(n_interv, ),
                                               replace=False)
                                
                interv = {
                    k: {
                        'mean': interv_val,
                        'noise': intervention_noise
                    }
                    for k in interv_targets
                }
                interv['n_samples'] = num_obs
                intervention_dict.append(interv)
        elif interventions == "random_all":
            # random intervention
            key, subk = random.split(key)
            interv_val = random.normal(subk, shape=(n_vars,)) * 2
            interv_val = jnp.sign(interv_val) * 5 + interv_val
            intervention_dict = [{
                "n_samples": n_observations # observational samples
            }] + [{
                i: {
                    'mean': interv_val[i],
                    'noise': intervention_noise
                },
                "n_samples": n_interv_obs, # interventional samples
            } for i in range(n_vars)]
        else:
            warn(
                f"intervention type {interventions} not known. random interventions will be done"
            )


    key, subk = random.split(key)
    x = generative_model.sample_obs(key=subk, n_samples=n_observations, g=g_gt, theta=theta)

    key, subk = random.split(key)
    x_ho = generative_model.sample_obs(key=subk, n_samples=n_ho_observations, g=g_gt, theta=theta)

    x_interv = []
    x_ho_interv = []

    """
        n_intervention_sets heldout interventional datasets used for evaluation.
        Each dataset has n_ho_observations
    """
    for idx in range(n_ho_intervention_sets):
        # random intervention
        key, subk = random.split(key)
        interv_targets = random.choice(subk,
                                       n_vars,
                                       shape=(n_interv, ),
                                       replace=False)
        interv = interv = {
                k: {
                    'mean': -intervention_val,
                    'noise': intervention_noise
                }
                for k in interv_targets
            }
        # observations from p(x | theta, G, interv) [n_samples, n_vars]
        key, subk = random.split(key)
        x_interv_ = generative_model.sample_obs(key=subk,
                                                n_samples=n_ho_observations,
                                                g=g_gt,
                                                theta=theta,
                                                interv=interv)
        x_ho_interv.append((interv, x_interv_))


    """
        Interventional datasets that can be used for inference
    """
    if intervention_dict is not None:
        n_intervention_sets = len(intervention_dict)

    """"
        n_intervention_sets random interventions
        where `perc_interv` % of nodes are intervened on,
        with an additional first set that is purely observational
        list of (interv dict, x)
    """
    for idx in range(n_intervention_sets):
        num_obs = n_interv_obs
        # random intervention
        if intervention_dict is None: # this is not executed anymore right now, since dict is always created
            key, subk = random.split(key)
            interv_targets = random.choice(subk,
                                           n_vars,
                                           shape=(n_interv, ),
                                           replace=False)
            interv = {
                k: {
                    'mean': intervention_val,
                    'noise': intervention_noise
                }
                for k in interv_targets
            }
        else:
            interv = intervention_dict[idx]
            num_obs = interv.get('n_samples') or num_obs
        # observations from p(x | theta, G, interv) [n_samples, n_vars]
        key, subk = random.split(key)
        x_interv_ = generative_model.sample_obs(key=subk,
                                                n_samples=num_obs,
                                                g=g_gt,
                                                theta=theta, interv=interv)
        x_interv.append((interv, x_interv_))

    if verbose:
        print(f'Sampled BN with {jnp.sum(g_gt_mat).item()}-edge DAG :\t {adjmat_to_str(g_gt_mat)}')


    gt_posterior_obs = None
    gt_posterior_interv = None

    if get_posteriors:
        obs_list = [({}, x)]
        gt_posterior_obs = gt_posterior(n_vars,
                                        *interv_list_to_tuple(
                                            obs_list, n_vars),
                                        graph_model=graph_model,
                                        inference_model=inference_model,
                                        verbose=verbose)
        gt_posterior_interv = gt_posterior(n_vars,
                                           *interv_list_to_tuple(
                                               x_interv, n_vars),
                                           graph_model=graph_model,
                                           inference_model=inference_model,
                                           verbose=verbose)

    (x_interv_data, _, envs) = interv_list_to_tuple(x_interv, n_vars)
    interv_targets = jnp.array(
        [[i in interv_set[0].keys() for i in range(n_vars)]
         for interv_set in x_interv])
    # if first env is not observational, add 1 to environment
    #  bc 0 is reserved for observational
    if not (interv_targets[0] == 0).all():
        interv_targets = jnp.concatenate([jnp.zeros((1, n_vars)), interv_targets])
        envs = envs + 1
    (x_ho_interv_data, _, envs_ho) = interv_list_to_tuple(x_ho_interv, n_vars)
    interv_targets_ho = jnp.array(
        [[i in interv_set[0].keys() for i in range(n_vars)]
         for interv_set in x_ho_interv])
    # if first env is not observational, add 1 to environment
    #  bc 0 is reserved for observational
    if not (interv_targets_ho[0] == 0).all():
        interv_targets_ho = jnp.concatenate([jnp.zeros((1, n_vars)), interv_targets_ho])
        envs_ho = envs_ho + 1

    # return and save generated target object
    obj = Target(passed_key=passed_key,
                 graph_model=graph_model,
                 n_vars=n_vars,
                 n_observations=n_observations,
                 n_ho_observations=n_ho_observations,
                 g=g_gt_mat,
                 theta=theta,
                 x=x,
                 x_ho=x_ho,
                 x_interv=x_interv,
                 x_interv_data=x_interv_data,
                 interv_targets=interv_targets,
                 envs=envs,
                 x_ho_interv=x_ho_interv,
                 x_ho_interv_data=x_ho_interv_data,
                 interv_targets_ho=interv_targets_ho,
                 envs_ho=envs_ho,
                 gt_posterior_obs=gt_posterior_obs,
                 gt_posterior_interv=gt_posterior_interv)
    return obj


def gt_posterior(n_vars,
                 data,
                 interv_targets,
                 *, graph_model,
                 inference_model,
                 verbose=False):
    """
        TODO: docstring
    """
    all_dags_n = make_all_dags(n_vars, return_matrices=False)
    all_dags_n_mat = jnp.array([graph_to_mat(g_) for g_ in all_dags_n])
    
    if verbose:
        # the index of our target graph:
        print('Number of DAGs: {all_dags_n_mat.shape[0]}')

    log_prob_g = jnp.array(
        [graph_model.unnormalized_log_prob(g=g_) for g_ in all_dags_n])

    log_marg_lik = jit(
        vmap(
            lambda g_: inference_model.log_marginal_likelihood_given_g(
                data=data, w=g_, interv_targets=interv_targets), (0, ),
            0))(all_dags_n_mat)

    log_posterior_unnormalized = log_prob_g + log_marg_lik
    log_posterior = log_posterior_unnormalized - logsumexp(
        log_posterior_unnormalized)
    return (bit2id(all_dags_n_mat), log_posterior)


def make_kernel(*, kernel, **kwargs):
    """
    Instantiates a kernel model

    Args:
        kernel: specifier (`frob-joint-add`, `frob-joint-mul`, `frob`)
        kwargs: dict
            must contain the key `h_latent`. for a joint kernel, additionally `h_theta`.

    Returns:
        `GraphDistribution`
    """
    ## Joint Kernels
    if kernel == 'frob-joint-add':
        kernel_ = JointAdditiveFrobeniusSEKernel(h_latent=kwargs['h_latent'],
                                                h_theta=kwargs['h_theta'])

    elif kernel == 'frob-joint-mul':
        kernel_ = JointMultiplicativeFrobeniusSEKernel(
            h_latent=kwargs['h_latent'], h_theta=kwargs['h_theta'])
    ## Joint Kernel with Interv
    elif kernel == 'frob-joint-interv-add':
        kernel_ = JointAdditiveInterv(h_latent=kwargs['h_latent'],
                                     h_theta=kwargs['h_theta'],
                                     h_interv=kwargs['h_interv'])
    ## Maginal Kernel with Interv
    elif kernel == 'frob-interv-add':
        kernel_ = MarginalAdditiveInterv(h_latent=kwargs['h_latent'],
                                        h_interv=kwargs['h_interv'])
    # Marginal
    elif kernel == 'frob':
        kernel_ = FrobeniusSquaredExponentialKernel(h=kwargs['h_latent'])

    return kernel_


def make_graph_model(*, n_vars, graph_prior_str, edges_per_node=2, n_edges=None):
    """
    Instantiates graph model

    Args:
        n_vars: number of variables
        graph_prior_str: specifier (`er`, `sf`)

    Returns:
        `GraphDistribution`
    """
    if graph_prior_str == 'er':
        graph_model = ErdosReniDAGDistribution(
            n_vars=n_vars, 
            n_edges=n_edges if n_edges is not None else edges_per_node * n_vars)

    elif graph_prior_str == 'sf':
        graph_model = ScaleFreeDAGDistribution(
            n_vars=n_vars,
            n_edges_per_node=edges_per_node)

    else:
        assert n_vars <= 5 
        graph_model = UniformDAGDistributionRejection(
            n_vars=n_vars)

    return graph_model

def make_inference_model(*, inference_str, n_vars, **kwargs):
    """
    Instantiates inference model

    Args:
        inference_str: specifier (`lingauss`, `fcgauss`, `bge`, `newbge`)
        n_vars: number of variables
        kwargs: dict
            for `lingauss`: must contain (`obs_noise`, `mean_edge`, `sig_edge`, `init_sig_edge`)
            for `bge`: must contain (`alpha_mu`)
            for `fcgauss`: must contain (`obs_noise`, `sig_param`, `hidden_layers`)

    Returns:
        Inference model `BasicModel`,
        either `LinearGaussianJAX`, `DenseNonlinearGaussianJAX`, `BGeJax` or `NewBGe` 
    """
    
    if inference_str == 'lingauss':
        inference_model = LinearGaussianJAX(
            obs_noise=kwargs['obs_noise'],
            mean_edge=kwargs['mean_edge'],
            sig_edge=kwargs['sig_edge'],
            init_sig_edge=kwargs['init_sig_edge'],
            interv_prior_mean=kwargs.get('interv_prior_mean') or 0.,
            interv_prior_std=kwargs.get('interv_prior_std') or 10.,
            interv_mean=kwargs.get('interv_mean'),
            interv_noise=kwargs.get('interv_noise'))

    elif inference_str == 'bge':
        alpha_lambd = kwargs[
            'alpha_lambd'] if 'alpha_lambd' in kwargs else n_vars + 2
        inference_model = BGeJAX(n_vars=n_vars,
                                 mean_obs=jnp.zeros(n_vars),
                                 alpha_lambd=alpha_lambd,
                                 alpha_mu=kwargs['alpha_mu'],
                                 interv_mean=kwargs.get('interv_mean'),
                                 interv_noise=kwargs.get('interv_noise'))

    elif inference_str == 'newbge':
        alpha_lambd = kwargs[
            'alpha_lambd'] if 'alpha_lambd' in kwargs else n_vars + 2
        inference_model = NewBGe(n_vars=n_vars,
                                 mean_obs=jnp.zeros(n_vars),
                                 alpha_lambd=alpha_lambd,
                                 alpha_mu=kwargs['alpha_mu'],
                                 interv_mean=kwargs.get('interv_mean'),
                                 interv_noise=kwargs.get('interv_noise'))

    elif inference_str == 'fcgauss':
        inference_model = DenseNonlinearGaussianJAX(
            obs_noise=kwargs['obs_noise'],
            sig_param=kwargs['sig_param'],
            init_sig_param=kwargs['init_sig_param'],
            hidden_layers=kwargs['hidden_layers'],
            interv_mean=kwargs.get('interv_mean'),
            interv_noise=kwargs.get('interv_noise'),
            init=kwargs.get('init') or 'xavier_normal',
            interv_prior_mean=kwargs.get('interv_prior_mean') or 0.,
            interv_prior_std=kwargs.get('interv_prior_std') or 10.,
            activation=kwargs.get('activation') or 'sigmoid',
            bias=kwargs.get('bias') or True)
    elif inference_str == 'sobolevgauss':
        inference_model = SobolevGaussianJAX(
            obs_noise=kwargs['obs_noise'],
            n_vars=n_vars,
            n_exp=kwargs.get('n_exp') or 10,
            mean_param=kwargs.get('mean_param') or 0.,
            sig_param=kwargs['sig_param'],
            init_sig_param=kwargs['sig_param'],
            init=kwargs.get('init') or 'normal')
    else:
        raise NotImplementedError(
            'Inference with {} not implemented.'.format(inference_str))

    return inference_model


def make_linear_gaussian_model(*, key, n_vars=20, graph_prior_str='er',
    inference='lingauss', obs_noise=0.1, random_noise=False, mean_edge=0.0, sig_edge=1.0,
    init_sig_edge=0.3, n_observations=100, n_interv_obs=10, n_ho_observations=100, n_intervention_sets=10,
    perc_intervened=0.1, interventions=None, alpha_mu=1., alpha_lambd_add=2,
    intervention_val=0,
    intervention_noise=0,
    edges_per_node=2, n_edges=None, g_gt_mat=None):

    """
    Samples a synthetic linear Gaussian BN instance.
    Depending on `inference`, use either inference
    with Bayesian Gaussian equivalent (BGe) marginal likelihood
    or a joint model with parameters (theta).

    By marginalizing out the parameters, the BGe model does not 
    allow inferring the parameters (theta). Additionally,
    it weights each DAG in an MEC equally,
    
    Args:
        key: rng key
        n_vars (int): number variables in BN
        graph_prior_str (str): graph prior (`er` or `sf`)
        inference (str): either use joint or marginal model (`lingauss` or `bge`)
        obs_noise (float): observation noise
        mean_edge (float): edge weight mean
        sig_edge (float): edge weight stddev
        n_observations (int): number of samples
        n_ho_observations (int): number of samples for held-out data
        n_intervention_sets (int): number of intervention sets. not used if
            interventions is not None
        perc_intervened (float): percentage of nodes to intervene on, sampled randomly.
            not used if interventions is not None
        interventions ("string" or list dict(int, float)): a string or dictionary to specify what interventions
            to perform. if list of dicts dict, overrides n_intervention_sets and perc_intervened
        alpha_mu (float): parameter for the BGe instance. not used for joint inference
    Returns:
        `Target` 
    """

    # init models
    graph_model = make_graph_model(n_vars=n_vars,
                                   graph_prior_str=graph_prior_str,
                                   edges_per_node=edges_per_node,
                                   n_edges=n_edges)

    if random_noise:
        key, subk = random.split(key)
        obs_noise = obs_noise * random.uniform(
            subk, shape=(n_vars, )) + obs_noise / 2
    generative_model = LinearGaussian(
        obs_noise=obs_noise, mean_edge=mean_edge, 
        sig_edge=sig_edge, g_dist=graph_model)
    
    inference_model = make_inference_model(inference_str=inference,
                                           n_vars=n_vars,
                                           alpha_mu=alpha_mu,
                                           alpha_lambd=n_vars +
                                           alpha_lambd_add,
                                           mean_edge=mean_edge,
                                           sig_edge=sig_edge,
                                           obs_noise=obs_noise,
                                           init_sig_edge=init_sig_edge)

    # sample synthetic BN and observations
    key, subk = random.split(key)
    target = make_synthetic_bayes_net(
        key=subk, n_vars=n_vars,
        graph_model=graph_model,
        generative_model=generative_model,
        inference_model=inference_model,
        n_observations=n_observations,
        n_interv_obs=n_interv_obs,
        n_ho_observations=n_ho_observations,
        n_intervention_sets=n_intervention_sets,
        perc_intervened=perc_intervened,
        interventions=interventions,
        intervention_val=intervention_val,
        intervention_noise=intervention_noise,
        g_gt_mat=g_gt_mat)

    return target


def make_nonlinear_gaussian_model(*, key, n_vars=20, graph_prior_str='er', model='fcgauss',
    obs_noise=0.1, random_noise=False, sig_param=1.0, init_sig_param=0.3, hidden_layers=[5,], n_observations=100,
    n_ho_observations=100, n_interv_obs=10, n_intervention_sets=10, perc_intervened=0.1,
    g_gt_mat=None, interventions=None, intervention_val=0, intervention_noise=0,
    edges_per_node=2, init_param='normal', activation='relu', bias=True, n_exp=10, mean_param=0.):

    """
    Samples a synthetic nonlinear Gaussian BN instance 
    where the local conditional distributions are parameterized
    by fully-connected neural networks

    Args:
        key: rng key
        n_vars (int): number variables in BN
        graph_prior_str (str): graph prior (`er` or `sf`)
        obs_noise (float): observation noise
        sig_param (float): stddev of the BN parameters,
            i.e. here the neural net weights and biases
        hidden_layers (list): list of ints specifying the hidden layer (sizes)
            of the neural nets parameterizatin the local condtitionals
        n_observations (int): number of samples
        n_ho_observations (int): number of samples for held-out data
        n_intervention_sets (int): number of intervention sets. not used if
            interventions is not None
        perc_intervened (float): percentage of nodes to intervene on, sampled randomly.
            not used if interventions is not None
        interventions ("string" or list dict(int, float)): a string or dictionary to specify what interventions
            to perform. if list of dicts dict, overrides n_intervention_sets and perc_intervened
    
    Returns:
        `Target` 
    """

    # init models
    graph_model = make_graph_model(n_vars=n_vars,
                                   graph_prior_str=graph_prior_str,
                                   edges_per_node=edges_per_node)

    if random_noise:
        key, subk = random.split(key)
        obs_noise = obs_noise * random.uniform(
            subk, shape=(n_vars, )) + obs_noise / 2
    if model == 'fcgauss':
        # generative model has same sig_param for sampling in the beginning
        generative_model = DenseNonlinearGaussianJAX(
            obs_noise=obs_noise,
            sig_param=sig_param,
            init_sig_param=sig_param,
            hidden_layers=hidden_layers,
            init=init_param,
            activation=activation,
            bias=bias)
        inference_model = DenseNonlinearGaussianJAX(
            obs_noise=obs_noise,
            sig_param=sig_param,
            init_sig_param=init_sig_param,
            hidden_layers=hidden_layers,
            init=init_param,
            activation=activation,
            bias=bias)
    elif model == 'sobolevgauss':
        generative_model = SobolevGaussianJAX(obs_noise=obs_noise,
                                              n_vars=n_vars,
                                              n_exp=n_exp,
                                              mean_param=mean_param,
                                              sig_param=sig_param,
                                              init_sig_param=sig_param,
                                              init=init_param)

        inference_model = SobolevGaussianJAX(obs_noise=obs_noise,
                                             n_vars=n_vars,
                                             n_exp=n_exp,
                                             mean_param=mean_param,
                                             sig_param=sig_param,
                                             init_sig_param=sig_param,
                                             init=init_param)
    else:
        raise ValueError(f'Unknown non-linear model: `{model}`')

    # sample synthetic BN and observations
    key, subk = random.split(key)
    target = make_synthetic_bayes_net(
        key=subk,
        n_vars=n_vars,
        graph_model=graph_model,
        g_gt_mat=g_gt_mat,
        generative_model=generative_model,
        inference_model=inference_model,
        n_observations=n_observations,
        n_interv_obs=n_interv_obs,
        n_ho_observations=n_ho_observations,
        n_intervention_sets=n_intervention_sets,
        perc_intervened=perc_intervened,
        interventions=interventions,
        intervention_val=intervention_val,
        intervention_noise=intervention_noise
    )

    return target


def make_sergio(n_vars,
                seed=1,
                sergio_hill=2,
                sergio_decay=0.8,
                sergio_noise_params=1.0,
                sergio_cell_types=10,
                sergio_k_lower_lim=1,
                sergio_k_upper_lim=5,
                graph_prior_edges_per_node=2,
                n_observations=100,
                n_ho_observations=100,
                n_intervention_sets=5,
                n_interv_obs=10):
    key = random.PRNGKey(seed)
    graph_model = make_graph_model(
        n_vars=n_vars,
        graph_prior_str='sf',
        edges_per_node=graph_prior_edges_per_node)

    key, subk = random.split(key)
    g_gt = graph_model.sample_G(subk)
    g_gt_mat = jnp.array(graph_to_mat(g_gt))

    n_ko_genes = n_intervention_sets  # TODO
    rng = onp.random.default_rng(seed)
    passed_rng = deepcopy(rng)

    def k_param(rng, shape):
        return rng.uniform(
            low=sergio_k_lower_lim,  # default 1 
            high=sergio_k_upper_lim,  # default 5
            size=shape)

    # MLE from e.coli
    def k_sign_p(rng, shape):
        return rng.beta(a=0.2588, b=0.2499, size=shape)

    def b(rng, shape):
        return rng.uniform(low=1, high=3, size=shape)

    n_obs = n_observations
    n_obs_ho = n_ho_observations
    spec = {}
    # number of intv
    spec['n_observations_int'] = 2 * n_intervention_sets * n_interv_obs
    # double to have heldout dataset
    spec['n_observations_obs'] = 2 * n_observations + n_ho_observations

    data = sergio_clean(
        spec=spec,
        rng=rng,  # function?
        g=g_gt_mat,
        effect_sgn=None,  # TODO
        toporder=None,
        n_vars=n_vars,
        b=b,  # function
        k_param=k_param,  # function
        k_sign_p=k_sign_p,  # function
        hill=sergio_hill,  # default 2
        decays=sergio_decay,  # default 0.8
        noise_params=sergio_noise_params,  # default 1.0
        cell_types=sergio_cell_types,  # default 10
        n_ko_genes=n_ko_genes)

    # extract data
    # obs
    x = jnp.array(data['x_obs'][:n_obs])
    x_ho = jnp.array(data['x_obs'][n_obs:(n_obs + n_obs_ho)])
    # interv
    x_interv_obs = jnp.array(data['x_obs'][-n_obs:])
    x_interv, x_ho_interv = jnp.split(data['x_int'], 2)
    envs, envs_ho = jnp.split(data['int_envs'], 2)
    x_interv_data = jnp.concatenate([x_interv_obs, x_interv])
    envs = jnp.concatenate([jnp.zeros(x_interv_obs.shape[0]),
                            envs]).astype(int)

    # ho interv
    x_ho_interv = jnp.array(x_ho_interv)
    envs_ho = jnp.array(envs_ho, dtype=int)
    interv_targets = data['interv_targets']

    # standardize
    mean = x_interv_data.mean(0)
    std = x_interv_data.std(0)

    x = (x - mean) / jnp.where(std == 0.0, 1.0, std)
    x_ho = (x_ho - mean) / jnp.where(std == 0.0, 1.0, std)
    x_interv_data = (x_interv_data - mean) / jnp.where(std == 0.0, 1.0, std)
    x_ho_interv = (x_ho_interv - mean) / jnp.where(std == 0.0, 1.0, std)

    target = Target(
        passed_key=passed_rng,
        graph_model=graph_model,
        n_vars=n_vars,
        n_observations=n_observations,
        n_ho_observations=n_ho_observations,
        g=g_gt_mat,
        theta=None,
        x=x,
        x_ho=x_ho,
        x_interv=[{} for _ in interv_targets],  # dummy
        x_interv_data=x_interv_data,
        interv_targets=interv_targets,
        envs=envs,
        x_ho_interv=[{} for _ in interv_targets],  # dummy
        x_ho_interv_data=x_ho_interv,
        interv_targets_ho=interv_targets,
        envs_ho=envs_ho,
        gt_posterior_obs=None,
        gt_posterior_interv=None)
    return target