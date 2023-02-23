import os
from jax import random
import jax.numpy as jnp

from bacadi.inference.joint_dibs_svgd import JointDiBS
from bacadi.inference.bacadi_joint import BaCaDIJoint
from bacadi.inference.marginal_dibs_svgd import MarginalDiBS
from bacadi.inference.bacadi_marginal import BaCaDIMarginal
from bacadi.eval.target import load_pickle, make_graph_model, make_linear_gaussian_model, make_nonlinear_gaussian_model, make_sergio, make_synthetic_bayes_net, options_to_str, save_pickle

from bacadi.models.linearGaussian import LinearGaussian
from bacadi.models.nonlinearGaussian import DenseNonlinearGaussianJAX
from bacadi.models.sobolev import SobolevGaussianJAX
from bacadi.utils.graph import adjmat_to_str


def linear_chain(n_vars):
    return [[1. if i == j - 1 else 0. for j in range(n_vars)]
            for i in range(n_vars)]


def tree_graph(n_vars):
    return [[
        1. if j == 2 * i + 1 or j == 2 * i + 2 else 0. for j in range(n_vars)
    ] for i in range(n_vars)]


def no_intervention_targets(n_vars):
    return jnp.zeros(n_vars).astype(bool)


DIBS_PAPER_EXAMPLE = [
    [0., 2., -2., 0.],
    [0., 0., 0., 3.],
    [0., 0., 0., 1.],
    [0., 0., 0., 0.],
]


def make_target(config, load=True):
    """
    Create synthetic Bayesian network
    TODO proper docstring
    """
    n_vars = config.n_vars

    key = random.PRNGKey(config.seed)

    filename = config.simulator
    if config.simulator == 'synthetic':
        filename += f'_{config.joint_inference_model}'
    filename += '_' + \
            options_to_str(
                d=n_vars,
                graph_type=config.target_graph,
                graph=config.graph_prior,
                edges_per_node=config.graph_prior_edges_per_node,
                n_obs=config.n_observations,
                n_interv_obs=config.n_interv_obs,
                n_ho=config.n_ho_observations,
                interventions=config.intervention_type,
                intervention_val=config.intervention_val,
                intervention_noise=config.intervention_noise,
                seed=config.seed,
            )
    if load:
        try:
            target = load_pickle(filename)

            print(
                'Loaded {}-vars, {}-edge graph with {} generative model:\t{}'.
                format(
                    target.n_vars,
                    jnp.sum(target.g).item(), config.joint_inference_model
                    if config.simulator == 'synthetic' else 'sergio',
                    adjmat_to_str(target.g)))
            return target, filename

        except FileNotFoundError:
            print('Loading failed: ' + filename + ' does not exist.')
            print('Generating from scratch...')
    ############################ SERGIO GRN
    if config.simulator == 'sergio':
        target = make_sergio(
            n_vars,
            seed=config.seed,
            sergio_hill=config.sergio_hill,
            sergio_decay=config.sergio_decay,
            sergio_noise_params=config.sergio_noise_params,
            sergio_cell_types=config.sergio_cell_types,
            sergio_k_lower_lim=config.sergio_k_lower_lim,
            sergio_k_upper_lim=config.sergio_k_upper_lim,
            graph_prior_edges_per_node=config.graph_prior_edges_per_node,
            n_observations=config.n_observations,
            n_ho_observations=config.n_ho_observations,
            n_intervention_sets=config.n_intervention_sets,
            n_interv_obs=config.n_interv_obs)
        save_pickle(target, filename)
        print(f"Wrote target file to {filename}.")
        return target, filename

    ############## SYNTHETIC RANDOM
    if config.target_graph == "random":
        if config.joint and config.joint_inference_model != "lingauss":
            if config.joint_inference_model == 'fcgauss':
                shared_params = {
                    'obs_noise': config.fcgauss_obs_noise,
                    'random_noise': config.fcgauss_random_noise,
                    'sig_param': config.fcgauss_sig_param,
                    'init_param': config.fcgauss_init_param,
                    'init_sig_param': config.fcgauss_init_sig_param,
                }
            else:
                shared_params = {
                    'obs_noise': config.sobolevgauss_obs_noise,
                    'random_noise': config.sobolevgauss_random_noise,
                    'sig_param': config.sobolevgauss_sig_param,
                    'init_param': config.sobolevgauss_init_param,
                    'init_sig_param': config.sobolevgauss_init_sig_param,
                }

            target = make_nonlinear_gaussian_model(
                model=config.joint_inference_model,
                key=key,
                n_vars=n_vars,
                graph_prior_str=config.graph_prior,
                # fcgauss
                hidden_layers=(config.fcgauss_n_neurons, ) *
                config.fcgauss_hidden_layers,
                activation=config.fcgauss_activation,
                # sobolevgauss
                n_exp=config.sobolevgauss_n_exp,
                mean_param=config.sobolevgauss_mean_param,
                # rest
                n_observations=config.n_observations,
                n_ho_observations=config.n_ho_observations,
                n_intervention_sets=config.n_intervention_sets,
                n_interv_obs=config.n_interv_obs,
                perc_intervened=config.perc_intervened,
                edges_per_node=config.graph_prior_edges_per_node,
                interventions=config.intervention_type,
                intervention_val=config.intervention_val,
                intervention_noise=config.intervention_noise,
                **shared_params)
        else:
            target = make_linear_gaussian_model(
                key=key,
                n_vars=n_vars,
                graph_prior_str=config.graph_prior,
                inference=config.inference_model,
                obs_noise=config.lingauss_obs_noise,
                random_noise=config.lingauss_random_noise,
                mean_edge=config.lingauss_mean_edge,
                sig_edge=config.lingauss_sig_edge,
                init_sig_edge=config.lingauss_init_sig_edge,
                alpha_mu=config.bge_alpha_mu,
                alpha_lambd_add=config.bge_alpha_lambd_add,
                n_observations=config.n_observations,
                n_ho_observations=config.n_ho_observations,
                n_intervention_sets=config.n_intervention_sets,
                n_interv_obs=config.n_interv_obs,
                perc_intervened=config.perc_intervened,
                edges_per_node=config.graph_prior_edges_per_node,
                interventions=config.intervention_type,
                intervention_val=config.intervention_val,
                intervention_noise=config.intervention_noise)

        save_pickle(target, filename)
        print(f"Wrote target file to {filename}.")
        return target, filename
    ############## SYNTHETIC SPECIFIC GRAPH
    # -- reach below here only if target graph is not random
    # -- or loading failed
    if config.target_graph == "linear_chain":
        adj_weight = jnp.array(linear_chain(n_vars))
    elif config.target_graph == "binary_tree":
        adj_weight = jnp.array(tree_graph(n_vars))
    else:
        adj_weight = jnp.array(DIBS_PAPER_EXAMPLE)
        n_vars = 4
        config.n_vars = 4

    target_g = (adj_weight != 0).astype(float)

    graph_model = make_graph_model(
        n_vars=n_vars,
        graph_prior_str=config.graph_prior,
        edges_per_node=config.graph_prior_edges_per_node)

    if config.joint and config.joint_inference_model == "fcgauss":
        if config.fcgauss_random_noise:
            key, subk = random.split(key)
            obs_noise = config.fcgauss_obs_noise * random.uniform(
                subk, shape=(n_vars, )) + config.fcgauss_obs_noise / 2
        else:
            obs_noise = config.fcgauss_obs_noise
        generative_model = DenseNonlinearGaussianJAX(
            obs_noise=obs_noise,
            sig_param=config.fcgauss_sig_param,
            init_sig_param=config.fcgauss_sig_param,
            hidden_layers=(config.bacadi_fcgauss_n_neurons, ) *
            config.bacadi_fcgauss_hidden_layers,
            activation=config.fcgauss_activation,
            init=config.fcgauss_init_param)
    elif config.joint and config.joint_inference_model == 'sobolevgauss':
        if config.fcgauss_random_noise:
            key, subk = random.split(key)
            obs_noise = config.sobolevgauss_obs_noise * random.uniform(
                subk, shape=(n_vars, )) + config.sobolevgauss_obs_noise / 2
        else:
            obs_noise = config.sobolevgauss_obs_noise

        generative_model = SobolevGaussianJAX(
            obs_noise=obs_noise,
            n_vars=n_vars,
            n_exp=config.sobolevgauss_n_exp,
            mean_param=config.sobolevgauss_mean_param,
            sig_param=config.sobolevgauss_sig_param,
            init_sig_param=config.sobolevgauss_init_sig_param,
            init=config.sobolevgauss_init)
    else:
        if config.lingauss_random_noise:
            key, subk = random.split(key)
            obs_noise = config.lingauss_obs_noise * random.uniform(
                subk, shape=(n_vars, )) + config.lingauss_obs_noise / 2
        else:
            obs_noise = config.lingauss_obs_noise
        generative_model = LinearGaussian(g_dist=graph_model,
                                          mean_edge=config.lingauss_mean_edge,
                                          sig_edge=config.lingauss_sig_edge,
                                          obs_noise=obs_noise)

    # inference_str = config.joint_inference_model if config.joint else config.inference_model
    # inference_model = make_inference_model(
    #     inference_str=inference_str,
    #     n_vars=config.n_vars,
    #     obs_noise=config.lingauss_obs_noise
    #     if inference_str == 'lingauss' else config.fcgauss_obs_noise,
    #     mean_edge=config.lingauss_mean_edge,
    #     sig_edge=config.lingauss_sig_edge,
    #     init_sig_edge=config.lingauss_init_sig_edge,
    #     sig_param=config.fcgauss_sig_param,
    #     init_sig_param=config.fcgauss_init_sig_param,
    #     hidden_layers=(config.bacadi_fcgauss_n_neurons, ) *
    #     config.bacadi_fcgauss_hidden_layers,
    #     alpha_mu=config.bge_alpha_mu,
    #     alpha_lambd=n_vars + config.bge_alpha_lambd_add,
    # )
    inference_model = None

    if config.random_theta or (config.joint
                               and config.joint_inference_model == "fcgauss"):
        theta = None
    else:
        theta = adj_weight
    target = make_synthetic_bayes_net(
        key=key,
        n_vars=n_vars,
        graph_model=graph_model,
        generative_model=generative_model,
        inference_model=inference_model,
        n_observations=config.n_observations,
        n_interv_obs=config.n_interv_obs,
        n_ho_observations=config.n_ho_observations,
        n_intervention_sets=config.n_intervention_sets,
        perc_intervened=config.perc_intervened,
        theta=theta,
        g_gt_mat=target_g,
        interventions=config.intervention_type,
        intervention_val=config.intervention_val,
        intervention_noise=config.intervention_noise,
        get_posteriors=n_vars <= 5 and not config.joint)

    save_pickle(target, filename)
    print(f"Wrote target file to {filename}.")
    return target, filename


def make_nointerv_bacadi(config, model_param, callback, key):
    # no interventions reduces to DiBS
    if config.joint:
        dibs = JointDiBS(
            random_state=key,
            model_param=model_param,
            kernel=config.joint_bacadi_kernel,
            graph_prior=config.joint_bacadi_graph_prior,
            edges_per_node=config.joint_bacadi_graph_prior_edges_per_node,
            model_prior=config.joint_bacadi_inference_model,
            alpha_linear=config.joint_bacadi_alpha_linear,
            beta_linear=config.joint_bacadi_beta_linear,
            tau=config.joint_bacadi_tau_linear,
            h_latent=config.joint_bacadi_h_latent,
            h_theta=config.joint_bacadi_h_theta,
            optimizer=dict(name=config.bacadi_optimizer,
                           stepsize=config.bacadi_optimizer_stepsize),
            n_grad_mc_samples=config.joint_bacadi_n_grad_mc_samples,
            n_acyclicity_mc_samples=config.joint_bacadi_n_acyclicity_mc_samples,
            grad_estimator_z=config.joint_bacadi_grad_estimator_z,
            score_function_baseline=config.joint_bacadi_score_function_baseline,
            n_steps=config.joint_bacadi_n_steps,
            n_particles=config.n_particles,
            callback_every=config.callback_every,
            callback=callback if config.callback else None,
            verbose=config.verbose,
        )
    else:
        dibs = MarginalDiBS(
            random_state=key,
            model_param=model_param,
            kernel=config.bacadi_kernel,
            graph_prior=config.bacadi_graph_prior,
            edges_per_node=config.bacadi_graph_prior_edges_per_node,
            model_prior=config.bacadi_inference_model,
            alpha_linear=config.bacadi_alpha_linear,
            beta_linear=config.bacadi_beta_linear,
            tau=config.bacadi_tau_linear,
            h_latent=config.bacadi_h_latent,
            optimizer=dict(name=config.bacadi_optimizer,
                           stepsize=config.bacadi_optimizer_stepsize),
            n_grad_mc_samples=config.bacadi_n_grad_mc_samples,
            n_acyclicity_mc_samples=config.bacadi_n_acyclicity_mc_samples,
            grad_estimator_z=config.bacadi_grad_estimator_z,
            score_function_baseline=config.bacadi_score_function_baseline,
            n_steps=config.bacadi_n_steps,
            n_particles=config.n_particles,
            callback_every=config.callback_every,
            callback=callback if config.callback else None,
            verbose=config.verbose,
        )

    return dibs


def make_interv_bacadi(config, model_param, callback, key):
    if config.joint:
        bacadi = BaCaDIJoint(
            random_state=key,
            model_param=model_param,
            kernel=config.joint_bacadi_interv_kernel,
            graph_prior=config.joint_bacadi_graph_prior,
            edges_per_node=config.joint_bacadi_graph_prior_edges_per_node,
            model_prior=config.joint_bacadi_inference_model,
            alpha_linear=config.joint_bacadi_alpha_linear,
            beta_linear=config.joint_bacadi_beta_linear,
            tau=config.joint_bacadi_tau_linear,
            h_latent=config.joint_bacadi_h_latent,
            h_theta=config.joint_bacadi_h_theta,
            h_interv=config.joint_bacadi_h_interv,
            interv_per_env=config.bacadi_interv_per_env,
            lambda_regul=config.bacadi_lambda_regul,
            optimizer=dict(name=config.bacadi_optimizer,
                           stepsize=config.bacadi_optimizer_stepsize),
            n_grad_mc_samples=config.joint_bacadi_n_grad_mc_samples,
            n_acyclicity_mc_samples=config.joint_bacadi_n_acyclicity_mc_samples,
            grad_estimator_z=config.joint_bacadi_grad_estimator_z,
            score_function_baseline=config.joint_bacadi_score_function_baseline,
            n_steps=config.joint_bacadi_n_steps,
            n_particles=config.n_particles,
            callback_every=config.callback_every,
            callback=callback if config.callback else None,
            verbose=config.verbose,
        )
    else:
        bacadi = BaCaDIMarginal(
            random_state=key,
            model_param=model_param,
            kernel=config.bacadi_interv_kernel,
            graph_prior=config.bacadi_graph_prior,
            edges_per_node=config.bacadi_graph_prior_edges_per_node,
            model_prior=config.bacadi_inference_model,
            alpha_linear=config.bacadi_alpha_linear,
            beta_linear=config.bacadi_beta_linear,
            tau=config.bacadi_tau_linear,
            h_latent=config.bacadi_h_latent,
            h_interv=config.bacadi_h_interv,
            interv_per_env=config.bacadi_interv_per_env,
            lambda_regul=config.bacadi_lambda_regul,
            optimizer=dict(name=config.bacadi_optimizer,
                           stepsize=config.bacadi_optimizer_stepsize),
            n_grad_mc_samples=config.bacadi_n_grad_mc_samples,
            n_acyclicity_mc_samples=config.bacadi_n_acyclicity_mc_samples,
            grad_estimator_z=config.bacadi_grad_estimator_z,
            score_function_baseline=config.bacadi_score_function_baseline,
            n_steps=config.bacadi_n_steps,
            n_particles=config.n_particles,
            callback_every=config.callback_every,
            callback=callback if config.callback else None,
            verbose=config.verbose,
        )

    return bacadi


def make_bacadi(target, config, callback, key):
    if config.joint:
        if config.joint_bacadi_inference_model == "lingauss":
            model_param = dict(
                obs_noise=config.bacadi_lingauss_obs_noise,
                mean_edge=config.bacadi_lingauss_mean_edge,
                sig_edge=config.bacadi_lingauss_sig_edge,
                init_sig_edge=config.bacadi_lingauss_init_sig_edge,
                interv_mean=config.intervention_val,
                interv_noise=config.intervention_noise,
                interv_prior_mean=config.interv_prior_mean,
                interv_prior_std=config.interv_prior_std)
        elif config.joint_bacadi_inference_model == 'fcgauss':
            model_param = dict(
                obs_noise=config.bacadi_fcgauss_obs_noise,
                sig_param=config.bacadi_fcgauss_sig_param,
                init_sig_param=config.bacadi_fcgauss_init_sig_param,
                hidden_layers=(config.bacadi_fcgauss_n_neurons, ) *
                config.bacadi_fcgauss_hidden_layers,
                init=config.bacadi_fcgauss_init_param,
                activation=config.bacadi_fcgauss_activation,
                interv_mean=config.intervention_val,
                interv_noise=config.intervention_noise,
                interv_prior_mean=config.interv_prior_mean,
                interv_prior_std=config.interv_prior_std)
        elif config.joint_bacadi_inference_model == 'sobolevgauss':
            model_param = dict(
                obs_noise=config.bacadi_sobolevgauss_obs_noise,
                sig_param=config.bacadi_sobolevgauss_sig_param,
                mean_param=config.bacadi_sobolevgauss_mean_param,
                n_vars=config.n_vars,
                n_exp=config.bacadi_sobolevgauss_n_exp,
                init_sig_param=config.bacadi_sobolevgauss_init_sig_param,
                init=config.bacadi_sobolevgauss_init_param,
                interv_mean=config.intervention_val,
                interv_noise=config.intervention_noise,
                interv_prior_mean=config.interv_prior_mean,
                interv_prior_std=config.interv_prior_std)
    else:
        if config.simulator == 'synthetic':
            # for the BGe prior mean vector over interventions, take the mean of the intervention
            # values where node i has been intervened on
            interv_means = jnp.zeros((len(target.x_interv) - 1, config.n_vars),
                                     dtype=float)
            for i, interv_set in enumerate(
                    target.x_interv[1:]):  # assume first is observational
                for k, v in interv_set[0].items():
                    if k in range(config.n_vars):  # intv on k in i-th env
                        if type(v) == dict:  # mean, noise dict
                            interv_means = interv_means.at[i, k].set(v['mean'])
                        else:  # direct value
                            interv_means = interv_means.at[i, k].set(v)
            interv_means = jnp.where(
                (interv_means != 0).sum(0) != 0,
                interv_means.sum(0) / (interv_means != 0).sum(0), -1e8)
        else:  # for sergio, take the argument
            interv_means = float(config.intervention_val)
        model_param = dict(alpha_mu=config.bacadi_bge_alpha_mu,
                           alpha_lambd=config.n_vars +
                           config.bacadi_bge_alpha_lambd_add,
                           interv_mean=interv_means,
                           interv_noise=config.intervention_noise)
    if config.infer_interv:
        return make_interv_bacadi(config, model_param, callback, key)
    else:
        return make_nointerv_bacadi(config, model_param, callback, key)