import functools
import warnings
from jax.scipy.special import logsumexp
from jax.scipy.stats import beta as jax_beta
from jax import lax
import tqdm

import jax.numpy as jnp
from jax import jit, vmap, random, grad
from jax.experimental import optimizers
from jax.tree_util import tree_map, tree_multimap
from jax.nn import sigmoid
from bacadi.eval.target import make_graph_model, make_inference_model, make_kernel

from bacadi.inference.bacadi_base import BaCaDIBase

from bacadi.utils.func import bit2id, expand_by, id2bit


class BaCaDIJoint(BaCaDIBase):
    """
    This class implements BaCaDI: Bayesian Causal Discovery with Unknown Interventions (Hägele et al., 2021)
    instantiated using Stein Variational Gradient Descent (SVGD) (Liu and Wang, 2016) as the underlying inference method.

    This class implements //joint// inference of the posterior p(G, theta, I | D) 

    Args:
        kernel: string that specifies which kernel to use. one of (`frob-joint-add`, `frob-joint-mul`)
        graph_prior: string (`er` or `sf`) that specifies the graph model object (GraphDistribution) to be used that captures the prior over graphs
        inference_model: string (`lingauss` or `fcgauss`) that specifies the JAX-BN model object for inference, i.e. defines log_likelihood of data and parameters given the graph
        alpha_linear (float): inverse temperature parameter schedule of sigmoid
        beta_linear (float): inverse temperature parameter schedule of prior
        interv_per_env (int): prior belief over how many nodes are intervened upon in a given environment. defaults to 1
        h_latent (float): bandwidth parameter for the Frobenius kernel for Z
        h_theta (float): bandwidth parameter for the Frobenius kernel for Theta
        optimizer (dict): dictionary with at least keys `name` and `stepsize`
        n_grad_mc_samples (int): MC samples in gradient estimator for likelihood term p(theta, D | G)
        n_acyclicity_mc_samples (int): MC samples in gradient estimator for acyclicity constraint
        grad_estimator_z (str): gradient estimator d/dZ of expectation; choices: `score` or `reparam`
        score_function_baseline (float): weight of addition in score function baseline; == 0.0 corresponds to not using a baseline
        latent_prior_std (float): standard deviation of Gaussian prior over Z; defaults to 1/sqrt(k)
        model_param (dict): dictionary specifying model parameters. 
        random_state: prng key
        n_steps (int): number of steps to iterate SVGD
        n_particles (int): number of particles for SVGD
        callback_every (int): if == 0, `callback` is never called. 
        callback (func): function to be called every `callback_every` steps of SVGD.
    """

    def __init__(self,
                 *,
                 graph_prior,
                 model_prior,
                 alpha_linear,
                 kernel='frob-joint-interv-add',
                 edges_per_node=1,
                 interv_per_env=0,
                 beta_linear=1.0,
                 tau=1.0,
                 h_latent=5.,
                 h_interv=5.,
                 h_theta=500.,
                 lambda_regul=10,
                 optimizer=dict(name='rmsprop', stepsize=0.005),
                 n_grad_mc_samples=128,
                 n_acyclicity_mc_samples=32,
                 grad_estimator_z='reparam',
                 score_function_baseline=0.0,
                 latent_prior_std=None,
                 gamma_prior_std=None,
                 model_param=dict(obs_noise=0.1,
                                  mean_edge=0.,
                                  sig_edge=1.0,
                                  interv_mean=0.,
                                  interv_noise=0.),
                 random_state=None,
                 n_steps=1000,
                 n_particles=20,
                 callback_every=50,
                 callback=None,
                 verbose=False):
        super().__init__(kernel=kernel,
                         edges_per_node=edges_per_node,
                         graph_prior=graph_prior,
                         model_prior=model_prior,
                         alpha_linear=alpha_linear,
                         beta_linear=beta_linear,
                         tau=tau,
                         lambda_regul=lambda_regul,
                         optimizer=optimizer,
                         model_param=model_param,
                         n_grad_mc_samples=n_grad_mc_samples,
                         n_acyclicity_mc_samples=n_acyclicity_mc_samples,
                         grad_estimator_z=grad_estimator_z,
                         score_function_baseline=score_function_baseline,
                         latent_prior_std=latent_prior_std,
                         gamma_prior_std=gamma_prior_std,
                         random_state=random_state,
                         n_steps=n_steps,
                         n_particles=n_particles,
                         callback_every=callback_every,
                         callback=callback,
                         verbose=verbose)
        self.h_latent = h_latent
        self.h_theta = h_theta
        self.h_interv = h_interv
        # to be initialized when calling .fit(X)
        self.particles_theta = None
        self.interv_per_env = interv_per_env

    def fit(self, data, envs, known_interv_targets=None):
        """
        Perform inference of the posterior joint probability using SVGD.

        Args:
            data (array): data that contains the observations of the form: (n_obs, n_vars)
            envs (array): array that contains the environment id of every datapoint,
                          i.e. [e_1, ..., e_N], shape (n_obs,)
            known_interv_targets (dict): dictionary of the form {env_i: [i_1, ..., i_j]} to indicate
                                   knowledge of intervention targets in environment i

        Returns:
            self
        """

        if self.is_fitted_ and \
          jnp.array_equal(self.data, data) and not self.key_changed_:
            warnings.warn(
                'The data and initial random state after fitting `self` have not changed and nothing will be done.' + \
                'In order to refit, either change the random state via `.set_new_key()` or create a new object.'
            )
            return self

        self.data = data
        self.envs = envs
        self.key_changed_ = False
        self.is_fitted_ = False
        self.n_vars = self.data.shape[-1]
        self.n_env = self.envs.max().item() + 1

        self.graph_model = make_graph_model(graph_prior_str=self.graph_prior,
                                            n_vars=self.n_vars,
                                            edges_per_node=self.edges_per_node)

        self.model_param['graph_model'] = self.graph_model

        self.inference_model = make_inference_model(
            inference_str=self.inference_prior,
            n_vars=self.n_vars,
            **self.model_param)

        self.kernel_ = make_kernel(kernel=self.kernel,
                                   h_latent=self.h_latent,
                                   h_theta=self.h_theta,
                                   h_interv=self.h_interv)

        key, subk = random.split(self.random_state)
        init_particles_z, init_particles_theta, init_particles_gamma = self.sample_initial_random_particles(
            key=subk,
            n_particles=self.n_particles,
            n_vars=self.n_vars,
            n_env=self.n_env,
            known_interv_targets=known_interv_targets)

        # iteratively transport particles
        key, subk = random.split(key)
        self.particles_z, self.particles_theta, self.particles_gamma = self.sample_particles(
            data=self.data,
            key=subk,
            n_steps=self.n_steps,
            init_particles_z=init_particles_z,
            init_particles_theta=init_particles_theta,
            init_particles_gamma=init_particles_gamma,
            callback_every=self.callback_every,
            callback=self.callback)

        self.is_fitted_ = True
        self.key_changed_ = False
        return self

    def sample_initial_random_particles(self,
                                        *,
                                        key,
                                        n_particles,
                                        n_vars,
                                        n_env,
                                        known_interv_targets=None,
                                        n_dim=None):
        """
        Samples random particles to initialize SVGD

        Args:
            key: rng key
            n_particles: number of particles for SVGD
            n_dim: size of latent dimension `k`. Defaults to `n_vars`, s.t. k == d

        Returns:
            z: batch of latent tensors [n_particles, d, k, 2]
            theta: batch of parameters PyTree with leading dim `n_particles`
            gamma: batch of tensors [n_particles, n_env-1, d]
        
        """
        # default full rank
        if n_dim is None:
            n_dim = n_vars

        # std like Gaussian prior over Z
        std = self.latent_prior_std or (1.0 / jnp.sqrt(n_dim))

        # sample
        key, subk = random.split(key)
        z = random.normal(subk, shape=(n_particles, n_vars, n_dim, 2)) * std

        key, subk = random.split(key)
        theta = self.inference_model.init_interv_parameters(
            key=subk,
            n_env=n_env,
            n_particles=n_particles,
            n_vars=n_vars)

        # key, subk = random.split(key)
        # theta_I = random.normal(
        #     subk, shape=(n_particles, n_env - 1, n_dim, 2)) * jnp.sqrt(0.1)
        # theta_I = theta_I.at[..., 1].add(jnp.sqrt(0.1))
        # if self.inference_prior == 'lingauss':
        #     theta = [theta, theta_I]
        # else:
        #     theta.append(theta_I)

        if self.gamma_prior_std is None:
            self.gamma_prior_std = jnp.sqrt(0.1)
        key, subk = random.split(key)
        gamma = random.normal(
            subk, shape=(n_particles, n_env - 1, n_dim)) * self.gamma_prior_std

        return z, theta, gamma

    def f_kernel(self, x_latent, x_theta, x_interv, y_latent, y_theta,
                 y_interv, h_latent, h_theta, h_interv):
        """
        Evaluates kernel

        Args:
            x_latent: latent tensor [d, k, 2]
            x_theta: parameter PyTree 
            x_interv: latent tensor [n_env-1, d]
            y_latent: latent tensor [d, k, 2]
            y_theta: parameter PyTree 
            y_interv: latent tensor [n_env-1, d]
            h_latent (float): kernel bandwidth for Z term
            h_theta (float): kernel bandwidth for theta term
            h_interv (float): kernel bandwidth for gamma term
            t: step

        Returns:
            kernel value
        """
        return self.kernel_.eval(x_latent=x_latent,
                                 x_theta=x_theta,
                                 x_interv=x_interv,
                                 y_latent=y_latent,
                                 y_theta=y_theta,
                                 y_interv=y_interv,
                                 h_latent=h_latent,
                                 h_theta=h_theta,
                                 h_interv=h_interv)

    def f_kernel_mat(self, *args):
        """
        Computes pairwise kernel matrix

        Args:
            x_latents, x_thetas, x_interv:  latent tensor, PyTree and latent tensor
                                            with leading dimension A
            y_latents, y_thetas, y_interv:  latent tensor, PyTree and latent tensor
                                            with leading dimension B
            h_latent, h_theta, h_interv (floats): kernel bandwidths

        Returns:
            [A, B] kernel values
        """
        return vmap(vmap(self.f_kernel, (None, None, None, 0, 0, 0, None, None, None), 0), \
                                        (0, 0, 0, None, None, None, None, None, None), 0)  \
                                        (*args)

    def eltwise_grad_kernel_z(self, *args):
        """
        Computes gradient d/dz k((z, theta, gamma), (z', theta', gamma')) elementwise for each provided particle (z, theta, gamma)

        Args:
            x_latents, x_thetas, x_interv:  latent tensor, PyTree and latent tensor
                                            with leading dimension `n_particles`
            y_latents, y_thetas, y_interv:  single latent tensor, PyTree and latent tensor
            
            h_latent, h_theta, h_interv (floats): kernel bandwidths

        Returns:
            batch of gradients for latent tensors Z [n_particles, d, k, 2]
        
        """
        grad_kernel_z = grad(self.f_kernel, 0)
        return vmap(grad_kernel_z,
                    (0, 0, 0, None, None, None, None, None, None), 0)(*args)

    def eltwise_grad_kernel_theta(self, *args):
        """
        Computes gradient d/dtheta k((z, theta, gamma), (z', theta', gamma')) elementwise for each provided particle (z, theta, gamma)

        Args:
            x_latents, x_thetas, x_interv:  latent tensor, PyTree and latent tensor
                                            with leading dimension `n_particles`
            y_latents, y_thetas, y_interv:  single latent tensor, PyTree and latent tensor

            h_latent, h_theta, h_interv (floats): kernel bandwidths

        Returns:
            batch of gradients for parameters (PyTree with leading dim `n_particles`)
        """
        grad_kernel_theta = grad(self.f_kernel, 1)
        return vmap(grad_kernel_theta,
                    (0, 0, 0, None, None, None, None, None, None), 0)(*args)

    def eltwise_grad_kernel_gamma(self, *args):
        """
        Computes gradient d/dgamma k((z, theta, gamma), (z', theta', gamma')) elementwise for each provided particle (z, theta, gamma)

        Args:
            x_latents, x_thetas, x_interv:  latent tensor, PyTree and latent tensor
                                            with leading dimension `n_particles`
            y_latents, y_thetas, y_interv:  single latent tensor, PyTree and latent tensor

            h_latent, h_theta, h_interv (floats): kernel bandwidths

        Returns:
            batch of gradients for gamma tensors [n_particles, n_env-1, d]
        """
        grad_kernel_gamma = grad(self.f_kernel, 2)
        return vmap(grad_kernel_gamma,
                    (0, 0, 0, None, None, None, None, None, None), 0)(*args)

    def z_update(self, single_z, single_theta, single_gamma, kxx, z, theta,
                 gamma, grad_log_prob_z, h_latent, h_theta, h_interv):
        """
        Computes SVGD update for `single_z` of a (single_z, single_theta) tuple given the kernel values 
        `kxx` and the d/dz gradients of the target density for each of the available particles 

        Args:
            single_z: single latent tensor Z [d, k, 2], which is the Z particle being updated
            single_theta: single parameter PyTree, the theta particle of the Z particle being updated
            single_gamma: single gamma tensor [n_env, d], which is the gamma particle being updated
            kxx: pairwise kernel values for all particles [n_particles, n_particles]
            z:  all latent tensor Z particles [n_particles, d, k, 2] 
            theta: all theta particles as PyTree with leading dim `n_particles` 
            z:  all latent gamma particles [n_particles, n_env-1, d] 
            grad_log_prob_z: gradients of all Z particles w.r.t target density  [n_particles, d, k, 2]  

        Returns
            transform vector of shape [d, k, 2] for the Z particle being updated        
        """

        # compute terms in sum
        weighted_gradient_ascent = kxx[..., None, None, None] * grad_log_prob_z
        repulsion = self.eltwise_grad_kernel_z(z, theta, gamma, single_z,
                                               single_theta, single_gamma,
                                               h_latent, h_theta, h_interv)

        # average and negate (for optimizer)
        return -(weighted_gradient_ascent + repulsion).mean(axis=0)

    def parallel_update_z(self, *args):
        """
        Parallelizes `z_update` for all available particles
        Otherwise, same inputs as `z_update`.
        """
        return vmap(self.z_update,
                    (0, 0, 0, 1, None, None, None, None, None, None, None),
                    0)(*args)

    def gamma_update(self, single_z, single_theta, single_gamma, kxx, z, theta,
                     gamma, grad_log_prob_gamma, h_latent, h_theta, h_interv):
        """
        Computes SVGD update for `single_gamma` of a (single_z, single_theta, single_gamma)
        tuple given the kernel values  `kxx` and the d/dz gradients of the target density
        for each of the available particles 

        Args:
            single_z: single latent tensor Z [d, k, 2], which is the Z particle being updated
            single_theta: single parameter PyTree, the theta particle of the Z particle being updated
            single_gamma: single gamma tensor [n_env, d], which is the gamma particle being updated
            kxx: pairwise kernel values for all particles [n_particles, n_particles]
            z:  all latent tensor Z particles [n_particles, d, k, 2] 
            theta: all theta particles as PyTree with leading dim `n_particles` 
            z:  all latent gamma particles [n_particles, n_env-1, d] 
            grad_log_prob_gamma: gradients of all gamma particles w.r.t target density  [n_particles, n_env-1, d]  

        Returns
            transform vector of shape [n_env-1, d] for the gamma particle being updated        
        """

        # compute terms in sum
        weighted_gradient_ascent = kxx[..., None, None] * grad_log_prob_gamma
        repulsion = self.eltwise_grad_kernel_gamma(z, theta, gamma, single_z,
                                                   single_theta, single_gamma,
                                                   h_latent, h_theta, h_interv)

        # average and negate (for optimizer)
        return -(weighted_gradient_ascent + repulsion).mean(axis=0)

    def parallel_update_gamma(self, *args):
        """
        Parallelizes `gamma_update` for all available particles
        Otherwise, same inputs as `gamma_update`.
        """
        return vmap(self.gamma_update,
                    (0, 0, 0, 1, None, None, None, None, None, None, None),
                    0)(*args)

    def theta_update(self, single_z, single_theta, single_gamma, kxx, z, theta,
                     gamma, grad_log_prob_theta, h_latent, h_theta, h_interv):
        """
        Computes SVGD update for `single_theta` of a (single_z, single_theta) tuple given the kernel values 
        `kxx` and the d/dtheta gradients of the target density for each of the available particles 

        Args:
            single_z: single latent tensor Z [d, k, 2], the Z particle of the theta particle being updated
            single_theta: single parameter PyTree being updated
            kxx: pairwise kernel values for all particles [n_particles, n_particles]  
            z:  all latent tensor Z particles [n_particles, d, k, 2] 
            theta: all theta particles as PyTree with leading dim `n_particles` 
            grad_log_prob_theta: gradients of all theta particles w.r.t target density 
                PyTree with leading dim `n_particles

        Returns:
            transform vector PyTree with leading dim `n_particles` for the theta particle being updated   
        """

        # compute terms in sum
        weighted_gradient_ascent = tree_map(
            lambda leaf_theta_grad: expand_by(kxx, leaf_theta_grad.ndim - 1) *
            leaf_theta_grad, grad_log_prob_theta)

        repulsion = self.eltwise_grad_kernel_theta(z, theta, gamma, single_z,
                                                   single_theta, single_gamma,
                                                   h_latent, h_theta, h_interv)

        # average and negate (for optimizer)
        return tree_multimap(
            lambda grad_asc_leaf, repuls_leaf: -(grad_asc_leaf + repuls_leaf).
            mean(axis=0), weighted_gradient_ascent, repulsion)

    def parallel_update_theta(self, *args):
        """
        Parallelizes `theta_update` for all available particles
        Otherwise, same inputs as `theta_update`.
        """
        return vmap(self.theta_update,
                    (0, 0, 0, 1, None, None, None, None, None, None, None),
                    0)(*args)

    # this is the crucial @jit
    @functools.partial(jit, static_argnums=(0, ))
    def svgd_step(self, opt_state_z, opt_state_theta, opt_state_gamma, key, t,
                  sf_baseline):
        """
        Performs a single SVGD step, updating Z, theta, gamma jointly.
        
        Args:
            opt_state_z: optimizer state for latent Z particles; contains [n_particles, d, k, 2]
            opt_state_theta: optimizer state for theta particles; contains PyTree with `n_particles` leading dim
            key: prng key
            t: step
            sf_baseline: batch of baseline values in case score function gradient is used [n_particles, ]

        Returns:
            the updated inputs
        """

        z = self.get_params(opt_state_z)  # [n_particles, d, k, 2]
        theta = self.get_params(
            opt_state_theta)  # PyTree with `n_particles` leading dim
        gamma = self.get_params(opt_state_gamma)  # [n_particles, n_env-1, d]

        n_particles = z.shape[0]

        # make sure same bandwith is used for all calls to k(x, x') (in case e.g. the median heuristic is applied)
        h_latent = self.kernel_.h_latent
        h_theta = self.kernel_.h_theta
        h_interv = self.kernel_.h_interv

        # d/dtheta log p(theta, D | z, gamma)
        key, *batch_subk = random.split(key, n_particles + 1)
        dtheta_log_prob = self.eltwise_grad_theta_likelihood(
            z, theta, gamma, t, jnp.array(batch_subk))

        # d/dz log p(theta, D | z, gamma)
        key, *batch_subk = random.split(key, n_particles + 1)
        dz_log_likelihood, sf_baseline = self.eltwise_grad_z_likelihood(
            z, theta, gamma, sf_baseline, t, jnp.array(batch_subk))

        # d/dz log p(z) (acyclicity)
        key, *batch_subk = random.split(key, n_particles + 1)
        dz_log_prior = self.eltwise_grad_latent_prior(z, jnp.array(batch_subk),
                                                      t)

        # d/dz log p(z, theta, gamma, D) = d/dz log p(z)  + log p(theta, D | z, gamma)
        dz_log_prob = dz_log_prior + dz_log_likelihood

        # d/dgamma log p(theta, D | z, gamma)
        key, *batch_subk = random.split(key, n_particles + 1)
        dgamma_log_likelihood, _ = self.eltwise_grad_gamma_likelihood(
            z, theta, gamma, sf_baseline, t, jnp.array(batch_subk))

        # d/dgamma log p(gamma) (prior for sparse interventions)
        key, *batch_subk = random.split(key, n_particles + 1)
        dgamma_log_prior = self.eltwise_grad_gamma_prior(
            gamma, jnp.array(batch_subk), t)

        # d/dgamma log p(z, theta, gamma, D) = d/dgamma log p(gamma)  + log p(theta, D | z, gamma)
        dgamma_log_prob = dgamma_log_prior + dgamma_log_likelihood

        # k((z, theta), (z, theta)) for all particles
        kxx = self.f_kernel_mat(z, theta, gamma, z, theta, gamma, h_latent,
                                h_theta, h_interv)

        # transformation phi() applied in batch to each particle individually
        phi_z = self.parallel_update_z(z, theta, gamma, kxx, z, theta, gamma,
                                       dz_log_prob, h_latent, h_theta,
                                       h_interv)
        phi_theta = self.parallel_update_theta(z, theta, gamma, kxx, z, theta,
                                               gamma, dtheta_log_prob,
                                               h_latent, h_theta, h_interv)
        phi_gamma = self.parallel_update_gamma(z, theta, gamma, kxx, z, theta,
                                               gamma, dgamma_log_prob,
                                               h_latent, h_theta, h_interv)

        # apply transformation
        # `x += stepsize * phi`; the phi returned is negated for SVGD
        opt_state_z = self.opt_update(t, phi_z, opt_state_z)
        opt_state_theta = self.opt_update(t, phi_theta, opt_state_theta)
        opt_state_gamma = self.opt_update(t, phi_gamma, opt_state_gamma)

        # opt_state_gamma = lax.cond(
        #     t % 20 != 0, lambda _: opt_state_gamma,
        #     lambda _: self.opt_update(t, phi_gamma, opt_state_gamma), None)
        return opt_state_z, opt_state_theta, opt_state_gamma, key, sf_baseline

    def sample_particles(self,
                         *,
                         data,
                         n_steps,
                         init_particles_z,
                         init_particles_theta,
                         init_particles_gamma,
                         key,
                         callback=None,
                         callback_every=0):
        """
        Deterministically transforms particles to minimize KL to target using SVGD

        Arguments:
            data: collection of datasets we perform inference for [n_datasets, n_obs, d]
            n_steps (int): number of SVGD steps performed
            init_particles_z: batch of initialized latent tensor particles [n_particles, d, k, 2]
            init_particles_theta:  batch of parameters PyTree (i.e. for a general parameter set shape) 
                with leading dimension `n_particles`
            init_particles_gamma: batch of initialized gamma tensor particles [n_particles, n_env-1, d]
            key: prng key
            callback: function to be called every `callback_every` steps of SVGD.
            callback_every: if == 0, `callback` is never called. 

        Returns: 
            `n_particles` samples that approximate the BaCaDI target density
            particles_z: [n_particles, d, k, 2]
            particles_theta: PyTree of parameters with leading dimension `n_particles`
           
        """
        self.data = data

        z = init_particles_z
        theta = init_particles_theta
        gamma = init_particles_gamma

        # initialize score function baseline (one for each particle)
        n_particles, _, n_dim, _ = z.shape
        sf_baseline = jnp.zeros(n_particles)

        if self.latent_prior_std is None:
            self.latent_prior_std = 1.0 / jnp.sqrt(n_dim)

        # init optimizer
        if self.optimizer['name'] == 'gd':
            opt = optimizers.sgd(self.optimizer['stepsize'] /
                                 10.0)  # comparable scale for tuning
        elif self.optimizer['name'] == 'momentum':
            opt = optimizers.momentum(self.optimizer['stepsize'])
        elif self.optimizer['name'] == 'adagrad':
            opt = optimizers.adagrad(self.optimizer['stepsize'])
        elif self.optimizer['name'] == 'adam':
            opt = optimizers.adam(self.optimizer['stepsize'])
        elif self.optimizer['name'] == 'rmsprop':
            opt = optimizers.rmsprop(self.optimizer['stepsize'])
        else:
            raise ValueError()

        opt_init, self.opt_update, get_params = opt
        self.get_params = jit(get_params)
        opt_state_z = opt_init(z)
        opt_state_theta = opt_init(theta)
        opt_state_gamma = opt_init(gamma)
        """Execute particle update steps for all particles in parallel using `vmap` functions"""
        it = tqdm.tqdm(range(n_steps), desc='BaCaDI', disable=not self.verbose)
        for t in it:

            # perform one SVGD step (compiled with @jit)
            opt_state_z, opt_state_theta, opt_state_gamma, key, sf_baseline = self.svgd_step(
                opt_state_z, opt_state_theta, opt_state_gamma, key, t,
                sf_baseline)

            # callback
            if callback and callback_every and (((t + 1) % callback_every == 0)
                                                or (t == (n_steps - 1))):
                z = self.get_params(opt_state_z)
                theta = self.get_params(opt_state_theta)
                gamma = self.get_params(opt_state_gamma)
                self.particles_z = z
                self.particles_theta = theta
                self.particles_gamma = gamma
                callback(model=self,
                         t=t,
                         zs=z,
                         thetas=theta,
                         grad_dict=self.grad_dict if hasattr(
                             self, 'grad_dict') else None,
                         gamma=gamma)

        # return transported particles
        z_final = self.get_params(opt_state_z)
        theta_final = self.get_params(opt_state_theta)
        gamma_final = self.get_params(opt_state_gamma)
        return z_final, theta_final, gamma_final

    def particle_empirical(self):
        """
        Returns the standardized form of a particle distribution 
        represented by this fitted object. 
        Converts the batch z particles into the binary adjacency matrices
        (for alpha -> inf) and returns it with associated parameters and
        empirical log probabilities (here uniform) as a tuple.

        Returns:
            tuple: 
                tuple[0] contains unique ids as by `bit2id`
                tuple[1] contains thetas
                tuple[2] contains intervention targets
                tuple[3] contains the empirical log probability (here uniform)

        """
        if self.particles_z is None:
            raise AttributeError(
                'The model has not been fitted yet. Call .fit(X) first to obtain particles.'
            )
        particles_g = self.particle_to_g_lim(self.particles_z)
        particles_I = self.particle_to_interv_lim(self.particles_gamma)
        ids = bit2id(particles_g)

        # empirical
        log_probs = -jnp.log(self.n_particles) * jnp.ones(self.n_particles)

        return ids, self.particles_theta, particles_I, log_probs

    def particle_mixture(self):
        """
        Returns the standardized form of a particle distribution weighted by
        the likelihood represented by this fitted object. 
        Converts the batch z particles into the binary adjacency matrices
        (for alpha -> inf) and returns it with associated parameters and
        empirical log probabilities as a tuple.

        Returns:
            tuple: 
                tuple[0] contains unique ids as by `bit2id`
                tuple[1] contains thetastuple[2] contains intervention targets
                tuple[3] contains the empirical log probability

        """
        if self.particles_z is None:
            raise AttributeError(
                'The model has not been fitted yet. Call .fit(X) first to obtain particles.'
            )

        particles_g = self.particle_to_g_lim(self.particles_z)
        particles_I = self.particle_to_interv_lim(self.particles_gamma)
        ids = bit2id(particles_g)

        eltwise_log_prob = jit(
            vmap(
                lambda g, theta, i_targets: self.target_log_joint_prob(
                    g, theta, i_targets, None), (0, 0, 0), 0))
        # mixture using relative log probs
        # assumes that every particle is unique (always true because of theta)
        log_probs = eltwise_log_prob(particles_g, self.particles_theta,
                                     particles_I)
        log_probs -= logsumexp(log_probs)

        return ids, self.particles_theta, particles_I, log_probs

    def get_particles(self):  # TODO
        """
        Get the particles that have been fit to approximate the joint posterior probability.
        Raises an error when called before model is fitted.

        Args: None

        Returns:
            particles_z: (n_particles, n_vars, n_dim, 2)
            particles_theta: PyTree with leading dim `n_particles` 
        """
        if self.particles_z is None:
            raise AttributeError(
                'The model has not been fitted yet. Call .fit(X) first to obtain particles.'
            )

        return self.particles_z, self.particles_theta, self.particles_gamma
