import functools
import warnings
import jax
from jax._src.scipy.special import logsumexp
import tqdm

import jax.numpy as jnp
from jax import jit, vmap, random, grad
from jax.experimental import optimizers
from jax.tree_util import tree_map, tree_multimap
from bacadi.eval.target import make_graph_model, make_inference_model, make_kernel

from bacadi.inference.dibs import DiBS

from bacadi.utils.func import bit2id, expand_by


class JointDiBS(DiBS):
    """
    This class implements DiBS: Differentiable Bayesian Structure Learning (Lorch et al., 2021)
    instantiated using Stein Variational Gradient Descent (SVGD) (Liu and Wang, 2016) as the underlying inference method.
    An SVGD update of vector v is defined as

        phi(v) = 1/n_particles sum_u k(v, u) d/du log p(u) + d/du k(u, v)
        
    This class implements //joint// inference of the posterior p(G, theta | D).
    For marginal inference of p(G | D), use the class `MarginalDiBS`

    Args:
        kernel: string that specifies which kernel to use. one of (`frob-joint-add`, `frob-joint-mul`)
        graph_prior: string (`er` or `sf`) that specifies the graph model object (GraphDistribution) to be used that captures the prior over graphs
        inference_model: string (`lingauss` or `fcgauss`) that specifies the JAX-BN model object for inference, i.e. defines log_likelihood of data and parameters given the graph
        alpha_linear (float): inverse temperature parameter schedule of sigmoid
        beta_linear (float): inverse temperature parameter schedule of prior
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
                 kernel,
                 graph_prior,
                 model_prior,
                 alpha_linear,
                 edges_per_node=1,
                 beta_linear=1.0,
                 tau=1.0,
                 h_latent=5.,
                 h_theta=500.,
                 optimizer=dict(name='rmsprop', stepsize=0.005),
                 n_grad_mc_samples=128,
                 n_acyclicity_mc_samples=32,
                 grad_estimator_z='reparam',
                 score_function_baseline=0.0,
                 latent_prior_std=None,
                 model_param=dict(
                     obs_noise=0.1,
                     mean_edge=0.,
                     sig_edge=1.0,
                 ),
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
                         optimizer=optimizer,
                         model_param=model_param,
                         n_grad_mc_samples=n_grad_mc_samples,
                         n_acyclicity_mc_samples=n_acyclicity_mc_samples,
                         grad_estimator_z=grad_estimator_z,
                         score_function_baseline=score_function_baseline,
                         latent_prior_std=latent_prior_std,
                         random_state=random_state,
                         n_steps=n_steps,
                         n_particles=n_particles,
                         callback_every=callback_every,
                         callback=callback,
                         verbose=verbose)
        self.h_latent = h_latent
        self.h_theta = h_theta
        # to be initialized when calling .fit(X)
        self.particles_theta = None

    def fit(self, data, interv_targets=None, envs=None):
        """
        Perform inference of the posterior joint probability using SVGD.

        Args:
            data: observations of the shape [n_samples, n_vars]
            interv_targets: boolean mask indicating interventions of the form [n_samples, n_vars]
                            or None if only observational data
        Returns:
            self
        """
        self.data, self.interv_targets, self.envs = data, interv_targets, envs

        self.key_changed_ = False
        self.is_fitted_ = False
        self.n_vars = self.data.shape[-1]

        self.graph_model = make_graph_model(graph_prior_str=self.graph_prior,
                                            n_vars=self.n_vars,
                                            edges_per_node=self.edges_per_node)

        self.model_param['graph_model'] = self.graph_model

        self.inference_model = make_inference_model(
            inference_str=self.inference_prior, 
            n_vars=self.n_vars, **self.model_param)

        self.kernel_ = make_kernel(kernel=self.kernel,
                                   h_latent=self.h_latent,
                                   h_theta=self.h_theta)

        key, subk = random.split(self.random_state)
        init_particles_z, init_particles_theta = self.sample_initial_random_particles(
            key=subk, n_particles=self.n_particles, n_vars=self.n_vars)

        # iteratively transport particles
        key, subk = random.split(key)
        self.particles_z, self.particles_theta = self.sample_particles(
            interv_targets=self.interv_targets,
            key=subk,
            n_steps=self.n_steps,
            init_particles_z=init_particles_z,
            init_particles_theta=init_particles_theta,
            callback_every=self.callback_every,
            callback=self.callback)

        self.is_fitted_ = True
        self.key_changed_ = False
        return self


    def sample_initial_random_particles(self, *, key, n_particles, n_vars, n_dim=None):
        """
        Samples random particles to initialize SVGD

        Args:
            key: rng key
            n_particles: number of particles for SVGD
            n_particles: number of variables `d` in inferred BN
            n_dim: size of latent dimension `k`. Defaults to `n_vars`, s.t. k == d

        Returns:
            z: batch of latent tensors [n_particles, d, k, 2]
            theta: batch of parameters PyTree with leading dim `n_particles`
        
        """
        # default full rank
        if n_dim is None:
            n_dim = n_vars 
        
        # std like Gaussian prior over Z           
        std = self.latent_prior_std  or (1.0 / jnp.sqrt(n_dim))

        # sample
        key, subk = random.split(key)
        z = random.normal(subk, shape=(n_particles, n_vars, n_dim, 2)) * std

        key, subk = random.split(key)
        theta = self.inference_model.init_parameters(key=subk, n_particles=n_particles, n_vars=n_vars)

        return z, theta


    def f_kernel(self, x_latent, x_theta, y_latent, y_theta, h_latent, h_theta, t):
        """
        Evaluates kernel

        Args:
            x_latent: latent tensor [d, k, 2]
            x_theta: parameter PyTree 
            y_latent: latent tensor [d, k, 2]
            y_theta: parameter PyTree 
            h_latent (float): kernel bandwidth for Z term
            h_theta (float): kernel bandwidth for theta term
            t: step

        Returns:
            kernel value
        """
        return self.kernel_.eval(
            x_latent=x_latent, x_theta=x_theta,
            y_latent=y_latent, y_theta=y_theta,
            h_latent=h_latent, h_theta=h_theta)


    def f_kernel_mat(self, x_latents, x_thetas, y_latents, y_thetas, h_latent, h_theta, t):
        """
        Computes pairwise kernel matrix

        Args:
            x_latents: latent tensor [A, d, k, 2]
            x_thetas: parameter PyTree with batch size A as leading dim
            y_latents: latent tensor [B, d, k, 2]
            y_thetas: parameter PyTree with batch size B as leading dim
            h_latent (float): kernel bandwidth for Z term
            h_theta (float): kernel bandwidth for theta term
            t: step

        Returns:
            [A, B] kernel values
        """
        return vmap(vmap(self.f_kernel, (None, None, 0, 0, None, None, None), 0), 
            (0, 0, None, None, None, None, None), 0)(x_latents, x_thetas, y_latents, y_thetas, h_latent, h_theta, t)


    def eltwise_grad_kernel_z(self, x_latents, x_thetas, y_latent, y_theta, h_latent, h_theta, t):
        """
        Computes gradient d/dz k((z, theta), (z', theta')) elementwise for each provided particle (z, theta)

        Args:
            x_latents: batch of latent particles [n_particles, d, k, 2]
            x_thetas: batch of parameter PyTree with leading dim `n_particles`
            y_latent: single latent particle [d, k, 2] (z')
            y_theta: single parameter PyTree (theta')
            h_latent (float): kernel bandwidth for Z term
            h_theta (float): kernel bandwidth for theta term
            t: step

        Returns:
            batch of gradients for latent tensors Z [n_particles, d, k, 2]
        
        """
        grad_kernel_z = grad(self.f_kernel, 0)
        return vmap(grad_kernel_z, (0, 0, None, None, None, None, None), 0)(x_latents, x_thetas, y_latent, y_theta, h_latent, h_theta, t)


    def eltwise_grad_kernel_theta(self, x_latents, x_thetas, y_latent, y_theta, h_latent, h_theta, t):
        """
        Computes gradient d/dtheta k((z, theta), (z', theta')) elementwise for each provided particle (z, theta)

        Args:
            x_latents: batch of latent particles [n_particles, d, k, 2]
            x_thetas: batch of parameter PyTree with leading dim `n_particles`
            y_latent: single latent particle [d, k, 2] (z')
            y_theta: single parameter PyTree (theta')
            h_latent (float): kernel bandwidth for Z term
            h_theta (float): kernel bandwidth for theta term
            t: step

        Returns:
            batch of gradients for parameters (PyTree with leading dim `n_particles`)
        """
        grad_kernel_theta = grad(self.f_kernel, 1)
        return vmap(grad_kernel_theta, (0, 0, None, None, None, None, None), 0)(x_latents, x_thetas, y_latent, y_theta, h_latent, h_theta, t)


    def z_update(self, single_z, single_theta, kxx, z, theta, grad_log_prob_z, h_latent, h_theta, t):
        """
        Computes SVGD update for `single_z` of a (single_z, single_theta) tuple given the kernel values 
        `kxx` and the d/dz gradients of the target density for each of the available particles 

        Args:
            single_z: single latent tensor Z [d, k, 2], which is the Z particle being updated
            single_theta: single parameter PyTree, the theta particle of the Z particle being updated
            kxx: pairwise kernel values for all particles [n_particles, n_particles]  
            z:  all latent tensor Z particles [n_particles, d, k, 2] 
            theta: all theta particles as PyTree with leading dim `n_particles` 
            grad_log_prob_z: gradients of all Z particles w.r.t target density  [n_particles, d, k, 2]  

        Returns
            transform vector of shape [d, k, 2] for the Z particle being updated        
        """

        # compute terms in sum
        weighted_gradient_ascent = kxx[..., None, None, None] * grad_log_prob_z
        repulsion = self.eltwise_grad_kernel_z(z, theta, single_z, single_theta, h_latent, h_theta, t)

        # average and negate (for optimizer)
        return - (weighted_gradient_ascent + repulsion).mean(axis=0)


    def parallel_update_z(self, *args):
        """
        Parallelizes `z_update` for all available particles
        Otherwise, same inputs as `z_update`.
        """
        return vmap(self.z_update, (0, 0, 1, None, None, None, None, None, None), 0)(*args)


    def theta_update(self, single_z, single_theta, kxx, z, theta, grad_log_prob_theta, h_latent, h_theta, t):
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
            lambda leaf_theta_grad: 
                expand_by(kxx, leaf_theta_grad.ndim - 1) * leaf_theta_grad, 
            grad_log_prob_theta)

        repulsion = self.eltwise_grad_kernel_theta(z, theta, single_z, single_theta, h_latent, h_theta, t)

        # average and negate (for optimizer)
        return  tree_multimap(
            lambda grad_asc_leaf, repuls_leaf: 
                - (grad_asc_leaf + repuls_leaf).mean(axis=0), 
            weighted_gradient_ascent, 
            repulsion)


    def parallel_update_theta(self, *args):
        """
        Parallelizes `theta_update` for all available particles
        Otherwise, same inputs as `theta_update`.
        """
        return vmap(self.theta_update, (0, 0, 1, None, None, None, None, None, None), 0)(*args)


    # this is the crucial @jit
    @functools.partial(jit, static_argnums=(0,))
    def svgd_step(self, opt_state_z, opt_state_theta, interv_targets, key, t, sf_baseline):
        """
        Performs a single SVGD step in the DiBS framework, updating Z and theta jointly.
        
        Args:
            opt_state_z: optimizer state for latent Z particles; contains [n_particles, d, k, 2]
            opt_state_theta: optimizer state for theta particles; contains PyTree with `n_particles` leading dim
            key: prng key
            t: step
            sf_baseline: batch of baseline values in case score function gradient is used [n_particles, ]

        Returns:
            the updated inputs
        """

        z = self.get_params(opt_state_z) # [n_particles, d, k, 2]
        theta = self.get_params(opt_state_theta) # PyTree with `n_particles` leading dim
        n_particles = z.shape[0]

        # make sure same bandwith is used for all calls to k(x, x') (in case e.g. the median heuristic is applied)
        h_latent = self.kernel_.h_latent
        h_theta = self.kernel_.h_theta

        # d/dtheta log p(theta, D | z)
        # -- this version uses the same key for all particles -> incorrect sampling?
        # key, subk = random.split(key)
        # dtheta_log_prob = self.eltwise_grad_theta_likelihood(z, theta, interv_targets, t, subk)
        key, *batch_subk = random.split(key, n_particles + 1)
        dtheta_log_prob = self.eltwise_grad_theta_likelihood(z, theta, interv_targets, t, jnp.array(batch_subk))

        # d/dz log p(theta, D | z)
        key, *batch_subk = random.split(key, n_particles + 1)
        dz_log_likelihood, sf_baseline = self.eltwise_grad_z_likelihood(z, theta, interv_targets, sf_baseline, t, jnp.array(batch_subk))

        # d/dz log p(z) (acyclicity)
        key, *batch_subk = random.split(key, n_particles + 1)
        dz_log_prior = self.eltwise_grad_latent_prior(z, jnp.array(batch_subk), t)

        # d/dz log p(z, theta, D) = d/dz log p(z)  + log p(theta, D | z)
        dz_log_prob = dz_log_prior + dz_log_likelihood

        # k((z, theta), (z, theta)) for all particles
        kxx = self.f_kernel_mat(z, theta, z, theta, h_latent, h_theta, t)

        # transformation phi() applied in batch to each particle individually
        phi_z = self.parallel_update_z(z, theta, kxx, z, theta, dz_log_prob, h_latent, h_theta, t)
        phi_theta = self.parallel_update_theta(z, theta, kxx, z, theta, dtheta_log_prob, h_latent, h_theta, t)

        # apply transformation
        # `x += stepsize * phi`; the phi returned is negated for SVGD
        opt_state_z = self.opt_update(t, phi_z, opt_state_z)
        opt_state_theta = self.opt_update(t, phi_theta, opt_state_theta)

        return opt_state_z, opt_state_theta, key, sf_baseline



    def sample_particles(self, *, n_steps, init_particles_z, init_particles_theta, key, interv_targets, callback=None, callback_every=0):
        """
        Deterministically transforms particles to minimize KL to target using SVGD

        Arguments:
            data: collection of datasets we perform inference for [n_datasets, n_obs, d]
            interv_targets: boolean masks for where interventions are performed (if known) [n_datasets, d]
            n_steps (int): number of SVGD steps performed
            init_particles_z: batch of initialized latent tensor particles [n_particles, d, k, 2]
            init_particles_theta:  batch of parameters PyTree (i.e. for a general parameter set shape) 
                with leading dimension `n_particles`
            key: prng key
            callback: function to be called every `callback_every` steps of SVGD.
            callback_every: if == 0, `callback` is never called. 

        Returns: 
            `n_particles` samples that approximate the DiBS target density
            particles_z: [n_particles, d, k, 2]
            particles_theta: PyTree of parameters with leading dimension `n_particles`
           
        """

        z = init_particles_z
        theta = init_particles_theta

        # initialize score function baseline (one for each particle)
        n_particles, _, n_dim, _ = z.shape
        sf_baseline = jnp.zeros(n_particles)

        if self.latent_prior_std is None:
            self.latent_prior_std = 1.0 / jnp.sqrt(n_dim)


        # init optimizer
        if self.optimizer['name'] == 'gd':
            opt = optimizers.sgd(self.optimizer['stepsize']/ 10.0) # comparable scale for tuning
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

        """Execute particle update steps for all particles in parallel using `vmap` functions"""
        it = tqdm.tqdm(range(n_steps), desc='DiBS', disable=not self.verbose)
        for t in it:

            # perform one SVGD step (compiled with @jit)
            opt_state_z, opt_state_theta, key, sf_baseline  = self.svgd_step(
                opt_state_z, opt_state_theta, interv_targets, key, t, sf_baseline)

            # callback
            if callback and callback_every and (((t+1) % callback_every == 0) or (t == (n_steps - 1))):
                z = self.get_params(opt_state_z)
                theta = self.get_params(opt_state_theta)
                self.particles_z = z
                self.particles_theta = theta
                callback(
                    dibs=self,
                    t=t,
                    zs=z,
                    thetas=theta,
                    grad_dict=self.grad_dict if hasattr(self, 'grad_dict') else None,
                    interv_belief=interv_targets
                )


        # return transported particles
        z_final = self.get_params(opt_state_z)
        theta_final = self.get_params(opt_state_theta)
        return z_final, theta_final


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
                tuple[2] contains the empirical log probability (here uniform)

        """
        if self.particles_z is None:
            raise AttributeError(
                'The model has not been fitted yet. Call .fit(X) first to obtain particles.'
            )
        particles_g = self.particle_to_g_lim(self.particles_z)
        ids = bit2id(particles_g)

        # empirical
        log_probs = - jnp.log(self.n_particles) * jnp.ones(self.n_particles)

        return ids, self.particles_theta, log_probs


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
                tuple[1] contains thetas
                tuple[2] contains the empirical log probability

        """
        if self.particles_z is None:
            raise AttributeError(
                'The model has not been fitted yet. Call .fit(X) first to obtain particles.'
            )

        particles_g = self.particle_to_g_lim(self.particles_z)
        ids = bit2id(particles_g)

        eltwise_log_prob = jit(
            vmap(lambda g, theta: self.target_log_joint_prob(g, theta, self.interv_targets), (0, 0), 0))
        # mixture using relative log probs
        # assumes that every particle is unique (always true because of theta)
        log_probs = eltwise_log_prob(particles_g, self.particles_theta)
        log_probs -= logsumexp(log_probs)

        return ids, self.particles_theta, log_probs


    def get_particles(self):
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
        
        return self.particles_z, self.particles_theta