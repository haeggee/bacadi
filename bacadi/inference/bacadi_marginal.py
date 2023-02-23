import functools
import warnings
from jax._src.scipy.special import logsumexp
import tqdm

import jax.numpy as jnp
from jax import jit, vmap, random, grad
from jax.experimental import optimizers
from bacadi.eval.target import make_graph_model, make_inference_model, make_kernel

from bacadi.inference.bacadi_base import BaCaDIBase
from bacadi.utils.func import bit2id, id2bit


class BaCaDIMarginal(BaCaDIBase):
    """
    This class implements BaCaDI: Bayesian Causal Discovery with Unknown Interventions (Hägele et al., 2021)
    instantiated using Stein Variational Gradient Descent (SVGD) (Liu and Wang, 2016) as the underlying inference method.

    This class implements //marginal// inference of the posterior p(G, I | D) 

    Args:
        kernel: string (`frob`) that specifies which kernel to use
        graph_prior: string (`er` or `sf`) that specifies the graph model object (GraphDistribution) to be used that captures the prior over graphs
        model_prior: string (`bge`) that specifies the JAX-BN model object for inference, i.e. defines log_likelihood of data and parameters given the graph
        alpha_linear (float): inverse temperature parameter schedule of sigmoid
        beta_linear (float): inverse temperature parameter schedule of prior
        h_latent (float): bandwidth parameter for the Frobenius kernel for Z
        optimizer (dict): dictionary with at least keys `name` and `stepsize`
        n_grad_mc_samples (int): MC samples in gradient estimator for likelihood term p(theta, D | G)
        n_acyclicity_mc_samples (int): MC samples in gradient estimator for acyclicity constraint
        grad_estimator_z (str): gradient estimator d/dZ of expectation; choices: `score` or `reparam`
        score_function_baseline (float): weight of addition in score function baseline; == 0.0 corresponds to not using a baseline
        latent_prior_std (float): standard deviation of Gaussian prior over Z; defaults to 1/sqrt(k)
        model_param (dict): dictionary specifying model parameters. 
        random_state: prng key
        n_steps (int): number of steps to iterate SVGD
        callback_every (int): if == 0, `callback` is never called. 
        callback (func): function to be called every `callback_every` steps of SVGD.
    """
    def __init__(self,
                 *,
                 graph_prior,
                 model_prior,
                 alpha_linear,
                 kernel='frob-interv-add',
                 edges_per_node=1,
                 interv_per_env=1,
                 beta_linear=1.0,
                 tau=1.0,
                 h_latent=5.,
                 h_interv=5.,
                 lambda_regul=10,
                 optimizer=dict(name='rmsprop', stepsize=0.005),
                 n_grad_mc_samples=128,
                 n_acyclicity_mc_samples=32,
                 grad_estimator_z='reparam',
                 score_function_baseline=0.0,
                 latent_prior_std=None,
                 model_param=dict(alpha_mu=1., ),
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
                         lambda_regul=lambda_regul,
                         interv_per_env=interv_per_env,
                         tau=tau,
                         h_latent=h_latent,
                         h_interv=h_interv,
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

        self.key_changed_ = False
        self.is_fitted_ = False
        self.data = data
        self.envs = envs
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
                                   h_interv=self.h_interv)

        key, subk = random.split(self.random_state)
        init_particles_z, init_particles_gamma = self.sample_initial_random_particles(
            key=subk,
            n_particles=self.n_particles,
            n_env=self.n_env,
            n_vars=self.n_vars,
            known_interv_targets=known_interv_targets)

        # iteratively transport particles
        key, subk = random.split(key)
        self.particles_z, self.particles_gamma = self.sample_particles(
            data=self.data,
            key=subk,
            n_steps=self.n_steps,
            init_particles_z=init_particles_z,
            init_particles_gamma=init_particles_gamma,
            callback_every=self.callback_every,
            callback=self.callback)

        self.is_fitted_ = True
        self.key_changed_ = False
        return self

    def get_particles(self):
        """
        Get the particles that have been fit to approximate the joint posterior probability.
        Raises an error when called before model is fitted.

        Args: None

        Returns:
            particles_z: (n_particles, n_vars, n_dim, 2)
            particles_gamma:  (n_particles, n_env - 1, n_dim)
        """
        if self.particles_z is None:
            raise AttributeError(
                'The model has not been fitted yet. Call .fit(X) first to obtain particles.'
            )

        return self.particles_z, self.particles_gamma

    """
    Additional functions needed for the marginal likelihood
    """
    def log_marginal_prob(self,
                          data,
                          single_w,
                          interv_targets,
                          envs=None,
                          rng=None):
        """
        log likelihood of a single dataset D_k given the graph and the intervention
        targets marginalized over theta.
        This is just a wrapper using the Jax-BN model that is the basis of inference.

        Args:
            data: single dataset [n_obs, d]
            single_w: single graph [d,d]
            interv_targets: boolean mask for interventions [n_env-1, d]
        
        Returns:
            log p(D_k | G, I_k)
        """

        return self.inference_model.log_marginal_likelihood_given_g(
            data=data,
            w=single_w,
            interv_targets=interv_targets,
            envs=envs)

    def log_marginal_prob_all(self,
                          data,
                          single_w,
                          interv_targets,
                          envs=None,
                          rng=None):
        """
        log likelihood of a single dataset D_k given the graph and the intervention
        targets marginalized over theta.
        This is just a wrapper using the Jax-BN model that is the basis of inference.

        Args:
            data: single dataset [n_obs, d]
            single_w: single graph [d,d]
            interv_targets: boolean mask for interventions [n_env-1, d]
        
        Returns:
            log p(D_k | G, I_k)
        """

        loglik_obs = self.inference_model.log_marginal_likelihood_given_g(
            data=data,
            w=single_w,
            interv_targets=interv_targets,
            envs=envs)
        loglik_intv = self.inference_model.log_score_interventions(
                data=data,
                interv_targets=interv_targets,
                envs=envs)
        return loglik_obs + loglik_intv

    def target_log_marginal_prob(self, single_w, interv_targets, rng=None):
        """
        log marginal probability of all data D given the graph.
        Uses the data and intervention targets passed to the `sample_particles` method
        and stored in `self`.
        
        Args:
            single_w: single graph [d, d]
            interv_targets: [n_env-1, d]
        
        Returns:
            log p(D | G, I): [1,]
        """
        return self.log_marginal_prob_all(self.data, single_w, interv_targets,
                                      self.envs, rng)

    def target_log_joint_prob(self, single_w, single_theta, interv_targets,
                              rng):
        """
        !! dummy function
        log joint probability of theta and D given the graph and parameters.
        Uses the data and intervention targets passed to the `sample_particles` method
        and stored in `self`.
        
        To unify the function signatures for the marginal and joint inference classes
        `BaCaDIJoint` and `BaCaDIMarginal`, we define this function as the marginal log likelihood
        variant with dummy parameter inputs. This will allow using the same 
        gradient estimator functions for both inference cases.
        
        
        Args:
            single_w: single graph [d, d]
            single_theta: single parameter PyTree
            interv_targets: [n_env, d]
            rng: [1, ]
        
        Returns:
            log p(theta, D | G, I): [1,]
        """
        return self.target_log_marginal_prob(single_w, interv_targets, rng)

    def target_log_joint_prob_soft_interv(self, single_w, single_theta,
                                          interv_targets, rng):
        """
        !! dummy function
        
        To unify the function signatures for the marginal and joint inference classes
        `BaCaDIJoint` and `BaCaDIMarginal`, we define this function as the marginal log likelihood
        variant with dummy parameter inputs. This will allow using the same 
        gradient estimator functions for both inference cases.
        
        Uses the data passed to the `sample_particles` method
        and stored in `self`.
        
        Args:
            single_w: single graph [d, d]
            single_theta: single parameter PyTree
            interv_targets: soft mask for interventions [n_env - 1, d]
            rng: [1, ]
        
        Returns:
            log p(theta, D | G, I): [1,]
        """
        # for marginal BGe model, marginal prob works with soft interv masks
        return self.target_log_marginal_prob(single_w, interv_targets, rng)

    def sample_initial_random_particles(self,
                                        *,
                                        key,
                                        n_particles,
                                        n_vars,
                                        n_env,
                                        known_interv_targets=None,
                                        n_dim=None):
        """
        TODO: use known interv targets somehow
        Samples random particles to initialize SVGD

        Args:
            key: rng key
            n_particles: number of particles for SVGD
            n_dim: size of latent dimension `k`. Defaults to `n_vars`, s.t. k == d

        Returns:
            z: batch of latent tensors [n_particles, d, k, 2]    
            gamma: batch of tensors [n_particles, n_env-1, d]
        """
        # default full rank
        if n_dim is None:
            n_dim = n_vars

        # like prior
        std = self.latent_prior_std or (1.0 / jnp.sqrt(n_dim))

        # sample
        key, subk = random.split(key)
        z = random.normal(subk, shape=(n_particles, n_vars, n_dim, 2)) * std

        # gamma like gaussian prior
        if self.gamma_prior_std is None:
            self.gamma_prior_std = jnp.sqrt(0.1)
        key, subk = random.split(key)
        gamma = random.normal(
            subk, shape=(n_particles, n_env - 1, n_dim)) * self.gamma_prior_std
        return z, gamma

    def f_kernel(self, x_latent, x_interv, y_latent, y_interv, h_latent,
                 h_interv, t):
        """
        Evaluates kernel

        Args:
            x_latent: latent tensor [d, k, 2]
            x_interv: parameter PyTree 
            y_latent: latent tensor [d, k, 2]
            y_interv: parameter PyTree 
            h_latent (float): kernel bandwidth for Z term
            h_interv (float): kernel bandwidth for gamma term
            t: step

        Returns:
            kernel value
        """
        return self.kernel_.eval(x_latent=x_latent,
                                 x_interv=x_interv,
                                 y_latent=y_latent,
                                 y_interv=y_interv,
                                 h_latent=h_latent,
                                 h_interv=h_interv)

    def f_kernel_mat(self, x_latents, x_intervs, y_latents, y_intervs,
                     h_latent, h_interv, t):
        """
        Computes pairwise kernel matrix

        Args:
            x_latents: latent tensor [A, d, k, 2]
            x_intervs: parameter PyTree with batch size A as leading dim
            y_latents: latent tensor [B, d, k, 2]
            y_intervs: parameter PyTree with batch size B as leading dim
            h_latent (float): kernel bandwidth for Z term
            h_interv (float): kernel bandwidth for gamma term
            t: step

        Returns:
            [A, B] kernel values
        """
        return vmap(
            vmap(self.f_kernel, (None, None, 0, 0, None, None, None),
                 0), (0, 0, None, None, None, None, None),
            0)(x_latents, x_intervs, y_latents, y_intervs, h_latent, h_interv,
               t)

    def eltwise_grad_kernel_z(self, x_latents, x_intervs, y_latent, y_interv,
                              h_latent, h_interv, t):
        """
        Computes gradient d/dz k((z, gamma), (z', gamma')) elementwise for each provided particle (z, gamma)

        Args:
            x_latents: batch of latent particles [n_particles, d, k, 2]
            x_intervs: batch of parameter PyTree with leading dim `n_particles`
            y_latent: single latent particle [d, k, 2] (z')
            y_interv: single parameter PyTree (gamma')
            h_latent (float): kernel bandwidth for Z term
            h_interv (float): kernel bandwidth for gamma term
            t: step

        Returns:
            batch of gradients for latent tensors Z [n_particles, d, k, 2]
        
        """
        grad_kernel_z = grad(self.f_kernel, 0)
        return vmap(grad_kernel_z, (0, 0, None, None, None, None, None),
                    0)(x_latents, x_intervs, y_latent, y_interv, h_latent,
                       h_interv, t)

    def eltwise_grad_kernel_gamma(self, x_latents, x_intervs, y_latent,
                                  y_interv, h_latent, h_interv, t):
        """
        Computes gradient d/dgamma k((z, gamma), (z', gamma')) elementwise for each provided particle (z, gamma)

        Args:
            x_latents: batch of latent particles [n_particles, d, k, 2]
            x_intervs: batch of parameter PyTree with leading dim `n_particles`
            y_latent: single latent particle [d, k, 2] (z')
            y_interv: single parameter PyTree (gamma')
            h_latent (float): kernel bandwidth for Z term
            h_interv (float): kernel bandwidth for gamma term
            t: step

        Returns:
            batch of gradients for parameters (PyTree with leading dim `n_particles`)
        """
        grad_kernel_gamma = grad(self.f_kernel, 1)
        return vmap(grad_kernel_gamma, (0, 0, None, None, None, None, None),
                    0)(x_latents, x_intervs, y_latent, y_interv, h_latent,
                       h_interv, t)

    def z_update(self, single_z, single_gamma, kxx, z, gamma, grad_log_prob_z,
                 h_latent, h_interv, t):
        """
        Computes SVGD update for `single_z` of a (single_z, single_gamma) tuple given the kernel values 
        `kxx` and the d/dz gradients of the target density for each of the available particles 

        Args:
            single_z: single latent tensor Z [d, k, 2], which is the Z particle being updated
            single_gamma: single parameter PyTree, the gamma particle of the Z particle being updated
            kxx: pairwise kernel values for all particles [n_particles, n_particles]  
            z:  all latent tensor Z particles [n_particles, d, k, 2] 
            gamma: all gamma particles as PyTree with leading dim `n_particles` 
            grad_log_prob_z: gradients of all Z particles w.r.t target density  [n_particles, d, k, 2]  

        Returns
            transform vector of shape [d, k, 2] for the Z particle being updated        
        """

        # compute terms in sum
        weighted_gradient_ascent = kxx[..., None, None, None] * grad_log_prob_z
        repulsion = self.eltwise_grad_kernel_z(z, gamma, single_z,
                                               single_gamma, h_latent,
                                               h_interv, t)

        # average and negate (for optimizer)
        return -(weighted_gradient_ascent + repulsion).mean(axis=0)

    def parallel_update_z(self, *args):
        """
        Parallelizes `z_update` for all available particles
        Otherwise, same inputs as `z_update`.
        """
        return vmap(self.z_update,
                    (0, 0, 1, None, None, None, None, None, None), 0)(*args)

    def gamma_update(self, single_z, single_gamma, kxx, z, gamma,
                     grad_log_prob_gamma, h_latent, h_interv, t):
        """
        Computes SVGD update for `single_gamma` of a (single_z, single_gamma) tuple given the kernel values 
        `kxx` and the d/dgamma gradients of the target density for each of the available particles 

        Args:
            single_z: single latent tensor Z [d, k, 2], the Z particle of the gamma particle being updated
            single_gamma: single parameter [n_env-1, d] being updated
            kxx: pairwise kernel values for all particles [n_particles, n_particles]  
            z:  all latent tensor Z particles [n_particles, d, k, 2] 
            gamma: all gamma particles [n_particles, n_env-1, d] 
            grad_log_prob_gamma: gradients of all gamma particles w.r.t target density 
                [n_particles, n_env-1, d] 
        Returns:
            transform vector PyTree with leading dim `n_particles` for the gamma particle being updated   
        """
        # compute terms in sum
        weighted_gradient_ascent = kxx[..., None, None] * grad_log_prob_gamma

        repulsion = self.eltwise_grad_kernel_gamma(z, gamma, single_z,
                                                   single_gamma, h_latent,
                                                   h_interv, t)

        # average and negate (for optimizer)
        return -(weighted_gradient_ascent + repulsion).mean(axis=0)

    def parallel_update_gamma(self, *args):
        """
        Parallelizes `gamma_update` for all available particles
        Otherwise, same inputs as `gamma_update`.
        """
        return vmap(self.gamma_update,
                    (0, 0, 1, None, None, None, None, None, None), 0)(*args)

    # this is the crucial @jit
    @functools.partial(jit, static_argnums=(0, ))
    def svgd_step(self, opt_state_z, opt_state_gamma, key, t, sf_baseline):
        """
        Performs a single SVGD step, updating all Z and gamma particles jointly.

        Args:
            opt_state_z: optimizer state for latent Z particles; contains [n_particles, d, k, 2]
            key: prng key
            t: step
            sf_baseline: batch of baseline values in case score function gradient is used [n_particles, ]

        Returns:
            the updated inputs
        """

        z = self.get_params(opt_state_z)  # [n_particles, d, k, 2]
        gamma = self.get_params(opt_state_gamma)
        n_particles = z.shape[0]

        # make sure same bandwith is used for all calls to k(x, x') (in case e.g. the median heuristic is applied)
        h_latent = self.kernel_.h_latent
        h_interv = self.kernel_.h_interv

        # d/dz log p(D | z, gamma)
        key, *batch_subk = random.split(key, n_particles + 1)
        dz_log_likelihood, sf_baseline = self.eltwise_grad_z_likelihood(
            z, None, gamma, sf_baseline, t, jnp.array(batch_subk))
        # here `None` is a placeholder for theta (in the joint inference case)
        # since this is an inherited function from the general `BaCaDIBase` class

        # d/dz log p(z) (acyclicity)
        key, *batch_subk = random.split(key, n_particles + 1)
        dz_log_prior = self.eltwise_grad_latent_prior(z, jnp.array(batch_subk),
                                                      t)

        # d/dz log p(z, D) = d/dz log p(z)  + log p(D | z)
        dz_log_prob = dz_log_prior + dz_log_likelihood

        # d/dgamma log p(D | z, gamma)
        key, *batch_subk = random.split(key, n_particles + 1)
        dgamma_log_likelihood, _ = self.eltwise_grad_gamma_likelihood(
            z, None, gamma, sf_baseline, t, jnp.array(batch_subk))

        # d/dgamma log p(gamma) (prior for sparse interventions)
        key, *batch_subk = random.split(key, n_particles + 1)
        dgamma_log_prior = self.eltwise_grad_gamma_prior(
            gamma, jnp.array(batch_subk), t)
        # d/dgamma log p(z, gamma, D) = d/dgamma log p(gamma)  + log p(D | z, gamma)
        dgamma_log_prob = dgamma_log_prior + dgamma_log_likelihood

        # k(z, z) for all particles
        kxx = self.f_kernel_mat(z, gamma, z, gamma, h_latent, h_interv, t)

        # transformation phi() applied in batch to each particle individually
        phi_z = self.parallel_update_z(z, gamma, kxx, z, gamma, dz_log_prob,
                                       h_latent, h_interv, t)
        phi_gamma = self.parallel_update_gamma(z, gamma, kxx, z, gamma,
                                               dgamma_log_prob, h_latent,
                                               h_interv, t)

        # apply transformation
        # `x += stepsize * phi`; the phi returned is negated for SVGD
        opt_state_z = self.opt_update(t, phi_z, opt_state_z)
        opt_state_gamma = self.opt_update(t, phi_gamma, opt_state_gamma)

        return opt_state_z, opt_state_gamma, key, sf_baseline

    def sample_particles(self,
                         *,
                         data,
                         n_steps,
                         init_particles_z,
                         init_particles_gamma,
                         key,
                         callback=None,
                         callback_every=0):
        """
        Deterministically transforms particles to minimize KL to target using SVGD

        Arguments:
            n_steps (int): number of SVGD steps performed
            init_particles_z: batch of initialized latent tensor particles [n_particles, d, k, 2]
            key: prng key
            callback: function to be called every `callback_every` steps of SVGD.
            callback_every: if == 0, `callback` is never called. 

        Returns: 
            `n_particles` samples that approximate the BaCaDI target density
            particles_z: [n_particles, d, k, 2]
        """

        z = init_particles_z
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
        opt_state_gamma = opt_init(gamma)
        """Execute particle update steps for all particles in parallel using `vmap` functions"""
        it = tqdm.tqdm(range(n_steps), desc='BaCaDI', disable=not self.verbose)
        for t in it:

            # perform one SVGD step (compiled with @jit)
            opt_state_z, opt_state_gamma, key, sf_baseline = self.svgd_step(
                opt_state_z, opt_state_gamma, key, t, sf_baseline)

            # callback
            if callback and callback_every and (((t + 1) % callback_every == 0)
                                                or (t == (n_steps - 1))):
                z = self.get_params(opt_state_z)
                gamma = self.get_params(opt_state_gamma)
                self.particles_z = z
                self.particles_gamma = gamma
                callback(
                    model=self,
                    t=t,
                    gamma=gamma,
                    zs=z,
                )

        # return transported particles
        z_final = self.get_params(opt_state_z)
        gamma_final = self.get_params(opt_state_gamma)
        return z_final, gamma_final

    def particle_empirical(self):
        """
        Returns the standardized form of a particle distribution 
        represented by this fitted object. 
        Converts the batch z particles into the binary adjacency matrices
        (for alpha -> inf) and returns it with associated gammas and
        empirical log probabilities (here uniform) as a tuple.

        Returns:
            tuple: 
                tuple[0] contains ids as by `bit2id`
                tuple[1] contains gammas
                tuple[2] contains the empirical log probability (here uniform)

        """
        if self.particles_z is None:
            raise AttributeError(
                'The model has not been fitted yet. Call .fit(X) first to obtain particles.'
            )

        particles_g = self.particle_to_g_lim(self.particles_z)
        particles_I = self.particle_to_interv_lim(self.particles_gamma)

        # particles for g in id format
        # particles for I can stay like it
        ids = bit2id(particles_g)
        # empirical
        log_probs = -jnp.log(self.n_particles) * jnp.ones(self.n_particles)

        return ids, particles_I, log_probs

    def particle_mixture(self, union=True):
        """
        Returns the standardized form of a particle distribution weighted by
        the likelihood represented by this fitted object. 
        Converts the batch z and gamma particles into the binary adjacency matrices
        and intervention masks (for alpha, gamma -> inf)
        and returns it with empirical log probabilities as a tuple.
        
        Args:
            union: whether to use union over DAGs (i.e. dropping duplicates)
                or keep them. this only has influence on the weighting of the mixture
        
        Returns:
            tuple: 
                tuple[0] contains ids as by `bit2id`
                tuple[1] contains gammas
                tuple[2] contains the empirical log probability

        """
        if self.particles_z is None:
            raise AttributeError(
                'The model has not been fitted yet. Call .fit(X) first to obtain particles.'
            )
        particles_g = self.particle_to_g_lim(self.particles_z)
        particles_I = self.particle_to_interv_lim(self.particles_gamma)
        # particles for g in id format
        ids = bit2id(particles_g)
        # particles for I can stay as masks, easier to use

        ### compute log probs with training data
        # enhance the targets
        eltwise_log_prob = jit(
            vmap(
                lambda g, i_targets: self.target_log_marginal_prob(
                    interv_targets=i_targets, single_w=g), (0, 0), 0))

        # mixture using relative log probs
        log_probs = eltwise_log_prob(particles_g, particles_I)

        log_probs -= logsumexp(log_probs)

        return ids, particles_I, log_probs
