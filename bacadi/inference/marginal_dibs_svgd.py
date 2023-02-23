import functools
import warnings
from jax._src.scipy.special import logsumexp
import tqdm

import jax.numpy as jnp
from jax import jit, vmap, random, grad
from jax.experimental import optimizers
from bacadi.eval.target import make_graph_model, make_inference_model, make_kernel

from bacadi.inference.dibs import DiBS
from bacadi.utils.func import bit2id, id2bit


class MarginalDiBS(DiBS):
    """
    This class implements DiBS: Differentiable Bayesian Structure Learning (Lorch et al., 2021)
    instantiated using Stein Variational Gradient Descent (SVGD) (Liu and Wang, 2016) as the underlying inference method.
    An SVGD update of vector v is defined as

        phi(v) = 1/n_particles sum_u k(v, u) d/du log p(u) + d/du k(u, v)

    This class implements //marginal// inference of p(G, theta | D).
    For joint inference of p(G | D), use the class `JointDiBS`

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
                 kernel,
                 graph_prior,
                 model_prior,
                 alpha_linear,
                 edges_per_node=1,
                 beta_linear=1.0,
                 tau=1.0,
                 h_latent=2.,
                 optimizer=dict(name='rmsprop', stepsize=0.005),
                 n_grad_mc_samples=128,
                 n_acyclicity_mc_samples=32,
                 grad_estimator_z='score',
                 score_function_baseline=0.0,
                 latent_prior_std=None,
                 model_param=dict(
                     alpha_mu=1.,
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

    def fit(self, data, interv_targets=None, envs=None):
        """
        Perform inference of the posterior joint probability using SVGD.

        Args:
            X (tuple): tuple with (data, intervention targets).
                data contains the observations of the form: (n_datasets, n_obs, n_vars)
                intv_targets contains the boolean masks for interventions of the form: (n_datasets, n_vars)
        
        Returns:
            self
        """
        if self.is_fitted_ and jnp.array_equal(self.data, data) and not self.key_changed_:
            warnings.warn(
                'The data and initial random state after fitting `self` have not changed and nothing will be done.' + \
                'In order to refit, either change the random state via `.set_new_key()` or create a new object.'
            )
            return self

        self.key_changed_ = False
        self.is_fitted_ = False
        self.data, self.interv_targets, self.envs = data, interv_targets, envs
        self.n_vars = self.data.shape[-1]

        self.graph_model = make_graph_model(graph_prior_str=self.graph_prior,
                                            n_vars=self.n_vars,
                                            edges_per_node=self.edges_per_node)

        self.model_param['graph_model'] = self.graph_model

        self.inference_model = make_inference_model(
            inference_str=self.inference_prior, 
            n_vars=self.n_vars, **self.model_param)

        self.kernel_ = make_kernel(kernel=self.kernel, h_latent=self.h_latent)

        key, subk = random.split(self.random_state)
        init_particles_z = self.sample_initial_random_particles(
            key=subk, n_particles=self.n_particles, n_vars=self.n_vars)

        # iteratively transport particles
        key, subk = random.split(key)
        self.particles_z = self.sample_particles(
            data=self.data,
            interv_targets=self.interv_targets,
            key=subk,
            n_steps=self.n_steps,
            init_particles_z=init_particles_z,
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
            particles_theta: PyTree with leading dim `n_particles` 
        """
        if self.particles_z is None:
            raise AttributeError(
                'The model has not been fitted yet. Call .fit(X) first to obtain particles.'
            )
        
        return self.particles_z

    """
    Additional functions needed for the marginal likelihood
    """

    def log_marginal_prob(self, data, single_w, interv_targets, envs):
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

    def target_log_marginal_prob(self, single_w, interv_targets):
        """
        log marginal probability of all data D given the graph.
        Uses the data and intervention targets passed to the `sample_particles` method
        and stored in `self`.
        
        Args:
            single_w: single graph [d, d]
        
        Returns:
            log p(D | G, I): [1,]
        """

        return self.log_marginal_prob(self.data, single_w, interv_targets, self.envs)


    def target_log_joint_prob(self, single_w, single_theta, interv_targets, rng):
        """
        !! dummy function
        log joint probability of theta and D given the graph and parameters.
        Uses the data and intervention targets passed to the `sample_particles` method
        and stored in `self`.
        
        To unify the function signatures for the marginal and joint inference classes
        `MarginalDiBS` and `JointDiBS`, we define this function as the marginal log likelihood
        variant with dummy parameter inputs. This will allow using the same 
        gradient estimator functions for both inference cases.
        
        
        Args:
            single_w: single graph [d, d]
            single_theta: single parameter PyTree
            rng: [1, ]
        
        Returns:
            log p(theta, D | G, I): [1,]
        """

        return self.target_log_marginal_prob(single_w, interv_targets)


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
        """
        # default full rank
        if n_dim is None:
            n_dim = n_vars 
        
        # like prior
        std = self.latent_prior_std or (1.0 / jnp.sqrt(n_dim))

        # sample
        key, subk = random.split(key)
        z = random.normal(subk, shape=(n_particles, n_vars, n_dim, 2)) * std        

        return z


    def f_kernel(self, x_latent, y_latent, h, t):
        """
        Evaluates kernel

        Args:
            x_latent: latent tensor [d, k, 2]
            y_latent: latent tensor [d, k, 2]
            h (float): kernel bandwidth 
            t: step

        Returns:
            [1, ] kernel value
        """
        return self.kernel_.eval(x=x_latent, y=y_latent, h=h)
    

    def f_kernel_mat(self, x_latents, y_latents, h, t):
        """
        Computes pairwise kernel matrix

        Args:
            x_latents: latent tensor [A, d, k, 2]
            y_latents: latent tensor [B, d, k, 2]
            h (float): kernel bandwidth 
            t: step

        Returns:
            [A, B] kernel values
        """
        return vmap(vmap(self.f_kernel, (None, 0, None, None), 0), 
            (0, None, None, None), 0)(x_latents, y_latents, h, t)


    def eltwise_grad_kernel_z(self, x_latents, y_latent, h, t):
        """
        Computes gradient d/dz k(z, z') elementwise for each provided particle z

        Args:
            x_latents: batch of latent particles [n_particles, d, k, 2]
            y_latent: single latent particle [d, k, 2] (z')
            h (float): kernel bandwidth 
            t: step

        Returns:
            batch of gradients for latent tensors Z [n_particles, d, k, 2]
        """        
        grad_kernel_z = grad(self.f_kernel, 0)
        return vmap(grad_kernel_z, (0, None, None, None), 0)(x_latents, y_latent, h, t)


    def z_update(self, single_z, kxx, z, grad_log_prob_z, h, t):
        """
        Computes SVGD update for `single_z` particlee given the kernel values 
        `kxx` and the d/dz gradients of the target density for each of the available particles 

        Args:
            single_z: single latent tensor Z [d, k, 2], which is the Z particle being updated
            kxx: pairwise kernel values for all particles [n_particles, n_particles]  
            z:  all latent tensor Z particles [n_particles, d, k, 2] 
            grad_log_prob_z: gradients of all Z particles w.r.t target density  [n_particles, d, k, 2]  

        Returns
            transform vector of shape [d, k, 2] for the Z particle being updated        

        """
    
        # compute terms in sum
        weighted_gradient_ascent = kxx[..., None, None, None] * grad_log_prob_z
        repulsion = self.eltwise_grad_kernel_z(z, single_z, h, t)

        # average and negate (for optimizer)
        return - (weighted_gradient_ascent + repulsion).mean(axis=0)


    def parallel_update_z(self, *args):
        """
        Parallelizes `z_update` for all available particles
        Otherwise, same inputs as `z_update`.
        """
        return vmap(self.z_update, (0, 1, None, None, None, None), 0)(*args)



    # this is the crucial @jit
    @functools.partial(jit, static_argnums=(0,))
    def svgd_step(self, opt_state_z, interv_targets, key, t, sf_baseline):
        """
        Performs a single SVGD step in the DiBS framework, updating all Z particles jointly.

        Args:
            opt_state_z: optimizer state for latent Z particles; contains [n_particles, d, k, 2]
            key: prng key
            t: step
            sf_baseline: batch of baseline values in case score function gradient is used [n_particles, ]

        Returns:
            the updated inputs
        """
     
        z = self.get_params(opt_state_z) # [n_particles, d, k, 2]
        n_particles = z.shape[0]

        # make sure same bandwith is used for all calls to k(x, x') (in case e.g. the median heuristic is applied)
        h = self.kernel_.h

        # d/dz log p(D | z)
        key, *batch_subk = random.split(key, n_particles + 1) 
        dz_log_likelihood, sf_baseline = self.eltwise_grad_z_likelihood(z, None, interv_targets, sf_baseline, t, jnp.array(batch_subk))
        # here `None` is a placeholder for theta (in the joint inference case) 
        # since this is an inherited function from the general `DiBS` class

        # d/dz log p(z) (acyclicity)
        key, *batch_subk = random.split(key, n_particles + 1)
        dz_log_prior = self.eltwise_grad_latent_prior(z, jnp.array(batch_subk), t)

        # d/dz log p(z, D) = d/dz log p(z)  + log p(D | z) 
        dz_log_prob = dz_log_prior + dz_log_likelihood
        
        # k(z, z) for all particles
        kxx = self.f_kernel_mat(z, z, h, t)

        # transformation phi() applied in batch to each particle individually
        phi_z = self.parallel_update_z(z, kxx, z, dz_log_prob, h, t)

        # apply transformation
        # `x += stepsize * phi`; the phi returned is negated for SVGD
        opt_state_z = self.opt_update(t, phi_z, opt_state_z)

        return opt_state_z, key, sf_baseline
    
    

    def sample_particles(self, *, data, n_steps, init_particles_z, key, interv_targets, callback=None, callback_every=0):
        """
        Deterministically transforms particles to minimize KL to target using SVGD

        Arguments:
            n_steps (int): number of SVGD steps performed
            init_particles_z: batch of initialized latent tensor particles [n_particles, d, k, 2]
            key: prng key
            callback: function to be called every `callback_every` steps of SVGD.
            callback_every: if == 0, `callback` is never called. 

        Returns: 
            `n_particles` samples that approximate the DiBS target density
            particles_z: [n_particles, d, k, 2]
        """

        self.data = data
        self.interv_targets = interv_targets
        z = init_particles_z
           
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

        """Execute particle update steps for all particles in parallel using `vmap` functions"""
        it = tqdm.tqdm(range(n_steps), desc='DiBS', disable=not self.verbose)
        for t in it:

            # perform one SVGD step (compiled with @jit)
            opt_state_z, key, sf_baseline  = self.svgd_step(
                opt_state_z, interv_targets, key, t, sf_baseline)

            # callback
            if callback and callback_every and (((t+1) % callback_every == 0) or (t == (n_steps - 1))):
                z = self.get_params(opt_state_z)
                self.particles_z = z
                callback(
                    model=self,
                    t=t,
                    zs=z,
                )

        # return transported particles
        z_final = self.get_params(opt_state_z)
        return z_final


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
        unique, counts = jnp.unique(ids, axis=0, return_counts=True)

        # empirical using counts
        log_probs = jnp.log(counts) - jnp.log(self.n_particles)

        return unique, log_probs


    def particle_mixture(self, union=True):
        """
        Returns the standardized form of a particle distribution weighted by
        the likelihood represented by this fitted object. 
        Converts the batch z particles into the binary adjacency matrices
        (for alpha -> inf) and returns it with empirical log probabilities as a tuple.
        
        Args:
            union: whether to use union over DAGs (i.e. dropping duplicates)
                or keep them. this only has influence on the weighting of the mixture
        
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
        unique, counts = jnp.unique(ids, axis=0, return_counts=True)

        eltwise_log_prob = jit(
            vmap(
                lambda g: self.log_marginal_prob(data=self.data,
                                                 interv_targets=self.
                                                 interv_targets,
                                                 single_w=g,
                                                 envs=self.envs), (0, ), 0))
        # mixture using relative log probs
        log_probs = eltwise_log_prob(id2bit(unique, self.n_vars))
        if not union:
            log_probs += jnp.log(counts)

        log_probs -= logsumexp(log_probs)

        return unique, log_probs

