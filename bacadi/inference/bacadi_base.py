import warnings
from jax.scipy.special import logsumexp
from jax.interpreters.masking import ShapeError
import tqdm

import jax.numpy as jnp
from jax import jit, vmap, random, grad
from jax.experimental import optimizers
from jax.tree_util import tree_map, tree_multimap
from jax.nn import sigmoid

from bacadi.inference.dibs import DiBS

from bacadi.utils.func import bit2id, expand_by, id2bit


class BaCaDIBase(DiBS):
    """
    This class implements the base class for BaCaDI, generalizing the DiBS model 

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
                 kernel='frob-interv-add',
                 edges_per_node=1,
                 interv_per_env=0,
                 beta_linear=1.0,
                 tau=1.0,
                 h_latent=5.,
                 h_interv=5.,
                 lambda_regul=10.,
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
        self.h_interv = h_interv
        self.gamma_prior_std = gamma_prior_std
        self.interv_per_env = interv_per_env
        self.lambda_regul = lambda_regul

    """
    Specific methods for modelling of interventions
    """

    def interv_probs(self, gamma, t):
        """
        Edge probabilities encoded by latent representation 

        Args:
            gamma: scores [..., n_env - 1, d]
            t: step
        
        Returns:
            intervention probabilities of shape [..., n_env - 1, d]
        """
        probs = sigmoid(self.alpha(t) / 2 * (gamma))
        return probs

    def sample_I(self, p, subk, n_samples):
        """
        Sample Bernoulli matrix according to matrix of probabilities

        Args:
            p: matrix of probabilities [n_env-1, d]
            n_samples: number of samples
            subk: rng key
        
        Returns:
            an array of matrices sampled according to `p` of shape [n_samples, n_env-1, d]
        """
        n_vars = p.shape[1]
        I_samples = random.bernoulli(subk,
                                     p=p,
                                     shape=(n_samples, self.n_env - 1,
                                            n_vars)).astype(jnp.int32)
        return I_samples

    def particle_to_soft_interv(self, gamma, eps, t):
        """ 
        Gumbel-softmax / concrete distribution using Logistic(0,1) samples `eps`

        Args:
            gamma: a single latent tensor Z of shape [n_env-1, d]
            eps: random iid Logistic(0,1) noise  of shape [n_env-1, d] 
            t: step
        
        Returns:
            Gumbel-softmax sample of intervention targets [n_env-1, d]
        """

        # soft reparameterization using gumbel-softmax/concrete distribution
        # eps ~ Logistic(0,1)
        soft_interv = sigmoid(self.tau * (eps + self.alpha(t) / 2 * (gamma)))
        # soft_interv = sigmoid(self.tau * (eps + gamma))
        return soft_interv

    def particle_to_interv_lim(self, gamma):
        """
        Returns intervention targets corresponding
        to alpha = infinity for particles `gamma`

        Args:
            gamma: a single latent tensor of shape [..., n_env-1, d]

        Returns:
            boolean mask representing intv. targets
            of shape [..., n_env-1, d]
        """
        interv = (gamma > 0).astype(bool)
        return interv

    """
    Methods from dibs.py that changed
    """
    """
    Methods for the log likelihood
    """

    def log_joint_likelihood_soft_interv(self,
                                         single_data,
                                         envs,
                                         single_theta,
                                         single_w,
                                         single_interv_targets,
                                         rng=None):
        """
        log likelihood of a single dataset D_k given the graph, parameters and the intervention
        targets. This is just a wrapper using the Jax-BN model that is the basis of inference.

        Args:
            single_data: single dataset [n_obs, d]
            single_theta: single parameter PyTree
            single_w: single graph [d,d]
            interv_targets: soft mask for interventions [n_env-1, d]
        
        Returns:
            log p(D_k, | single_theta, G, I_k)
        """
        return self.inference_model.log_likelihood_soft_interv_targets(
            data=single_data,
            envs=envs,
            theta=single_theta,
            w=single_w,
            interv_targets=single_interv_targets)

    def log_joint_likelihood(self,
                             single_data,
                             single_theta,
                             single_w,
                             single_interv_targets,
                             envs,
                             rng=None):
        """
        log likelihood of a single dataset D_k given the graph, parameters and the intervention
        targets. This is just a wrapper using the Jax-BN model that is the basis of inference.

        Args:
            single_data: single dataset [n_obs, d]
            single_theta: single parameter PyTree
            single_w: single graph [d,d]
            single_interv_targets: boolean mask for interventions [n_obs, d]
        
        Returns:
            log p(D_k, | single_theta, G, I_k)
        """
        return self.inference_model.log_likelihood(
            data=single_data,
            theta=single_theta,
            w=single_w,
            interv_targets=single_interv_targets,
            envs=envs)

    def target_log_joint_prob_soft_interv(self, single_w, single_theta,
                                          soft_interv_targets, rng):
        """
        log joint probability of theta and D given the graph and parameters,
        using a soft intervention matrix.
        Uses the data passed to the `sample_particles` method
        and stored in `self`.
        
        Args:
            single_w: single graph [d, d]
            single_theta: single parameter PyTree
            interv_targets: soft mask for interventions [n_env-1, d]
            rng: [1, ]
        
        Returns:
            log p(theta, D | G, I): [1,]
        """
        # if self.inference_prior == 'lingauss':
        #     # theta_g [d, d]
        #     theta_g = single_theta[0]
        # else:
        #     theta_g = single_theta[:-1]

        log_prob_theta = self.inference_model.log_prob_interv_parameters(
            theta=single_theta, w=single_w, I=soft_interv_targets)

        # [n_env-1, d]
        # theta_I = single_theta[-1]
        # log_prob_interv = self.inference_model.log_prob_interv_parameters(
        #     theta_I=theta_I, I=soft_interv_targets[1:])

        # go from [n_env, d] to [n_obs, d]
        # soft_interv_targets = soft_interv_targets[self.envs]
        # theta_I = jnp.concatenate(
        #     (jnp.zeros(shape=(1, self.n_vars, 2)).at[...,1].add(1), theta_I), axis=0)
        # theta_I_per_sample = theta_I[self.envs]
        loglik = self.log_joint_likelihood_soft_interv(self.data, self.envs,
                                                       single_theta, single_w,
                                                       soft_interv_targets,
                                                       rng)
        return log_prob_theta + loglik

    def log_joint_prob_soft_interv(self, single_g, single_theta, single_gamma,
                                   eps, t, subk):
        """
        This is the composition of 
            log p(theta, D | G, I)
        and
            G(gamma, U)  (Gumbel-softmax graph sample given gamma)

        Args:
            single_z: single latent tensor [d, k, 2]
            single_theta: single parameter PyTree
            single_gamma: single gamma tensor [n_env-1, d]
            eps: i.i.d Logistic noise of shape [n_env-1, d] 
            t: step 
            subk: rng key

        Returns:
            logprob of shape [1, ]

        """
        soft_interv_sample = self.particle_to_soft_interv(single_gamma, eps, t)
        return self.target_log_joint_prob_soft_interv(single_g, single_theta,
                                                      soft_interv_sample, subk)

    def target_log_joint_prob(self,
                              single_w,
                              single_theta,
                              interv_targets,
                              rng=None):
        """
        log joint probability of theta and D given the graph and parameters.
        Uses the data and intervention targets passed to the `sample_particles` method
        and stored in `self`.

        log p(theta, D | G, I) =  log p(theta | G, I) + log p(D | G, theta, I)

        Args:
            single_w: single graph [d, d]
            single_theta: single parameter PyTree
            interv_targets: [n_env - 1, d]
            rng: [1, ]
        
        Returns:
            log p(theta, D | G, I): [1,]
        """
        return self.target_log_joint_prob_soft_interv(single_w, single_theta,
                                                      interv_targets, rng)

    def eltwise_log_joint_prob(self, gs, single_theta, interv_targets, rng):
        """
        log p(theta, D | G, I) batched over samples of G as well as interv_targets,
        i.e. `self.target_log_joint_prob` batched over the first and third argument.
        Used for calculating expectations of the log posterior with MC samples.

        Args:
            gs: batch of graphs [n, d, d]
            single_theta: single parameter PyTree
            interv_targets: batch of interv_targets [n, n_env - 1, d]
            rng:  [1, ]

        Returns:
            batch of logprobs [n, ]
        """

        return vmap(self.target_log_joint_prob, (0, None, 0, None),
                    0)(gs, single_theta, interv_targets, rng)

    """
    Estimators for dtheta
    """

    def eltwise_grad_theta_likelihood(self, zs, thetas, gammas, t, subk):
        """
        Computes batch of estimators for the score
            
            d/dtheta log p(theta, D| Z, gamma) 

        (i.e. w.r.t the conditional distribution parameters)

        This does not use d/dG log p(theta, D | G, I) and is hence applicable when not defined.
        Analogous to `eltwise_grad_z_likelihood` but w.r.t theta

        Args:
            zs: batch of latent tensors Z [n_particles, d, k, 2]
            thetas: batch of parameter PyTree with `n_mc_samples` as leading dim
            gammas: [n_particles, n_env-1, d]

        Returns:
            batch of gradients in form of PyTree with `n_particles` as leading dim     

        """
        return vmap(self.grad_theta_likelihood, (0, 0, 0, None, 0),
                    0)(zs, thetas, gammas, t, subk)

    def grad_theta_likelihood(self, single_z, single_theta, single_gamma, t,
                              subk):
        """
        Computes Monte Carlo estimator for the score 
            
            d/dtheta log p(theta, D | Z) 

        Uses hard samples of G; reparameterization like for d/dZ is also possible
        Uses same G samples for expectations in numerator and denominator.

        Args:
            single_z: single latent tensor [d, k, 2]
            single_theta: single parameter PyTree
            single_interv_belief: single tensor [n_env - 1, d]
            t: step
            subk: rng key

        Returns:
            parameter gradient PyTree

        """
        # [d, d]
        p_g = self.edge_probs(single_z, t)

        # [n_grad_mc_samples, d, d]
        subk, subk_ = random.split(subk)
        g_samples = self.sample_g(p_g, subk_, self.n_grad_mc_samples)

        # [n_env - 1, d]
        p_I = self.interv_probs(single_gamma, t)

        # [n_grad_mc_samples, n_env - 1, d]
        subk, subk_ = random.split(subk)
        I_samples = self.sample_I(p_I, subk_, self.n_grad_mc_samples)

        # same MC samples for numerator and denominator
        n_mc_numerator = self.n_grad_mc_samples
        n_mc_denominator = self.n_grad_mc_samples

        # [n_mc_numerator, ]
        subk, subk_ = random.split(subk)
        logprobs_numerator = self.eltwise_log_joint_prob(
            g_samples, single_theta, I_samples, subk_)
        logprobs_denominator = logprobs_numerator

        # PyTree  shape of `single_theta` with additional leading dimension [n_mc_numerator, ...]
        # d/dtheta log p(theta, D | G) for a batch of G samples
        # use the same minibatch of data as for other log prob evaluation (if using minibatching)
        grad_theta_log_joint_prob = grad(self.target_log_joint_prob, 1)
        grad_theta = vmap(grad_theta_log_joint_prob, (0, None, 0, None),
                          0)(g_samples, single_theta, I_samples, subk_)

        # stable computation of exp/log/divide and PyTree compatible
        # sums over MC graph samples dimension to get MC gradient estimate of theta
        # original PyTree shape of `single_theta`
        log_numerator = tree_map(
            lambda leaf_theta: logsumexp(a=expand_by(logprobs_numerator,
                                                     leaf_theta.ndim - 1),
                                         b=leaf_theta,
                                         axis=0,
                                         return_sign=True)[0], grad_theta)

        # original PyTree shape of `single_theta`
        sign = tree_map(
            lambda leaf_theta: logsumexp(a=expand_by(logprobs_numerator,
                                                     leaf_theta.ndim - 1),
                                         b=leaf_theta,
                                         axis=0,
                                         return_sign=True)[1], grad_theta)

        # []
        log_denominator = logsumexp(logprobs_denominator, axis=0)

        # original PyTree shape of `single_theta`
        stable_grad = tree_multimap(
            lambda sign_leaf_theta, log_leaf_theta: (sign_leaf_theta * jnp.exp(
                log_leaf_theta - jnp.log(n_mc_numerator) - log_denominator +
                jnp.log(n_mc_denominator))), sign, log_numerator)

        return stable_grad

    """
    Estimators for score d/dZ log p(theta, D | Z)   
    (i.e. w.r.t the latent embeddings Z for graph G)
    """

    def eltwise_grad_z_likelihood(self, zs, thetas, gammas, baselines, t,
                                  subkeys):
        """
        Computes batch of estimators for score
            
            d/dZ log p(theta, D | Z) 

        Selects corresponding estimator used for the term `d/dZ E_p(G|Z)[ p(theta, D | G) ]`
        and executes it in batch.

        Args:
            zs: batch of latent tensors Z [n_particles, d, k, 2]
            thetas: batch of parameters PyTree with `n_particles` as leading dim
            baselines: array of score function baseline values of shape [n_particles, ]

        Returns:
            tuple: batch of (gradient estimates, baselines) of shapes [n_particles, d, k, 2], [n_particles, ]        
        """

        # select the chosen gradient estimator
        if self.grad_estimator_z == 'score':
            grad_z_likelihood = self.grad_z_likelihood_score_function

        elif self.grad_estimator_z == 'reparam':
            grad_z_likelihood = self.grad_z_likelihood_gumbel

        else:
            raise ValueError(
                f'Unknown gradient estimator `{self.grad_estimator_z}`')

        # vmap
        return vmap(grad_z_likelihood, (0, 0, 0, 0, None, 0),
                    (0, 0))(zs, thetas, gammas, baselines, t, subkeys)

    def grad_z_likelihood_score_function(self, single_z, single_theta,
                                         interv_targets, single_sf_baseline, t,
                                         subk):
        raise NotImplementedError(
            "grad_z_likelihood_score_function for inference with intervention targets not implemented yet."
        )

    def grad_z_likelihood_gumbel(self, single_z, single_theta, single_gamma,
                                 single_sf_baseline, t, subk):
        """
        Reparameterization estimator for the score

            d/dZ log p(theta, D | Z) 
            
        Using the Gumbel-softmax / concrete distribution reparameterization trick.
        Uses same G samples for expectations in numerator and denominator.

        Args:
            single_z: single latent tensor [d, k, 2]
            single_theta: single parameter PyTree
            single_sf_baseline: [1, ]

        Returns:
            tuple: gradient, baseline of shape [d, k, 2], [1, ]

        """

        # [n_env - 1, d]
        p_I = self.interv_probs(single_gamma, t)

        # [n_grad_mc_samples, n_env - 1, d]
        subk, subk_ = random.split(subk)
        I_samples = self.sample_I(p_I, subk_, self.n_grad_mc_samples)

        n_vars = single_z.shape[0]

        # same MC samples for numerator and denominator
        n_mc_numerator = self.n_grad_mc_samples
        n_mc_denominator = self.n_grad_mc_samples

        # sample Logistic(0,1) as randomness in reparameterization
        subk, subk_ = random.split(subk)
        eps = random.logistic(subk_,
                              shape=(self.n_grad_mc_samples, n_vars, n_vars))

        # [n_grad_mc_samples, ]
        # since we don't backprop per se, it leaves us with the option of having
        # `soft` and `hard` versions for evaluating the non-grad p(.))
        subk, subk_ = random.split(subk)

        # [d, k, 2], [d, d], [n_grad_mc_samples, n_env-1, d], [n_grad_mc_samples, d, d], [1,], [1,] -> [n_grad_mc_samples]
        logprobs_numerator = vmap(self.log_joint_prob_soft,
                                  (None, None, 0, 0, None, None),
                                  0)(single_z, single_theta, I_samples, eps, t,
                                     subk_)
        logprobs_denominator = logprobs_numerator

        # [n_grad_mc_samples, d, k, 2]
        # d/dx log p(theta, D | G(x, eps)) for a batch of `eps` samples
        # use the same minibatch of data as for other log prob evaluation (if using minibatching)

        # [d, k, 2], [d, d], [n_grad_mc_samples, n_env-1, d], [n_grad_mc_samples, d, d], [1,], [1,] -> [n_grad_mc_samples, d, k, 2]
        grad_z = vmap(grad(self.log_joint_prob_soft,
                           0), (None, None, 0, 0, None, None),
                      0)(single_z, single_theta, I_samples, eps, t, subk_)

        # stable computation of exp/log/divide
        # [d, k, 2], [d, k, 2]
        log_numerator, sign = logsumexp(a=logprobs_numerator[:, None, None,
                                                             None],
                                        b=grad_z,
                                        axis=0,
                                        return_sign=True)

        # []
        log_denominator = logsumexp(logprobs_denominator, axis=0)

        # [d, k, 2]
        stable_grad = sign * jnp.exp(log_numerator - jnp.log(n_mc_numerator) -
                                     log_denominator +
                                     jnp.log(n_mc_denominator))

        return stable_grad, single_sf_baseline

    """
    Estimators for score d/dgamma log p(theta, D | Z, gamma)   
    (i.e. w.r.t the embeddings gamma the interventions I)
    """

    def eltwise_grad_gamma_likelihood(self, zs, thetas, gammas, baselines, t,
                                      subkeys):
        # vmap
        return vmap(self.grad_gamma_likelihood, (0, 0, 0, 0, None, 0),
                    (0, 0))(zs, thetas, gammas, baselines, t, subkeys)

    def grad_gamma_likelihood(self, single_z, single_theta, single_gamma,
                              single_sf_baseline, t, subk):
        """
        Reparameterization estimator for the score

            d/dgamma log p(theta, D | Z, gamma) 
            
        Using the Gumbel-softmax / concrete distribution reparameterization trick.
        Uses same I samples for expectations in numerator and denominator.

        Args:
            single_z: single latent tensor [d, k, 2]
            single_theta: single parameter PyTree
            single_gamma: single gamma tensor [n_env-1, d]
            single_sf_baseline: [1, ]

        Returns:
            tuple: gradient, baseline of shape [d, k, 2], [1, ]

        """

        # [d, d]
        p_g = self.edge_probs(single_z, t)

        # [n_grad_mc_samples, d, d]
        subk, subk_ = random.split(subk)
        g_samples = self.sample_g(p_g, subk_, self.n_grad_mc_samples)

        n_vars = single_z.shape[0]

        # same MC samples for numerator and denominator
        n_mc_numerator = self.n_grad_mc_samples
        n_mc_denominator = self.n_grad_mc_samples

        # sample Logistic(0,1) as randomness in reparameterization
        subk, subk_ = random.split(subk)
        eps = random.logistic(subk_,
                              shape=(self.n_grad_mc_samples, self.n_env - 1,
                                     n_vars))

        # [n_grad_mc_samples, ]
        # since we don't backprop per se, it leaves us with the option of having
        # `soft` and `hard` versions for evaluating the non-grad p(.))
        subk, subk_ = random.split(subk)

        # [n_grad_mc_samples, d, k, 2], [d, d], [n_env-1, d], [n_grad_mc_samples,n_env-1,d], [1,], [1,] -> [n_grad_mc_samples]
        logprobs_numerator = vmap(self.log_joint_prob_soft_interv,
                                  (0, None, None, 0, None, None),
                                  0)(g_samples, single_theta, single_gamma,
                                     eps, t, subk_)
        logprobs_denominator = logprobs_numerator

        # d/dx log p(theta, D | I(x, eps)) for a batch of `eps` samples
        # use the same minibatch of data as for other log prob evaluation (if using minibatching)
        #  [n_grad_mc_samples, n_env-1, d]
        grad_gamma = vmap(grad(self.log_joint_prob_soft_interv,
                               2), (0, None, None, 0, None, None),
                          0)(g_samples, single_theta, single_gamma, eps, t,
                             subk_)

        # stable computation of exp/log/divide
        # [n_grad_mc_samples, n_env-1, d], [n_grad_mc_samples, n_env-1, d]
        log_numerator, sign = logsumexp(a=logprobs_numerator[:, None, None],
                                        b=grad_gamma,
                                        axis=0,
                                        return_sign=True)

        # []
        log_denominator = logsumexp(logprobs_denominator, axis=0)

        # [d, k, 2]
        stable_grad = sign * jnp.exp(log_numerator - jnp.log(n_mc_numerator) -
                                     log_denominator +
                                     jnp.log(n_mc_denominator))

        return stable_grad, single_sf_baseline

    """
    Estimators for the prior d/dgamma log p(gamma)
    """

    def log_dpp_likelihood(self, k, c):
        """
            unnormalized loglik of DPP with eye(d) + (1- eye(d)) * c
        """
        return (jnp.log((1 + c / (1 - c) * k)) + k * jnp.log(1 - c))

    def log_smooth_binomial_likelihood(self, I, a, b):
        """
        Computes unnormalized logprob of seeing I intvs
        using beta distribution as approx. to binomial

        Args:
            a, b: parameters for beta
            I: latent tensor  [n_env-1,]

        Returns:
            logpdf [1,]

        """
        per_env = I / self.n_vars
        log_lik = (a - 1) * jnp.log(per_env) + (b - 1) * jnp.log(1 - per_env)
        return log_lik

    def regularizer_gamma(self, single_gamma, eps, t):
        """ 
        Evaluates Lasso regularizer using 
        Gumbel-softmax instead of Bernoulli samples. 
        The regularizer enforces only `self.interv_per_env` intervention target per environment.

        Args:
            single_gamma: single latent tensor [n_env-1, d]
            single_eps: i.i.d. Logistic noise of shape [n_env-1, d] for Gumbel-softmax
            t: step
        
        Returns:
            constraint value of shape [1,]
        """
        # [n_env-1, d]
        soft_interv_sample = self.particle_to_soft_interv(single_gamma, eps, t)
        # n_env-1
        # scores = vmap(self.log_smooth_binomial_likelihood, (0, None, None),
        #               0)(soft_interv_sample.sum(axis=0), 2, self.n_vars)
        return soft_interv_sample.sum()

    def grad_regularizer_gamma_gumbel(self, single_gamma, key, t):
        """
        Reparameterization estimator for the gradient

           d/dgamma E_p(I|gamma) [regularizer(I)]
            
        Using the Gumbel-softmax / concrete distribution reparameterization trick.

        Args:
            z: single latent tensor [n_env-1, d]                
            key: rng key [1,]    
            t: step

        Returns         
            gradient of constraint [n_env-1, d] 
        """
        n_vars = single_gamma.shape[-1]

        # [n_mc_samples, n_env-1, d]
        eps = random.logistic(key,
                              shape=(self.n_acyclicity_mc_samples,
                                     self.n_env - 1, n_vars))

        # [n_mc_samples, n_env-1, d]
        mc_gradient_samples = vmap(grad(self.regularizer_gamma, 0),
                                   (None, 0, None), 0)(single_gamma, eps, t)

        # [n_env-1, d]
        return mc_gradient_samples.mean(axis=0)

    def log_gamma_beta_prob(self, gamma, t):
        """
        Computes unnormalized logprob of sigmoid(gamma) for beta distribution
        with a = 1 / d and b = 1 - a

        -> beta distribution is U-shaped with expected value = 1 / d

        Args:
            gammas: latent tensor  [n_env-1, d]
            t: [1,]

        Returns:
            logpdf [1,]

        """
        # [n_env - 1, d]
        p_I = self.interv_probs(gamma, t)
        # beta pdf is ill defined for the boundaries {0,1}
        p_I = jnp.maximum(p_I, 1e-8)
        p_I = jnp.minimum(p_I, (1 - 1e-8))
        a = 1 / self.n_vars
        # a = 0.5
        b = 1 - a
        ind_bern = (a - 1) * jnp.log(p_I) + (b - 1) * jnp.log(1 - p_I)
        return jnp.sum(ind_bern)

    def eltwise_grad_gamma_prior(self, gammas, subkeys, t):
        """
        Computes batch of estimators for the score

            d/dgamma log p(gamma)
        
        where log p(gamma) = - beta(t) E_p(I|gamma) [regularizer(I)]
                         + log f(gamma) 
        
        and f(gamma) is an additional prior factor (still TODO)

        Args:
            gammas: single latent tensor  [n_particles, n_env-1, d]
            subkeys: batch of rng keys [n_particles, ...]

        Returns:
            batch of gradients of shape [n_particles, n_env-1, d]

        """
        # constraint term
        # [n_particles, n_env-1, d], [n_particles,], [1,] -> [n_particles, n_env-1, d]
        eltwise_grad_regularizer = vmap(self.grad_regularizer_gamma_gumbel,
                                        (0, 0, None), 0)(gammas, subkeys, t)

        eltwise_grad_prior = vmap(grad(self.log_gamma_beta_prob, 0), (0, None),
                                  0)(gammas, t)
        return - self.lambda_regul * eltwise_grad_regularizer + eltwise_grad_prior \
                - gammas / (self.gamma_prior_std**2)
