from jax.scipy.special import logsumexp
import scipy
from scipy.stats import multivariate_normal

import jax.numpy as jnp
from jax import random, vmap
from jax.ops import index, index_update
from jax.scipy.stats import norm as jax_normal
from jax.scipy.stats import multivariate_normal as jax_multivar_normal
from jax.scipy.stats import gamma as jax_gamma
from jax.lax import fori_loop

from .basic import BasicModel


class LinearGaussian(BasicModel):
    """
    Linear Gaussian BN model (continuous variables)
    corresponding to linear structural equation model (SEM) with Gaussian noise

    Each variable distributed as Gaussian with mean being the linear combination of its parents 
    weighted by a Gaussian parameter vector (i.e., with Gaussian-valued edges).
    """

    def __init__(self, *, g_dist, obs_noise=.1, mean_edge=0., sig_edge=1., verbose=False):
        super().__init__(g_dist=g_dist, verbose=verbose)

        self.obs_noise = obs_noise
        self.mean_edge = mean_edge
        self.sig_edge = sig_edge

    def sample_parameters_normal(self, *, key, g):
        """Samples parameters given graph g, here corresponding to edge weights
           To have the edge weights bounded away from zero, we add a
           sign(theta) * 0.5 offset to the Gaussian sample

        Args:
            g (igraph.Graph): grpah
            key: rng key

        Returns:
            theta : [n_vars, n_vars] to have consistent shape
        """
        key, subk = random.split(key)
        theta = self.mean_edge + self.sig_edge * random.normal(subk, shape=(len(g.vs), len(g.vs)))
        return theta + 0.5 * jnp.sign(theta)
    
    def sample_parameters(self, *, key, g):
        """Samples parameters given graph g, here corresponding to edge weights
           Compared to the method above, we sample here uniformly in
           [-2 * sig_edge, -0.5] U [0.5, 2 * sig_edge]

        Args:
            g (igraph.Graph): grpah
            key: rng key

        Returns:
            theta : [n_vars, n_vars] to have consistent shape
        """
        key, subk = random.split(key)
        theta = random.uniform(subk, shape=(len(g.vs), len(g.vs))) * (2 * self.sig_edge - 0.5) + 0.5
        key, subk = random.split(key)
        theta = theta * random.choice(subk, a=jnp.array([-1,1]), shape=(len(g.vs), len(g.vs)))
        return theta

    def sample_obs(self, *, key, n_samples, g, theta, toporder=None, interv={}):
        """Samples `n_samples` observations given graph and parameters
        Args:
            key: rng
            n_samples (int): number of samples
            g (igraph.Graph): graph
            theta : parameters
            interv: {intervened node : clamp value} or {intervened node : {'mean': val, 'noise': val}}

        Returns:
            x : [n_samples, d] 
        """
        if toporder is None:
            toporder = g.topological_sorting()

        x = jnp.zeros((n_samples, len(g.vs)))

        key, subk = random.split(key)
        z = jnp.sqrt(self.obs_noise) * random.normal(subk, shape=(n_samples, len(g.vs)))

        # ancestral sampling
        for j in toporder:

            # intervention
            if j in interv.keys():
                if type(interv[j]) is dict:
                    # z = z.at[:, j].set(z[:,j] + 10) 
                    key, subk = random.split(key)
                    z_ = jnp.sqrt(interv[j]['noise']) * random.normal(
                        subk, shape=(n_samples,))
                    x = index_update(x, index[:, j], interv[j]['mean'] + z_)
                else:
                    x = index_update(x, index[:, j], interv[j])    
                continue
            
            # regular ancestral sampling
            parent_edges = g.incident(j, mode='in')
            parents = list(g.es[e].source for e in parent_edges)

            if parents:
                mean = x[:, jnp.array(parents)] @ theta[jnp.array(parents), j]
                x = index_update(x, index[:, j], mean + z[:, j])
            else:
                x = index_update(x, index[:, j], z[:, j])

        return x
    
    def log_prob_parameters(self, *, theta, g):
        """log p(theta | g)
        Assumes N(mean_edge, sig_edge^2) distribution for any given edge 
        In the linear Gaussian model, g does not matter.
        
        Args:
            g (igraph.Graph): graph
            theta : parameters
        
        Returns:
            [1, ]
        """
        logprob = 0.0
        # done analogously to `sample_obs` to ensure indexing into theta matrix 
        # is consistent and not silently messed up by e.g. ig.Graph vertex index naming
        for j in range(len(g.vs)):
            parent_edges = g.incident(j, mode='in')
            parents = list(g.es[e].source for e in parent_edges)
            if parents:
                logprob += scipy.stats.norm.logpdf(x=theta[jnp.array(parents), j], loc=self.mean_edge, scale=self.sig_edge).sum()

        # assert(logprob == jnp.sum(graph_to_mat(g) * scipy.stats.norm.logpdf(x=theta, loc=self.mean_edge, scale=self.sig_edge)).item())
        return logprob


    def log_likelihood(self, *, x, theta, g):
        """log p(x | theta, G)
        Assumes N(mean_obs, obs_noise) distribution for any given observation
        
        Args:
            g (igraph.Graph): graph
            theta : parameters
            x : [n_samples, d] 

        Returns:
            [1, ]
        """
        n_vars = x.shape[-1]    
        logp = 0.0

        for j in range(n_vars):
            parent_edges = g.incident(j, mode='in')
            parents = list(g.es[e].source for e in parent_edges)
            
            if parents:
                mean = x[:, jnp.array(parents)] @ theta[jnp.array(parents), j]
            else:
                mean = jnp.zeros(x.shape[0])

            # since observations iid, faster not to use multivariate_normal
            logp += scipy.stats.norm.logpdf(x=x[..., j], loc=mean, scale=jnp.sqrt(self.obs_noise)).sum()

        return logp

    def log_marginal_likelihood_given_g_single(self, g, x, j, interv_targets):
        """Computes log p(x | G) in closed form for a single node 
        using conjugacy properties of Gaussian

        Args:
            g (igraph.Graph): graph
            x : [n_samples, d] 
            j (int): node index for score
            interv_targets: [n_samples, d] or [1, d]

        Returns:
            [1, ]
        """
        parent_edges = g.incident(j, mode='in')
        parents = list(g.es[e].source for e in parent_edges)
        n_parents = len(parents)

        # mask data that has not been intervened on
        x = x[~interv_targets[..., j]]
        n_samples_left, _ = x.shape

        # mean
        mean_theta_j = self.mean_edge * jnp.ones(n_parents)
        mean_j = x[..., parents] @ mean_theta_j

        # cov
        # Note: `cov_j` is a NxN cov matrix, which can be huge for large N
        cov_theta_j = self.sig_edge**2.0 * jnp.eye(n_parents)
        cov_j = self.obs_noise * jnp.eye(n_samples_left) + \
            x[..., parents] @ cov_theta_j @ x[..., parents].T

        # log prob
        return multivariate_normal.logpdf(x=x[..., j], mean=mean_j, cov=cov_j)

    def log_marginal_likelihood_given_g(self, g, x, interv_targets=None):
        """Computes log p(x | G) in closed form using properties of Gaussian

        Args:
            g (igraph.Graph): graph
            x : [n_samples, d] 
            interv_targets: [n_samples, d] or [1, d]

        Returns:
            [1, ]
        """
        _, n_vars = x.shape
        if interv_targets is None:
            interv_targets = jnp.zeros(n_vars).astype(bool)
        logp = 0.0
        for j in range(n_vars):
            logp += self.log_marginal_likelihood_given_g_single(
                g=g, x=x, j=j, interv_targets=interv_targets)
        return logp


"""	
+++++   JAX implementation +++++	
"""


class LinearGaussianJAX:
    """	
    LinearGaussians above but using JAX and adjacency matrix representation	
    """

    def __init__(self,
                 *,
                 obs_noise,
                 mean_edge,
                 sig_edge,
                 init_sig_edge=0.3,
                 interv_prior_mean=0,
                 interv_prior_std=10,
                 interv_mean=None,
                 interv_noise=None,
                 verbose=False):
        super().__init__()

        self.obs_noise = obs_noise
        self.mean_edge = mean_edge
        self.sig_edge = sig_edge
        self.init_sig_edge = init_sig_edge
        self.verbose = verbose
        self.interv_mean = interv_mean or 0.
        self.interv_prior_mean = interv_prior_mean
        self.interv_prior_std = interv_prior_std
        self.interv_noise = interv_noise or 1.0

    def _log_interv_lik(self, data):
        if self.interv_noise is not None and self.interv_noise != 0:
            return jax_normal.logpdf(x=data,
                                     loc=self.interv_mean,
                                     scale=jnp.sqrt(self.interv_noise))
        else:
            return jnp.zeros(shape=data.shape)

    def get_theta_shape(self, *, n_vars):
        """PyTree of parameter shape"""
        return jnp.array((n_vars, n_vars))


    def init_parameters(self, *, key, n_vars, n_particles, batch_size=0):
        """Samples batch of random parameters given dimensions of graph, from p(theta | G) 
        Args:
            key: rng
            n_vars: number of variables in BN
            n_particles: number of parameter particles sampled
            batch_size: number of batches of particles being sampled

        Returns:
            theta : [n_particles, n_vars, n_vars]
        """
        key, subk = random.split(key)

        if batch_size != 0:
            raise ValueError("batch_size for lingauss not implemented")

        theta = self.mean_edge + self.init_sig_edge * random.normal(
            subk, shape=(n_particles, *self.get_theta_shape(n_vars=n_vars)))

        return theta


    def init_interv_parameters(self,
                               *,
                               key,
                               n_env,
                               n_vars,
                               n_particles,
                               batch_size=0):
        """Samples batch of random parameters given dimensions of graph, from p(theta | G) 
        Args:
            key: rng
            n_vars: number of variables in BN
            n_env: number of environments (including observational one)
            n_particles: number of parameter particles sampled
            batch_size: number of batches of particles being sampled

        Returns:
            theta : pytree with [theta_g, theta_sig, theta_interv_mean, theta_interv_sig]
                            where theta_g is [d,d]; theta_sig is [d,]; 
                            theta_interv_mean is [n_env, d]; theta_interv_sig is [n_env, d]
            
        """
        key, subk = random.split(key)

        if batch_size != 0:
            raise ValueError("batch_size for lingauss not implemented")

        # weights
        theta_g = self.mean_edge + self.init_sig_edge * random.normal(
            subk, shape=(n_particles, *self.get_theta_shape(n_vars=n_vars)))

        # mean of interventions
        key, subk = random.split(key)
        theta_interv_mean = jnp.sqrt(0.1) * random.normal(
            subk,
            shape=(n_particles, n_env - 1, n_vars)) + self.interv_prior_mean

        # sigma of interventions
        # key, subk = random.split(key)
        # theta_interv_sig = jnp.sqrt(0.01) * random.normal(
        #     subk, shape=(n_particles, n_env - 1, n_vars)) + env_stds[1:][None]
        theta = [theta_g, theta_interv_mean]
        return theta



    def log_marginal_likelihood_given_g_single(self, w, data, j, interv_targets):
        """ Computes log p(x | G) in closed form using properties of Gaussian
            for a fixed node j

        Args:
            data : [n_samples, n_vars]	
            w:     [n_vars, n_vars]
            interv_targets : [n_samples, n_vars]

        Returns:
            [1, ]
        """
        n_samples, n_vars = data.shape
        
        # mask data depending on interventions ?
        mask_data = data * (1 - interv_targets[..., j])[..., None]
        # mean
        # [n_vars,]
        mean_theta = self.mean_edge * w[:, j]
        # [n_samples, 1]
        mean = (1 - interv_targets[..., j]) * (data @ mean_theta) # + interv_targets[..., j] * self.interv_mean 
        
        # cov
        # Note: `cov` is a NxN cov matrix, which can be huge for large N
        cov_theta = w[:, j] * self.sig_edge**2.0
        # [n_samples, n_samples]
        cov = self.obs_noise * jnp.eye(n_samples) + \
            mask_data @ jnp.diag(cov_theta) @ mask_data.T
        return jax_multivar_normal.logpdf(x=mask_data[..., j],
                                          mean=mean,
                                          cov=cov)

    def log_marginal_likelihood_given_g(self, *, w, data, interv_targets=None):
        """Computes log p(x | G) in closed form using properties of Gaussian

        Args:
            data :          [n_samples, n_vars]	
            w:              [n_vars, n_vars]	
            interv_targets: [n_samples, n_vars]

        Returns:
            [1, ]        
        """
        n_samples, n_vars = data.shape
        if interv_targets is None:
            interv_targets = jnp.zeros(n_vars).astype(bool)

        # [n_vars,]
        scores = vmap(
            lambda j: self.log_marginal_likelihood_given_g_single(
                w=w, data=data, j=j, interv_targets=interv_targets), (0, ),
            0)(jnp.arange(n_vars))

        # sum scores for all nodes
        return jnp.sum(scores)

    def log_prob_parameters(self, *, theta, w):
        """log p(theta | g)
        Assumes N(mean_edge, sig_edge^2) distribution for any given edge 
        In the linear Gaussian model, g does not matter.
        
        Args:
            theta: [n_vars, n_vars]
            w: [n_vars, n_vars]
            
        Returns:
            [1, ]
        """
        score_g = jnp.sum(w * jax_normal.logpdf(
            x=theta[0], loc=self.mean_edge, scale=self.sig_edge))
        return score_g

    def log_prob_interv_parameters(self, *, theta, w, I):
        """log p(theta | G, I)
        Assumes N(mean_val, sig_edge^2) distribution for any given edge 
        
        
        Args:
            theta:  pytree with [theta_g, theta_sig, theta_interv_mean, theta_interv_sig]
                            where theta_g is [d,d]; theta_sig is [d,]; 
                            theta_interv_mean is [n_env, d]; theta_interv_sig is [n_env, d]
            I:      [n_env-1, n_vars]
            
        Returns:
            [1, ]
        """

        [theta_g, theta_interv_mean] = theta

        score_g = jnp.sum(w * jax_normal.logpdf(
            x=theta_g, loc=self.mean_edge, scale=self.sig_edge))

        score_interv_mean = jnp.sum(
            I * jax_normal.logpdf(x=theta_interv_mean,
                                  loc=self.interv_prior_mean,
                                  scale=self.interv_prior_std))
        # score_interv_sig = jnp.sum(
        #     jax_gamma.logpdf(x=theta_interv_sig**2,
        #                      a=2,
        #                      scale=self.interv_noise))
        return score_g + score_interv_mean #+ score_interv_sig

    def log_likelihood(self, *, data, theta, w, interv_targets=None, envs=None):
        """Computes p(x | theta, G). Assumes N(mean_obs, obs_noise) distribution for any given observation
            data :          [n_observations, n_vars]
            theta:          pytree with [theta_g, theta_sig], where theta_g is [d,d] and theta_sig is [d,]
            w:              [n_vars, n_vars]
            interv_targets: [n_observations, n_vars]
        """
        n_vars = data.shape[-1]
        if type(theta) is list:
            theta = theta[0]
        if interv_targets is None or envs is None:
            interv_targets = jnp.zeros(n_vars)
        else:
            # extend with no targets for observ. env
            interv_targets = jnp.concatenate(
                (jnp.zeros(shape=(1, n_vars)), interv_targets), axis=0)
            interv_targets = interv_targets[envs]
        # sum scores for all nodes
        return jnp.sum(
            jnp.where(
                # [1, n_vars]
                interv_targets,
                # TODO something like this for shift interventions
                # jax_normal.logpdf(x=data, loc=10 + data @ (w * theta), scale=jnp.sqrt(self.obs_noise)),
                0,
                # self.log_interv_lik(data),
                # [n_observations, n_vars]
                jax_normal.logpdf(x=data,
                                  loc=data @ (w * theta),
                                  scale=jnp.sqrt(self.obs_noise))))

    def log_likelihood_soft_interv_targets(self, *, data, theta, w,
                                           interv_targets=None, envs=None):
        """Computes p(x | theta, G). Assumes N(mean_obs, obs_noise)
            distribution for any given observation. 
            This functions works with an interv_targets mask
            that has non_boolean values in [0,1]
            data :          [n_observations, n_vars]
            envs:           [n_observations, ]
            theta:          pytree with [theta_g, theta_sig, theta_interv_mean, theta_interv_sig]
                            where theta_g is [d,d]; theta_sig is [d,]; 
                            theta_interv_mean is [n_env, d]; theta_interv_sig is [n_env, d]
            w:              [n_vars, n_vars]
            interv_targets: [n_observations, n_vars]
        """
        n_vars = w.shape[0]
        [theta_g,  theta_interv_mean] = theta

        if envs is None:
            envs = jnp.zeros(data.shape[0])
        if interv_targets is None:
            interv_targets = jnp.zeros(data.shape)
        ## obs. loglik
        log_nointerv_lik = jax_normal.logpdf(x=data,
                                             loc=data @ (w * theta_g),
                                             scale=jnp.sqrt(self.obs_noise))

        ## interv. loglik
        # add observational environment
        interv_targets = jnp.concatenate(
            (jnp.zeros(shape=(1, n_vars)), interv_targets), axis=0)

        theta_interv_mean = jnp.concatenate(
            (jnp.zeros(shape=(1, n_vars)), theta_interv_mean), axis=0)

        log_interv_lik = jax_normal.logpdf(x=data,
                                           loc=theta_interv_mean[envs],
                                           scale=jnp.sqrt(self.interv_noise))
        
        # go from [n_envs, d] to [n_obs, d]
        interv_targets = interv_targets[envs]

        return jnp.sum((1 - interv_targets) * log_nointerv_lik +
                       interv_targets * log_interv_lik)
