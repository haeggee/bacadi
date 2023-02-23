import jax.numpy as jnp
from jax import vmap
from jax import random
from jax.ops import index, index_update
from jax.scipy.stats import norm as jax_normal


class SobolevGaussianJAX:
    """	
    Non-linear Gaussian BN with interactions modeled
    with a orthogonal basis expansion with the Sobolev basis.
    See: https://arxiv.org/abs/1909.13189

    Let {\phi_r}_{r=1}^\inf
    be an orthonormal basis of H^1(R^d) such that E[\phi_r (X)]=0
    for each r. Then any f \in H^1(R^d) can be written uniquely
    as
        f(x)=\sum_{r=1}^{\inf} \alpha_r \phi_r(x)
    
    If the \alpha_r decay fast enough, one can use a finite series
    to approximate the function f.

    We here implement an additive model with one-dimensional
    expansions. It is based on the Sobolev basis
    
        \phi_r(x) = s_r \sin(x / s_r) with s_r = 2 / ((2r - 1)\pi)
    
    The moddel is additive, i.e. we express a node j given its parents:

        f_j(x_1, ..., x_d) = \sum_{i \in pa(j)} \sum_r \alpha_{j,i,r} \phi_r(x_i)
    
    The summation over r is approximated via a fixed sum up to `n_exp`.

    Moreover, we assume additive Gaussian noise with parameter \sigma.
    """
    def __init__(self,
                 *,
                 obs_noise,
                 n_vars,
                 n_exp,
                 mean_param,
                 sig_param,
                 init_sig_param=1.0,
                 init='normal',
                 interv_mean=None,
                 interv_noise=None,
                 verbose=False):
        super().__init__()

        self.n_vars = n_vars
        self.n_exp = n_exp

        self.obs_noise = obs_noise
        self.mean_param = mean_param
        self.sig_param = sig_param
        self.init_sig_param = init_sig_param
        self.verbose = verbose

        self.init = init

        self.interv_mean = interv_mean or 0.
        self.interv_noise = interv_noise

    def _log_interv_lik(self, data):
        if self.interv_noise is not None and self.interv_noise != 0:
            return jax_normal.logpdf(x=data,
                                     loc=self.interv_mean,
                                     scale=jnp.sqrt(self.interv_noise))
        else:
            return jnp.zeros(shape=data.shape)

    def sobolev_basis(self, x):
        """
        TODO docstring
        Args:
            x: [n_obs, ]
        Returns:
            Sobolev basis of shape [n_obs, n_exp]
        """

        # [n_exp, ]
        mu = 2.0 / (2 * jnp.arange(self.n_exp) + 1) / jnp.pi
        # x: [n_obs,]
        psi = mu * jnp.sin(x[..., None] / mu)
        # [n_obs, n_exp]
        return psi

    def eltwise_sobolev_basis(self, x):
        """
        TODO docstring
        Args:
            x: [n_obs, n_vars]
        Returns:
            Sobolev basis of shape [n_obs, n_vars, n_exp]
        """
        psi = vmap(self.sobolev_basis, (1, ), 1)(x)
        return psi

    def get_theta_shape(self, *, n_vars):
        """ Returns tree shape of the parameters of the neural networks
        Args:
            n_vars

        Returns:
            PyTree of parameter shape
        """
        return jnp.array((n_vars, n_vars, self.n_exp))

    def init_parameters(self, *, key, n_vars, n_particles, batch_size=0):
        """ Samples batch of random parameters given dimensions of graph,
            from p(theta | G) 
        Args:
            key: rng
            n_vars: number of variables in BN
            n_particles: number of parameter particles sampled
            batch_size: number of batches of particles being sampled

        Returns:
            theta : PyTree with leading dimension of `n_particles`
        """

        if batch_size == 0:
            theta = self.mean_param + self.init_sig_param * random.normal(
                key, shape=(n_particles, *self.get_theta_shape(n_vars=n_vars)))
        else:
            theta = self.mean_param + self.init_sig_param * random.normal(
                key,
                shape=(batch_size, n_particles,
                       *self.get_theta_shape(n_vars=n_vars)))
        return theta

    def sample_parameters(self, *, key, g):
        """Samples parameters for Sobolev model. Here, g is ignored.
        Args:
            g (igraph.Graph): graph
            key: rng

        Returns:
            theta : [n_vars, n_vars, n_exp] to have consistent shape
        """
        n_vars = len(g.vs)
        if self.init == 'uniform':
            key, subk = random.split(key)
            theta = random.uniform(subk, shape=(
                n_vars, n_vars, self.n_exp)) * (2 * self.sig_param - 0.5) + 0.5
            key, subk = random.split(key)
            theta = theta * random.choice(subk,
                                          a=jnp.array([-1, 1]),
                                          shape=(len(g.vs), len(
                                              g.vs), self.n_exp))
        elif self.init == 'normal':
            key, subk = random.split(key)
            theta = self.mean_param + self.sig_param * random.normal(
                subk, shape=(n_vars, n_vars, self.n_exp))
            theta = theta + 0.5 * jnp.sign(theta)
        else:
            raise ValueError(f'Unknown init type `{self.init}`')
        return theta

    def sample_obs(self,
                   *,
                   key,
                   n_samples,
                   g,
                   theta,
                   toporder=None,
                   interv={}):
        """
        Samples `n_samples` observations by doing single forward
        passes in topological order
        Args:
            key: rng
            n_samples (int): number of samples
            g (igraph.Graph): graph
            theta : parameters [n_vars, n_vars, n_exp]
            interv: {intervened node : clamp value} or {intervened node : {'mean': val, 'noise': val}}

        Returns:
            x : [n_samples, d] 
        """
        if toporder is None:
            toporder = g.topological_sorting()

        x = jnp.zeros((n_samples, len(g.vs)))

        key, subk = random.split(key)
        z = jnp.sqrt(self.obs_noise) * random.normal(
            subk, shape=(n_samples, len(g.vs)))

        # ancestral sampling
        for j in toporder:

            # intervention
            if j in interv.keys():
                if type(interv[j]) is dict:
                    key, subk = random.split(key)
                    z_ = jnp.sqrt(interv[j]['noise']) * random.normal(
                        subk, shape=(n_samples, ))
                    x = index_update(x, index[:, j], interv[j]['mean'] + z_)
                else:
                    x = index_update(x, index[:, j], interv[j])
                continue

            # regular ancestral sampling
            parent_edges = g.incident(j, mode='in')
            parents = list(g.es[e].source for e in parent_edges)

            if parents:
                # [n_samples, n_pars, n_exp]
                sobolev_basis = self.eltwise_sobolev_basis(
                    x[:, jnp.array(parents)])
                # [n_samples, n_pars, n_exp], [n_pars, n_exp] -> [n_samples,]
                mean = jnp.einsum('nik,ik->n', sobolev_basis,
                                  theta[jnp.array(parents), j])
                x = index_update(x, index[:, j], mean + z[:, j])
            else:
                x = index_update(x, index[:, j], z[:, j])

        return x

    def log_prob_parameters(self, *, theta, w):
        """log p(theta | g)
        Assumes N(mean_edge, sig_edge^2) distribution for any given edge 

        Args:
            theta: parameters [n_vars, n_vars, n_exp]
            w: adjacency matrix of graph [n_vars, n_vars]

        Returns:
            logprob [1,]
        """
        return jnp.sum(w[..., None] * jax_normal.logpdf(
            x=theta, loc=self.mean_param, scale=self.sig_param))

    def log_likelihood(self, *, data, theta, w, interv_targets):
        """ Computes p(x | theta, G). Assumes N(mean_obs, obs_noise)
            distribution for any given observation

        Args:
            data :          [n_observations, n_vars]
            theta:          [n_vars, n_vars, n_exp]
            w:              [n_vars, n_vars]
            interv_targets: [n_observations, n_vars]

        Returns:
            logprob [1,]
        """

        if interv_targets is None:
            interv_targets = jnp.zeros(data.shape[-1])
        # [n_obs, n_vars, n_exp]
        sobolev_basis = self.eltwise_sobolev_basis(data)
        # [n_vars,n_vars], [n_vars, n_vars, n_exp] -> [n_vars, n_vars, n_exp]
        masked_theta = w[..., None] * theta
        # [n_obs, n_vars, n_exp], [n_vars, n_vars, n_exp] -> [n_obs, n_vars]
        mean = jnp.einsum('nik,ijk->nj', sobolev_basis, masked_theta)

        # sum scores for all nodes
        return jnp.sum(
            jnp.where(
                # [n_obs, n_vars]
                interv_targets,
                0,
                # [n_obs, n_vars]
                jax_normal.logpdf(x=data,
                                  loc=mean,
                                  scale=jnp.sqrt(self.obs_noise))))

    def log_likelihood_soft_interv_targets(self, *, data, theta, w,
                                           interv_targets):
        """log p(x | theta, G, I)
        Assumes N(mean_obs, obs_noise^2) distribution for any given observation
        
        Args:
            data: observations [N, d]
            theta: [n_vars, n_vars, n_exp]
            w: adjacency matrix [d, d]
            interv_targets: soft indicator of intervention locations [N, d]
        
        Returns:
            logprob [d,]
        """
        if interv_targets is None:
            interv_targets = jnp.zeros(data.shape[-1])
        # [n_obs, n_vars, n_exp]
        sobolev_basis = self.eltwise_sobolev_basis(data)
        # [n_vars,n_vars], [n_vars, n_vars, n_exp] -> [n_vars, n_vars, n_exp]
        masked_theta = w[..., None] * theta
        # [n_obs, n_vars, n_exp], [n_vars, n_vars, n_exp] -> [n_obs, n_vars]
        mean = jnp.einsum('nik,ijk->nj', sobolev_basis, masked_theta)

        log_nointerv_lik = jax_normal.logpdf(x=data,
                                             loc=mean,
                                             scale=jnp.sqrt(self.obs_noise))
        # sum scores for all nodes and data
        return jnp.sum((1 - interv_targets) * log_nointerv_lik)
