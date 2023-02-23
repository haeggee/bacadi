import functools 

import numpy as np
import scipy

from .basic import BasicModel

import jax.numpy as jnp
from jax import jit, vmap
import jax.lax as lax
from jax.scipy.special import gammaln
from jax.scipy.stats import norm as jax_normal
from bacadi.utils.func import leftsel

class BGe(BasicModel):
    """
    Linear Gaussian-Gaussian model (continuous)

    Each variable distributed as Gaussian with mean being the linear combination of its parents 
    weighted by a Gaussian parameter vector (i.e., with Gaussian-valued edges).

    The parameter prior over (mu, lambda) of the joint Gaussian distribution (mean `mu`, precision `lambda`) over x is Gaussian-Wishart, 
    as introduced in 
        Geiger et al (2002):  https://projecteuclid.org/download/pdf_1/euclid.aos/1035844981

    Computation is based on
        Kuipers et al (2014): https://projecteuclid.org/download/suppdf_1/euclid.aos/1407420013 

    Note: 
        - (mu, Sigma) of joint is not factorizable into independent theta, but there exists a one-to-one correspondence.
        - lambda = Sigma^{-1}
        - assumes default diagonal parametric matrix T

    Some inspiration was drawn from: https://bitbucket.org/jamescussens/pygobnilp/src/master/pygobnilp/scoring.py 

    For interventional data, for each local score of node `j`, the datapoints where node `j`
    was intervened on are dropped. For a discussion of this, see
        https://link.springer.com/content/pdf/10.1007%2F978-1-60761-800-3.pdf
    Chapter 6, paragraph 3.2 "The BGe Metric for Static Interventional Data" on page 134.
    """

    def __init__(self, *,
            g_dist,
            mean_obs,
            alpha_mu,
            alpha_lambd,
            verbose=False
            ):
        super().__init__(g_dist=g_dist, verbose=verbose)

        self.n_vars = g_dist.n_vars
        self.mean_obs = mean_obs
        self.alpha_mu = alpha_mu
        self.alpha_lambd = alpha_lambd

        assert(self.alpha_lambd > self.n_vars + 1)

        # pre-compute matrices
        self.small_t = (self.alpha_mu * (self.alpha_lambd - self.n_vars - 1)) / (self.alpha_mu + 1)
        self.T = self.small_t * np.eye(self.n_vars)

    def log_marginal_likelihood_given_g_single(self, g, x, j, R=None, interv_targets=None):
        """Computes log p(x | G) in closed form using properties of Gaussian-Wishart
        
        Args:
            g (igraph.Graph): graph
            x: observations [N, d]
            j (int): node index for node score
            R: internal matrix for BGe score [d, d], only precomputed if interv_targets is None
            interv_targets: boolean mask of shape [N,d] of whether or not a node was intervened on
                    datapoints where node j was intervened are ignored in likelihood computation

        Returns:
            [1, ]        
        """

        if interv_targets is not None:
            # only take samples where j has not been intervened on
            x = x[~interv_targets[..., j]]

        N, d = x.shape
        assert(d == self.n_vars)

        if R is None:
            # with interventions, R has to be computed individually for each node
            x_bar = x.mean(axis=0, keepdims=True)
            x_center = x - x_bar
            s_N = x_center.T @ x_center  # [d, d]

            # Kuipers (2014) states R wrongly in the paper, using alpha_lambd rather than alpha_mu;
            # the supplementary contains the correct term
            R = self.T + s_N + ((N * self.alpha_mu) / (N + self.alpha_mu)) * \
                ((x_bar - self.mean_obs).T @ (x_bar - self.mean_obs))  # [d, d]

        parent_edges = g.incident(j, mode='in')
        parents = list(g.es[e].source for e in parent_edges)
        l = len(parents)

        # compute gamma term + log ratio of det(T) (since we use default prior matrix T)
        log_gamma_terms_const = 0.5 * (np.log(self.alpha_mu) - np.log(N + self.alpha_mu))
        log_gamma_term = (
            log_gamma_terms_const
            + scipy.special.loggamma(0.5 * (N + self.alpha_lambd - d + l + 1))
            - scipy.special.loggamma(0.5 * (self.alpha_lambd - d + l + 1))
            - 0.5 * N * np.log(np.pi)
            # log det(T_JJ)^(..) / det(T_II)^(..) for default T
            + 0.5 * (self.alpha_lambd - d + 2 * l + 1) * \
            np.log(self.small_t)
        )

        # leaf node case
        if l == 0:
            # log det(R)^(...)
            R_II = R[j, j]
            log_term_r = - 0.5 * \
                (N + self.alpha_lambd - d + 1) * np.log(np.abs(R_II))
        else:
            # log det(R_II)^(..) / det(R_JJ)^(..)
            log_term_r = (
                0.5 * (N + self.alpha_lambd - d + l) *
                np.linalg.slogdet(R[np.ix_(parents, parents)])[1]
                - 0.5 * (N + self.alpha_lambd - d + l + 1) *
                np.linalg.slogdet(
                    R[np.ix_([j] + parents, [j] + parents)])[1]
            )

        return log_gamma_term + log_term_r

    def log_marginal_likelihood_given_g(self, g, x, interv_targets=None):
        """Computes log p(x | G) in closed form using properties of Gaussian-Wishart

        Args:
            g (igraph.Graph): graph
            x: observations [N, d]
            interv_targets: [N, d] boolean mask indicating interventions

        Returns:
            [1, ]        
        """

        N, d = x.shape
        assert (d == self.n_vars)

        if interv_targets is None:
            x_bar = x.mean(axis=0, keepdims=True)
            x_center = x - x_bar
            s_N = x_center.T @ x_center  # [d, d]
            # Kuipers (2014) states R wrongly in the paper, using alpha_lambd rather than alpha_mu;
            # the supplementary contains the correct term
            R = self.T + s_N + ((N * self.alpha_mu) / (N + self.alpha_mu)) * \
                ((x_bar - self.mean_obs).T @ (x_bar - self.mean_obs))  # [d, d]
        else:
            # R has to be computed individually for each node j
            R = None
        # node-wise score
        logp = 0.0
        for j in range(self.n_vars):
            # node has been intervened on in all samples
            if interv_targets is not None and (interv_targets[..., j]).all():
                continue
            logp += self.log_marginal_likelihood_given_g_single(
                g=g, x=x, j=j, R=R, interv_targets=interv_targets)

        return logp


class BGeJAX:
    """
    JAX implementation of BGe that allows for @jax.jit
    """

    def __init__(self, *,
            n_vars,
            mean_obs,
            alpha_mu,
            alpha_lambd,
            interv_mean=None,
            interv_noise=None,
            verbose=False
            ):
        super().__init__()

        self.n_vars = n_vars
        self.mean_obs = mean_obs
        self.alpha_mu = alpha_mu
        self.alpha_lambd = alpha_lambd
        assert(self.alpha_lambd > self.n_vars + 1)

        self.interv_mean = interv_mean or 0.
        self.interv_noise = interv_noise
        
        # pre-compute matrices
        self.small_t = (self.alpha_mu * (self.alpha_lambd - self.n_vars - 1)) / (self.alpha_mu + 1)
        self.T = self.small_t * np.eye(self.n_vars)

    def _log_interv_lik(self, data):
        if self.interv_noise is not None and self.interv_noise != 0:
            return jax_normal.logpdf(x=data,
                                     loc=self.interv_mean,
                                     scale=jnp.sqrt(self.interv_noise))
        else:
            return jnp.zeros(shape=data.shape)

    def slogdet_jax(self, m, parents, n_parents):
        """
        jax.jit-compatible log determinant of a submatrix

        Done by masking everything but the submatrix and
        adding a diagonal of ones everywhere else for the 
        valid determinant

        Args:
            m: [d, d] matrix
            parents: [d, ] boolean indicator of parents
            n_parents: number of parents total

        Returns:
            natural log of determinant of `m`
        """

        n_vars = parents.shape[0]
        submat = leftsel(m, parents, maskval=np.nan)
        submat = leftsel(submat.T, parents, maskval=np.nan).T
        submat = jnp.where(jnp.isnan(submat), jnp.eye(n_vars), submat)
        return jnp.linalg.slogdet(submat)[1]


    def log_marginal_likelihood_given_g_single(self, j, n_parents, w, x, R=None, interv_targets=None):
        """
        Computes node specific term of BGe metric
        jit-compatible

        Args:
            j (int): node index for score
            n_parents (int): number of parents of node j
            w: adjacency matrix [d, d] 
            x: observations [N, d] 
            R: internal matrix for BGe score [d, d], only precomputed if interv_targets is None
            interv_targets: boolean mask of shape [N,d] of whether or not a node was intervened on
                    datapoints where node j was intervened are ignored in likelihood computation


        Returns:
            BGe score for node j
        """

        N, d = x.shape
        if interv_targets is not None:
            # mask data depending on interventions
            x = x * (1 - interv_targets[..., j][..., None])
            N = (1 - interv_targets[..., j]).sum()

        if R is None:
            # with interventions, R has to be computed individually for each node
            
            # compute mean for the N non-masked values
            x_bar = x.sum(axis=0, keepdims=True) / N
            x_center = x - x_bar
            # mask data depending on interventions
            x_center = x_center * (1 - interv_targets[..., j][..., None])
            s_N = x_center.T @ x_center  # [d, d]

            # Kuipers et al. (2014) state `R` wrongly in the paper, using `alpha_lambd` rather than `alpha_mu`
            # their supplementary contains the correct term
            R = self.T + s_N + ((N * self.alpha_mu) / (N + self.alpha_mu)) * \
                ((x_bar - self.mean_obs).T @ (x_bar - self.mean_obs))  # [d, d]

        log_gamma_term = (
            0.5 * (jnp.log(self.alpha_mu) - jnp.log(N + self.alpha_mu))
            + gammaln(0.5 * (N + self.alpha_lambd - d + n_parents + 1))
            - gammaln(0.5 * (self.alpha_lambd - d + n_parents + 1))
            - 0.5 * N * jnp.log(jnp.pi)
            # log det(T_JJ)^(..) / det(T_II)^(..) for default T
            + 0.5 * (self.alpha_lambd - d + 2 * n_parents + 1) *
            jnp.log(self.small_t)
        )


        isj = jnp.arange(d) == j
        parents = w[:, j] == 1
        parents_and_j = parents | isj

        # if JAX_DEBUG_NANS raises NaN error here,
        # ignore (happens due to lax.cond evaluating the second clause when n_parents == 0)
        log_term_r = lax.cond(
            n_parents == 0,
            # leaf node case
            lambda _: (
                # log det(R)^(...)
                - 0.5 * (N + self.alpha_lambd - d + 1) * jnp.log(jnp.abs(R[j, j]))
            ),
            # child case
            lambda _: (
                # log det(R_II)^(..) / det(R_JJ)^(..)
                0.5 * (N + self.alpha_lambd - d + n_parents) *
                    self.slogdet_jax(R, parents, n_parents)
                - 0.5 * (N + self.alpha_lambd - d + n_parents + 1) *
                    self.slogdet_jax(R, parents_and_j, n_parents + 1)
            ),
            operand=None,
        )

        return log_gamma_term + log_term_r
        

    @functools.partial(jit, static_argnums=(0, ))
    def eltwise_log_marginal_likelihood_given_g_single(self, *args):
        """
        Same inputs as `log_marginal_likelihood_given_g_single`,
        but batched over `j` and `n_parents` dimensions
        """
        return vmap(self.log_marginal_likelihood_given_g_single, (0, 0, None, None, None, None), 0)(*args)


    def log_marginal_likelihood_given_g(self, *, w, data, interv_targets=None):
        """Computes BGe marignal likelihood  log p(x | G) in closed form 

        Args:	
            data: observations [N, d]	
            w: adjacency matrix [d, d]	
            interv_targets: boolean mask of shape [N,d] of whether or not a node was intervened on
                    intervened nodes are ignored in likelihood computation

        Returns:
            [1, ] BGe Score
        """
        
        N, d = data.shape        

        # intervention
        if interv_targets is None:
            # pre-compute matrices
            small_t = (self.alpha_mu * (self.alpha_lambd - d - 1)) / \
                (self.alpha_mu + 1)
            T = small_t * jnp.eye(d)

            x_bar = data.mean(axis=0, keepdims=True)
            x_center = data - x_bar
            s_N = x_center.T @ x_center  # [d, d]

            # Kuipers (2014) states R wrongly in the paper, using alpha_lambd rather than alpha_mu;
            # the supplementary contains the correct term
            R = T + s_N + ((N * self.alpha_mu) / (N + self.alpha_mu)) * \
                ((x_bar - self.mean_obs).T @ (x_bar - self.mean_obs))  # [d, d]
        else:
            # with interventions, R has to be computed individually for each node
            R = None

        # compute number of parents for each node
        n_parents_all = w.sum(axis=0)

        # sum scores for all nodes
        scores = vmap(self.log_marginal_likelihood_given_g_single,
                      (0, 0, None, None, None, None),
                      0)(jnp.arange(d), n_parents_all, w, data, R, interv_targets)
        return jnp.sum(scores)

                
"""
everything above here is from the repo
"""

class NewBGe:
    def __init__(self, *,
                 n_vars,
                 mean_obs=None,
                 alpha_mu=None,
                 alpha_lambd=None,
                 interv_mean=None,
                 interv_noise=None,
                 ):
        super().__init__()

        self.n_vars = n_vars

        self.mean_obs = mean_obs
        self.alpha_mu = alpha_mu or 1.0
        self.alpha_lambd = alpha_lambd or (self.n_vars + 2)

        assert self.alpha_lambd > self.n_vars + 1

        self.interv_mean = interv_mean if interv_mean is not None else 0.
        self.mean_obs_interv = jnp.array([self.interv_mean] * n_vars) if type(
            self.interv_mean == float) else self.interv_mean
        self.interv_noise = interv_noise
        self.no_interv_targets = jnp.zeros(self.n_vars).astype(bool)        
        # pre-compute matrices
        self.small_t = (self.alpha_mu * (self.alpha_lambd - self.n_vars - 1)) / (self.alpha_mu + 1)
        self.T = self.small_t * np.eye(self.n_vars)

    def _log_interv_lik(self, data):
        if self.interv_noise != 0:
            return jax_normal.logpdf(x=data,
                                     loc=self.interv_mean,
                                     scale=jnp.sqrt(self.interv_noise))
        else:
            return jnp.zeros(shape=data.shape)

    def get_theta_shape(self, *, n_vars):
        raise NotImplementedError("Not available for BGe score; use `LinearGaussian` model instead.")

    def sample_parameters(self, *, key, n_vars, n_particles=0, batch_size=0):
        raise NotImplementedError("Not available for BGe score; use `LinearGaussian` model instead.")

    def sample_obs(self, *, key, n_samples, g, theta, toporder=None, interv=None):
        raise NotImplementedError("Not available for BGe score; use `LinearGaussian` model instead.")

    """
    The following functions need to be functionally pure and jax.jit-compilable
    """

    def _slogdet_jax(self, m, parents):
        n_vars = parents.shape[0]
        mask = jnp.einsum('...i,...j->...ij', parents, parents)
        submat = mask * m + (1 - mask) * jnp.eye(n_vars)
        return  jnp.linalg.slogdet(submat)[1]

    def _log_marginal_likelihood_single(self, j, n_parents, g, x, R=None, interv_targets=None):
        """
        Computes node specific term of BGe metric
        jit-compatible

        Args:
            j (int): node index for score
            n_parents (int): number of parents of node j
            g: adjacency matrix [d, d] 
            x: observations [N, d] 
            R: internal matrix for BGe score [d, d], only precomputed if interv_targets is None	
            interv_targets: boolean mask of shape [N,d] of whether or not a node was intervened on
                    datapoints where node j was intervened are ignored in likelihood computation


        Returns:
            BGe score for node j
        """
        N, d = x.shape
        if interv_targets is not None:
            # mask data depending on interventions
            x = x * (1 - interv_targets[..., j][..., None])
            N = (1 - interv_targets[..., j]).sum()

        if R is None:
            # with interventions, R has to be computed individually for each node
            mean_obs = self.mean_obs
            T = self.T

            # for interv. BGe prior, take the prior mean for interventions

            # compute mean for the N non-masked values
            x_bar = x.sum(axis=0, keepdims=True) / (N + 1e-8)
            x_center = x - x_bar
            # mask data depending on interventions
            x_center = x_center * (1 - interv_targets[..., j][..., None])
            s_N = x_center.T @ x_center  # [d, d]

            # Kuipers et al. (2014) state `R` wrongly in the paper, using `alpha_lambd` rather than `alpha_mu`
            # their supplementary contains the correct term
            R = T + s_N + ((N * self.alpha_mu) / (N + self.alpha_mu)) * \
                ((x_bar - mean_obs).T @ (x_bar - mean_obs))  # [d, d]


        parents = g[:, j]
        parents_and_j = (g + jnp.eye(d))[:, j]

        log_gamma_term = (
            0.5 * (jnp.log(self.alpha_mu) - jnp.log(N + self.alpha_mu))
            + gammaln(0.5 * (N + self.alpha_lambd - d + n_parents + 1))
            - gammaln(0.5 * (self.alpha_lambd - d + n_parents + 1))
            - 0.5 * N * jnp.log(jnp.pi)
            # log det(T_JJ)^(..) / det(T_II)^(..) for default T
            + 0.5 * (self.alpha_lambd - d + 2 * n_parents + 1) *
            jnp.log(self.small_t)
        )

        log_term_r = (
            # log det(R_II)^(..) / det(R_JJ)^(..)
            0.5 * (N + self.alpha_lambd - d + n_parents) *
            self._slogdet_jax(R, parents)
            - 0.5 * (N + self.alpha_lambd - d + n_parents + 1) *
            self._slogdet_jax(R, parents_and_j)
        )

        return log_gamma_term + log_term_r


    def _log_marginal_likelihood_single_interv(self, j, x, interv_targets):
        """
        Computes node specific term of BGe metric
        with empty graph for interventional prior
        jit-compatible

        Args:
            j (int): node index for score
            n_parents (int): number of parents of node j
            g: adjacency matrix [d, d] 
            x: observations [N, d] 
            R: internal matrix for BGe score [d, d], only precomputed if interv_targets is None	
            interv_targets: boolean mask of shape [N,d] of whether or not a node was intervened on
                    datapoints where node j was intervened are ignored in likelihood computation


        Returns:
            BGe score for node j
        """
        N, d = x.shape
        # mask data depending on interventions, keeping where interv. happen
        x = x * (interv_targets[..., j][..., None])
        N = (interv_targets[..., j]).sum()

        # with interventions, R has to be computed individually for each node
        T = self.T

        # for interv. BGe prior, take the prior mean for interventions
        mean_obs = self.mean_obs_interv

        # compute mean for the N non-masked values
        x_bar = x.sum(axis=0, keepdims=True) / (N + 1e-8)
        x_center = x - x_bar
        # mask data depending on interventions
        x_center = x_center * (interv_targets[..., j][..., None])
        s_N = x_center.T @ x_center  # [d, d]

        # Kuipers et al. (2014) state `R` wrongly in the paper, using `alpha_lambd` rather than `alpha_mu`
        # their supplementary contains the correct term
        R = T + s_N + ((N * self.alpha_mu) / (N + self.alpha_mu)) * \
            ((x_bar - mean_obs).T @ (x_bar - mean_obs))  # [d, d]

        log_gamma_term = (
            0.5 * (jnp.log(self.alpha_mu) - jnp.log(N + self.alpha_mu))
            + gammaln(0.5 * (N + self.alpha_lambd - d + 1))
            - gammaln(0.5 * (self.alpha_lambd - d  + 1))
            - 0.5 * N * jnp.log(jnp.pi)
            # log det(T_JJ)^(..) / det(T_II)^(..) for default T
            + 0.5 * (self.alpha_lambd - d + 1) *
            jnp.log(self.small_t)
        )

        log_term_r = (
            # log det(R_II)^(..) / det(R_JJ)^(..)
            0.5 * (N + self.alpha_lambd - d) *
            1 # == self._slogdet_jax(R, parents)
            - 0.5 * (N + self.alpha_lambd - d + 1) *
            jnp.log(R[j,j]) # == self._slogdet_jax(R, parents_and_j)
        )

        return log_gamma_term + log_term_r

    def log_marginal_likelihood_given_g(self,
                                        *,
                                        w,
                                        data,
                                        interv_targets=None,
                                        envs=None):
        """ Computes BGe marignal likelihood  log p(x | G) in closed form 

        Args:	
            data: observations [N, d]	
            w: adjacency matrix [d, d]	
            interv_targets: boolean mask of shape [n_env,d] of whether or not a node was intervened on
                    intervened nodes are ignored in likelihood computation
            envs: [N,] indicator for which environment

        Returns:
            [1, ] BGe Score
        """
        g = w
        N, d = data.shape

        # intervention
        if interv_targets is None or envs is None:
            x_bar = data.mean(axis=0, keepdims=True)
            x_center = data - x_bar
            s_N = x_center.T @ x_center  # [d, d]

            # Kuipers et al. (2014) state `R` wrongly in the paper, using `alpha_lambd` rather than `alpha_mu`
            # their supplementary contains the correct term
            R = self.T + s_N + ((N * self.alpha_mu) / (N + self.alpha_mu)) * \
                ((x_bar - self.mean_obs).T @ (x_bar - self.mean_obs))  # [d, d]
        else:
            # with interventions, R has to be computed individually for each node
            R = None
            # extend with no targets for observational case
            interv_targets = jnp.concatenate(
                (jnp.zeros(shape=(1, d)), interv_targets), axis=0)
            interv_targets = interv_targets[envs]

        # compute number of parents for each node
        n_parents_all = g.sum(axis=0)

        # sum scores for all nodes
        scores = vmap(self._log_marginal_likelihood_single,
                      (0, 0, None, None, None, None),
                      0)(jnp.arange(d), n_parents_all, g, data, R,
                         interv_targets)
        return jnp.sum(scores)

    def log_score_interventions(self, data, interv_targets, envs):
        # interventional BGe score with empty graph
        N, d = data.shape
        interv_targets = jnp.concatenate(
            (jnp.zeros(shape=(1, d)), interv_targets), axis=0)
        interv_targets = interv_targets[envs]

        scores_interv = vmap(self._log_marginal_likelihood_single_interv,
                             (0, None, None), 0)(jnp.arange(d), data,
                                                 interv_targets)
        return jnp.sum(scores_interv)

    def log_marginal_likelihood(self, w, data):
        return self.log_marginal_likelihood_interv(
            w=w, data=data, interv_targets=self.no_interv_targets)

