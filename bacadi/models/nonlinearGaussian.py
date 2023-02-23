import jax.numpy as jnp
from jax import vmap
from jax import random
from jax.ops import index, index_update
from jax.scipy.stats import norm as jax_normal
from jax.tree_util import tree_map, tree_reduce
from jax.scipy.special import logsumexp
from jax.scipy.stats import gamma as jax_gamma

import jax.experimental.stax as stax
from jax.experimental.stax import Dense, Sigmoid, LeakyRelu, Relu, Tanh, Gelu, Selu

from jax.nn.initializers import normal, xavier_normal, he_normal

from bacadi.utils.graph import graph_to_mat
from bacadi.utils.tree import tree_shapes


def DenseNoBias(out_dim, W_init=normal()):
    """Layer constructor function for a dense (fully-connected) layer _without_ bias"""

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        W = W_init(rng, (input_shape[-1], out_dim))
        return output_shape, (W, )

    def apply_fun(params, inputs, **kwargs):
        W, = params
        return jnp.dot(inputs, W)

    return init_fun, apply_fun


def makeDenseNet(*,
                 hidden_layers,
                 weight_init,
                 sig_param,
                 bias=True,
                 activation='relu'):
    """
    Generates functions defining a fully-connected NN
    with Gaussian initialized parameters

    Args:
        hidden_layers (list): list of ints specifying the dimensions of the hidden sizes
        sig_weight: std dev of weight initialization
        sig_bias: std dev of weight initialization
    
    Returns:
        stax.serial neural net object
    """

    if weight_init == 'xavier_normal':
        init_fun = xavier_normal()
    elif weight_init == 'he_normal':
        init_fun = he_normal()
    elif weight_init == 'normal':
        init_fun = normal(sig_param)
    elif weight_init == 'normal_shifted':
        def init_fun(key, shape, dtype=jnp.float_):
            theta = random.normal(key, shape) * sig_param
            return theta + 0.5 * jnp.sign(theta)
    elif weight_init == 'uniform_shifted':
        # uniform in [-sig_param, sig_param]
        def init_fun(key, shape, dtype=jnp.float_):
            key, subk = random.split(key)
            theta = random.uniform(subk,
                                   shape=shape) * (2 * sig_param - 0.5) + 0.5
            key, subk = random.split(key)
            theta = theta * random.choice(
                subk, a=jnp.array([-1, 1]), shape=shape)
            return theta
    else:
        raise KeyError(f'Unknown parameter init function `{weight_init}`')

    # bias is just always normal for now
    bias_init_fun = normal(sig_param)

    # features: [hidden_layers[0], hidden_layers[0], ..., hidden_layers[-1], 1]
    if activation == 'sigmoid':
        f_activation = Sigmoid
    elif activation == 'tanh':
        f_activation = Tanh
    elif activation == 'relu':
        f_activation = Relu
    elif activation == 'leakyrelu':
        f_activation = LeakyRelu
    elif activation == 'gelu':
        f_activation = Gelu
    elif activation == 'selu':
        f_activation = Selu
    else:
        raise KeyError(f'Invalid activation function `{activation}`')

    modules = []
    if bias:
        for dim in hidden_layers:
            modules += [
                Dense(dim, W_init=init_fun, b_init=bias_init_fun), f_activation
            ]
        modules += [Dense(1, W_init=init_fun, b_init=bias_init_fun)]
    else:
        for dim in hidden_layers:
            modules += [DenseNoBias(dim, W_init=init_fun), f_activation]
        modules += [DenseNoBias(1, W_init=init_fun)]

    return stax.serial(*modules)
    

class DenseNonlinearGaussianJAX:
    """	
    Non-linear Gaussian BN with interactions modeled by a fully-connected neural net
    See: https://arxiv.org/abs/1909.13189    
    """
    def __init__(self,
                 *,
                 obs_noise,
                 sig_param,
                 hidden_layers,
                 init_sig_param=1.0,
                 init='normal_shifted',
                 interv_prior_mean=0.0,
                 interv_prior_std=10.0,
                 interv_mean=None,
                 interv_noise=None,
                 verbose=False,
                 activation='relu',
                 bias=True):
        super().__init__()

        self.obs_noise = obs_noise
        self.sig_param = sig_param
        self.init_sig_param = init_sig_param
        self.hidden_layers = hidden_layers
        self.verbose = verbose
        self.activation = activation
        self.bias = bias

        self.interv_prior_mean = interv_prior_mean
        self.interv_prior_std = interv_prior_std
        self.interv_mean = interv_mean or 0.
        self.interv_noise = interv_noise or 1.0

        # init single neural net function for one variable with jax stax
        self.nn_init_random_params, self._nn_forward = makeDenseNet(
            hidden_layers=self.hidden_layers, 
            weight_init=init,
            sig_param=init_sig_param,
            activation=self.activation,
            bias=self.bias)
        
        # vectorize init and forward functions
        self.eltwise_nn_init_random_params = vmap(self.nn_init_random_params, (0, None), 0)
        self.double_eltwise_nn_init_random_params = vmap(self.eltwise_nn_init_random_params, (0, None), 0)
        self.triple_eltwise_nn_init_random_params = vmap(self.double_eltwise_nn_init_random_params, (0, None), 0)
        
        # [d2, ?], [N, d] -> [N, d2]
        self.eltwise_nn_forward = vmap(self.nn_forward, (0, None), 1)

        # [d2, ?], [d2, N, d] -> [N, d2]
        self.double_eltwise_nn_forward = vmap(self.nn_forward, (0, 0), 1)

    # [?], [N, d] -> [N,]
    def nn_forward(self, theta, x):
        return self._nn_forward(theta, x).squeeze(-1)

    def leaf_log_prob(self, leaf_theta):
        return jax_normal.logpdf(x=leaf_theta, loc=0, scale=self.sig_param)

    def _log_interv_lik(self, data):
        if self.interv_noise is not None and self.interv_noise != 0:
            return jax_normal.logpdf(x=data,
                                     loc=self.interv_mean,
                                     scale=jnp.sqrt(self.interv_noise))
        else:
            return jnp.zeros(shape=data.shape)

    def get_theta_shape(self, *, n_vars):
        """ Returns tree shape of the parameters of the neural networks
        Args:
            n_vars

        Returns:
            PyTree of parameter shape
        """
        
        dummy_subkeys = jnp.zeros((n_vars, 2), dtype=jnp.uint32)
        _, theta = self.eltwise_nn_init_random_params(dummy_subkeys, (n_vars, )) # second arg is `input_shape` of NN forward pass

        theta_shape = tree_shapes(theta)
        return theta_shape

    def init_parameters(self, *, key, n_vars, n_particles, batch_size=0):
        """Samples batch of random parameters given dimensions of graph, from p(theta | G) 
        Args:
            key: rng
            n_vars: number of variables in BN
            n_particles: number of parameter particles sampled
            batch_size: number of batches of particles being sampled

        Returns:
            theta : PyTree with leading dimension of `n_particles`
        """

        key, subk = random.split(key)
        if batch_size != 0:
            raise ValueError("batch_size for nonlingauss not implemented")

        subkeys = random.split(subk, n_particles * n_vars).reshape(
            n_particles, n_vars, -1)
        _, theta = self.double_eltwise_nn_init_random_params(
            subkeys, (n_vars, ))
        # to float64
        theta = tree_map(lambda arr: arr.astype(jnp.float64), theta)

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
            theta : pytree with [...theta_g, theta_sig, theta_interv_mean, theta_interv_sig]
                            where theta_g is NN parameters; theta_sig is [d,]; 
                            theta_interv_mean is [n_env, d]; theta_interv_sig is [n_env, d]
            
        """
        key, subk = random.split(key)

        if batch_size != 0:
            raise ValueError("batch_size for nonlingauss not implemented")

        subkeys = random.split(subk, n_particles * n_vars).reshape(
            n_particles, n_vars, -1)
        _, theta_g = self.double_eltwise_nn_init_random_params(
            subkeys, (n_vars, ))
        # to float64
        theta_g = tree_map(lambda arr: arr.astype(jnp.float64), theta_g)

        # mean of interventions
        key, subk = random.split(key)
        theta_interv_mean = jnp.sqrt(0.1) * random.normal(
            subk, shape=(n_particles, n_env - 1, n_vars)) + self.interv_prior_mean

        theta_g.append(theta_interv_mean)
        return theta_g

    def sample_parameters(self, *, key, g):
        """Samples parameters for neural network. Here, g is ignored.
        Args:
            g (igraph.Graph): graph
            key: rng

        Returns:
            theta : list of (W, b) tuples, dependent on `hidden_layers`
        """
        n_vars = len(g.vs)

        subkeys = random.split(key, n_vars)
        _, theta = self.eltwise_nn_init_random_params(subkeys, (n_vars, ))

        return theta


    def sample_obs(self, *, key, n_samples, g, theta, toporder=None, interv={}):
        """
        Samples `n_samples` observations by doing single forward passes in topological order
        Args:
            key: rng
            n_samples (int): number of samples
            g (igraph.Graph): graph
            theta : PyTree of parameters
            interv: {intervened node : clamp value} or {intervened node : {'mean': val, 'noise': val}}

        Returns:
            x : [n_samples, d] 
        """

        # find topological order for ancestral sampling
        if toporder is None:
            toporder = g.topological_sorting()

        n_vars = len(g.vs)
        x = jnp.zeros((n_samples, n_vars))

        key, subk = random.split(key)
        z = jnp.sqrt(self.obs_noise) * random.normal(subk, shape=(n_samples, n_vars))

        g_mat = graph_to_mat(g)

        # ancestral sampling
        # for simplicity, does d full forward passes for simplicity, which avoids indexing into python list of parameters
        for j in toporder:

            # intervention
            if j in interv.keys():
                if type(interv[j]) is dict:
                    key, subk = random.split(key)
                    z_ = jnp.sqrt(interv[j]['noise']) * random.normal(
                        subk, shape=(n_samples,))
                    x = index_update(x, index[:, j], interv[j]['mean'] + z_)
                else:
                    x = index_update(x, index[:, j], interv[j])    
                continue

            # regular ancestral sampling
            parents = g_mat[:, j].reshape(1, -1)

            has_parents = parents.sum() > 0

            if has_parents:
                # [N, d] = [N, d] * [1, d] mask non-parent entries of j
                x_msk = x * parents

                # [N, d] full forward pass
                means = self.eltwise_nn_forward(theta, x_msk)

                # [N,] update j only
                x = index_update(x, index[:, j], means[:, j] + z[:, j])
            else:
                x = index_update(x, index[:, j], z[:, j])

        return x


    def log_prob_parameters(self, *, theta, w):
        """log p(theta | g)
        Assumes N(mean_edge, sig_edge^2) distribution for any given edge 

        Args:
            theta: parmeter PyTree
            w: adjacency matrix of graph [n_vars, n_vars]

        Returns:
            logprob [1,]
        """

        # compute log prob for each weight
        logprobs = tree_map(lambda leaf_theta: self.leaf_log_prob(leaf_theta),
                            theta)

        # mask logprobs of first layer weight matrix [0][0] according to graph
        # [d, d, dim_first_layer] = [d, d, dim_first_layer] * [d, d, 1]
        if self.bias:
            first_weight_logprobs, first_bias_logprobs = logprobs[0]
            logprobs[0] = (first_weight_logprobs * w.T[:, :, None],
                           first_bias_logprobs)
        else:
            first_weight_logprobs, = logprobs[0]
            logprobs[0] = (first_weight_logprobs * w.T[:, :, None], )
        # sum logprobs of every parameter tensor and add all up
        score_weights = tree_reduce(jnp.add, tree_map(jnp.sum, logprobs))

        return score_weights
    

    def log_prob_interv_parameters(self, *, theta, w, I):
        """log p(theta_I | I)
        Assumes N(mean_val, sig_edge^2) distribution for any given edge 
        
        
        Args:
            theta:  pytree with [...theta_g, theta_sig, theta_interv_mean, theta_interv_sig]
                            where theta_g is NN pytree; theta_sig is [d,]; 
                            theta_interv_mean is [n_env, d]; theta_interv_sig is [n_env, d]
            I:      [n_env-1, n_vars]
            
        Returns:
            [1, ]
        """
        
        theta_g = theta[:-1]
        theta_interv_mean = theta[-1]
        
        # compute log prob for each weight
        logprobs = tree_map(lambda leaf_theta: self.leaf_log_prob(leaf_theta),
                            theta_g)

        # mask logprobs of first layer weight matrix [0][0] according to graph
        # [d, d, dim_first_layer] = [d, d, dim_first_layer] * [d, d, 1]
        if self.bias:
            first_weight_logprobs, first_bias_logprobs = logprobs[0]
            logprobs[0] = (first_weight_logprobs * w.T[:, :, None],
                           first_bias_logprobs)
        else:
            first_weight_logprobs, = logprobs[0]
            logprobs[0] = (first_weight_logprobs * w.T[:, :, None], )
        # sum logprobs of every parameter tensor and add all up
        score_weights = tree_reduce(jnp.add, tree_map(jnp.sum, logprobs))

        score_interv_mean = jnp.sum(
            I * jax_normal.logpdf(x=theta_interv_mean,
                                  loc=self.interv_prior_mean,
                                  scale=self.interv_prior_std))

        return score_weights + score_interv_mean


    def log_likelihood(self, *, data, theta, w, interv_targets, envs):
        """log p(x | theta, G)
        Assumes N(mean_obs, obs_noise^2) distribution for any given observation
        
        Args:
            data: observations [N, d]
            theta: parameter PyTree
            w: adjacency matrix [d, d]
            interv_targets: boolean indicator of intervention locations [N, d]
        
        Returns:
            logprob [d,]
        """
        n_vars = data.shape[-1]
        if interv_targets is None or envs is None:
            interv_targets = jnp.zeros(n_vars)
        else:
            # extend with no targets for observ. env
            interv_targets = jnp.concatenate(
                (jnp.zeros(shape=(1, n_vars)), interv_targets), axis=0)
            interv_targets = interv_targets[envs]

        # [d2, N, d] = [1, N, d] * [d2, 1, d] mask non-parent entries of each j
        all_x_msk = data[None] * w.T[:, None]

        # [N, d2] NN forward passes for parameters of each param j
        all_means = self.double_eltwise_nn_forward(theta, all_x_msk)

        # sum scores for all nodes and data
        return jnp.sum(
            jnp.where(
                # [1, n_vars]
                interv_targets,
                # self.log_interv_lik(data),
                0.0,
                # [n_observations, n_vars]
                jax_normal.logpdf(x=data,
                                  loc=all_means,
                                  scale=jnp.sqrt(self.obs_noise))))

    def log_likelihood_soft_interv_targets(
        self,
        *,
        data,
        theta,
        w,
        interv_targets=None,
        envs=None,
    ):
        """log p(x | theta, G, I)
        Assumes N(mean_obs, obs_noise^2) distribution for any given observation
        
        Args:
            data: observations  [N, d]
            theta:              parameter PyTree
            w: adjacency matrix [d, d]
            interv_targets:     [N, d] soft indicator of intervention locations 
            env:                [N,] indicator which env observ. i is from
        
        Returns:
            logprob [d,]
        """
        n_vars = w.shape[0]

        if envs is None:
            envs = jnp.zeros(data.shape[0], dtype=int)
        if interv_targets is None:
            interv_targets = jnp.zeros(data.shape)
        theta_g = theta[:-1]
        theta_interv_mean = theta[-1]

        # [d2, N, d] = [1, N, d] * [d2, 1, d] mask non-parent entries of each j
        all_x_msk = data[None] * w.T[:, None]

        # [N, d2] NN forward passes for parameters of each param j
        all_means = self.double_eltwise_nn_forward(theta_g, all_x_msk)

        # log_interv_lik = self.log_interv_lik(data)
        log_nointerv_lik = jax_normal.logpdf(x=data,
                                             loc=all_means,
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


    def log_likelihood_with_noise(self, *, data, theta, noise, w, interv_targets, envs):
        """log p(x | theta, G)
        Assumes N(mean_obs, obs_noise^2) distribution for any given observation
        
        Args:
            data: observations [N, d]
            theta: parameter PyTree
            w: adjacency matrix [d, d]
            interv_targets: boolean indicator of intervention locations [N, d]
        
        Returns:
            logprob [d,]
        """
        n_vars = data.shape[-1]
        if interv_targets is None or envs is None:
            interv_targets = jnp.zeros(n_vars)
        else:
            # extend with no targets for observ. env
            interv_targets = jnp.concatenate(
                (jnp.zeros(shape=(1, n_vars)), interv_targets), axis=0)
            interv_targets = interv_targets[envs]

        # [d2, N, d] = [1, N, d] * [d2, 1, d] mask non-parent entries of each j
        all_x_msk = data[None] * w.T[:, None]

        # [N, d2] NN forward passes for parameters of each param j
        all_means = self.double_eltwise_nn_forward(theta, all_x_msk)

        # sum scores for all nodes and data
        return jnp.sum(
            jnp.where(
                # [1, n_vars]
                interv_targets,
                # self.log_interv_lik(data),
                0.0,
                # [n_observations, n_vars]
                jax_normal.logpdf(x=data,
                                  loc=all_means,
                                  scale=noise)))