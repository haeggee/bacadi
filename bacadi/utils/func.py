import os
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision

import functools

import numpy as onp # needed for np.unique(axis=0)

import jax.numpy as jnp
from jax import jit
from jax.scipy.special import logsumexp
from jax.tree_util import tree_flatten, tree_map, tree_multimap, tree_reduce


def expand_by(arr, n):
    """
    Expands jnp.array by n dimensions at the end
    Args:
        arr: shape [...]
        n (int)
    
    Returns:
        arr of shape [..., 1, ..., 1] with `n` ones
    """
    return jnp.expand_dims(arr, axis=tuple(arr.ndim + j for j in range(n)))


@jit
def sel(mat, mask):
    """
    jit/vmap helper function

    Args:
        mat: [N, d]
        mask: [d, ]  boolean 

    Returns:
        [N, d] with columns of `mat` with `mask` == 1 non-zero a
        and the columns with `mask` == 0 are zero

    Example: 
        mat 
        1 2 3
        4 5 6
        7 8 9

        mask
        1 0 1

        out
        1 0 3
        4 0 6
        7 0 9
    """
    return jnp.where(mask, mat, 0)

@jit
def leftsel(mat, mask, maskval=0.0):
    """
    jit/vmap helper function

    Args:
        mat: [N, d]
        mask: [d, ]  boolean 

    Returns:
        [N, d] [N, d] with columns of `mat` with `mask` == 1 non-zero a
        and pushed leftmost; the columns with `mask` == 0 are zero

    Example: 
        mat 
        1 2 3
        4 5 6
        7 8 9

        mask
        1 0 1

        out
        1 3 0
        4 6 0
        7 9 0
    """
    valid_indices = jnp.where(
        mask, jnp.arange(mask.shape[0]), mask.shape[0])
    padded_mat = jnp.concatenate(
        [mat, maskval * jnp.ones((mat.shape[0], 1))], axis=1)
    padded_valid_mat = padded_mat[:, jnp.sort(valid_indices)]
    return padded_valid_mat

@functools.partial(jit, static_argnums=(1,))
def mask_topk(x, topkk):
    """
    Returns indices of `topk` entries of `x` in decreasing order

    Args:
        x: [N, ]
        topk (int)

    Returns:
        array of shape [topk, ]
        
    """
    mask = x.argsort()[-topkk:][::-1]
    return mask


@jit
def bit2id(b):
    """
    Converts a batch of binary (adjacency) matrices into 
    low-memory bit representation to facilitate handling of
    larger problems.
    See `id2bit` for counterpart.

    Args:
        b: batch of matrices of shape [N, ...] with {0,1} values

    Returns:
        array of shape [N, ] with integer bit representation
    """
    N = b.shape[0]
    b_flat = b.reshape(N, -1)
    return jnp.packbits(b_flat, axis=1, bitorder='little')


@functools.partial(jit, static_argnums=(1,))
def id2bit(id, d):
    """
    Converts a batch of bit representations into 
    their corresponding binary (adjacency matrices).
    See `bit2id` for counterpart.

    low-memory bit representation to facilitate handling of
    larger problems

    Args:
        id: [N, ?] with integer bit representation and number of vars rep. by of matrix
        d (int): number of variables

    Returns:
        batch of matrices of shape [N, d, d] with {0,1} values
    """
    N, _ = id.shape
    b_flat = jnp.unpackbits(id, axis=1, bitorder='little')
    b_flat = b_flat[:, :d * d]
    return b_flat.reshape(N, d, d)

def dist_is_none(dist):
    """
    Checks whether distribution tuple `dist` is None
    """
    return (dist[0] is None) or (dist[1] is None)


def joint_dist_to_marginal(dist):
    """
    Drops all but the particles_z and log_probs
    of the joint distribution tuple
    """
    return (dist[0], dist[-1]) 


def squared_norm_pytree(x, y):
    """Computes squared euclidean norm between two pytrees
    
    Args:
        x:  PyTree 
        y:  PyTree 

    Returns:
        shape [] 
    """ 

    diff = tree_multimap(jnp.subtract, x, y)
    squared_norm_ind = tree_map(lambda leaf: jnp.square(leaf).sum(), diff)
    squared_norm = tree_reduce(jnp.add, squared_norm_ind)
    return squared_norm


def pairwise_structural_hamming_distance(*, x, y, axis=None, atol=1e-8):
    """Simpler implementation taken from cdt.SHD
    (CausalDiscoveryToolbox https://github.com/FenTechSolutions/CausalDiscoveryToolbox)

    Computes pairwise Structural Hamming distance, i.e.
    the number of edge insertions, deletions or flips in order to transform one graph to another
        - this means, edge reversals do not double count
        - this means, getting an undirected edge wrong only counts 1

    Args:
        x:  [N, ...]
        y:  [M, ...]

    Returns:
        [N, M] where elt i,j is  SHD(x[i], y[j]) = sum(x[i] != y[j])
    """

    # all but first axis is usually used for the norm, assuming that first dim is batch dim
    assert(x.ndim == 3 and y.ndim == 3)

    # via computing pairwise differences
    pw_diff = jnp.abs(jnp.expand_dims(x, axis=1) - jnp.expand_dims(y, axis=0))
    pw_diff = pw_diff + pw_diff.transpose((0, 1, 3, 2))

    # ignore double edges
    pw_diff = jnp.where(pw_diff > 1, 1, pw_diff)
    shd = jnp.sum(pw_diff, axis=(2, 3)) / 2 

    return shd

def expected_graph(dist, n_vars):
    particles, log_weights = dist[0], dist[-1]
    particles = id2bit(particles, n_vars)

    # P(G_ij = 1) = sum_G w_G 1[G = G] in log space
    log_p, log_p_sign = logsumexp(
        log_weights[..., jnp.newaxis, jnp.newaxis], 
        b=particles.astype(log_weights.dtype), 
        axis=0, return_sign=True)
    
    p_edge = log_p_sign * jnp.exp(log_p)
    return p_edge

def expected_interv(dist):
    particles, log_weights = dist[-2], dist[-1]

    # P(I_ij = 1) = sum_I w_I 1[I = I] in log space
    log_p, log_p_sign = logsumexp(
        log_weights[..., jnp.newaxis, jnp.newaxis], 
        b=particles.astype(log_weights.dtype), 
        axis=0, return_sign=True)
    
    p_edge = log_p_sign * jnp.exp(log_p)
    return p_edge