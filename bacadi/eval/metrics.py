from asyncio import FastChildWatcher
import warnings
from cdt.metrics import get_CPDAG
import causaldag
from jax import vmap
warnings.filterwarnings("ignore", message="No GPU automatically detected")

import jax.numpy as jnp
from jax.scipy.special import logsumexp

import numpy as onp
import cdt

from bacadi.utils.func import bit2id, id2bit, pairwise_structural_hamming_distance
from bacadi.utils.tree import tree_mul, tree_select
from bacadi.utils.graph import elwise_acyclic_constr_nograd


from sklearn import metrics as sklearn_metrics

#
# marginal posterior p(G | D) metrics
#

def l1_edge_belief(*, dist, g):
    """
    L1 edge belief error as defined by Murphy, 2001.

    Args:
        dist: log distribution tuple as e.g. given by `particle_marginal_empirical`
        g: ground truth graph [d, d] 

    Returns:
        [1, ]
    """
    n_vars = g.shape[0]

    # convert graph ids to adjacency matrices
    id_particles_cyc, log_weights_cyc = dist 
    particles_cyc = id2bit(id_particles_cyc, n_vars)

    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(particles_cyc, n_vars) == 0
    if is_dag.sum() == 0:
        # score as "wrong on every edge"
        return n_vars * (n_vars - 1) / 2

    particles = particles_cyc[is_dag, :, :]
    log_weights = log_weights_cyc[is_dag] - logsumexp(log_weights_cyc[is_dag])

    # P(G_ij = 1) = sum_G w_G 1[G = G] in log space
    log_edge_belief, log_edge_belief_sgn = logsumexp(
        log_weights[..., jnp.newaxis, jnp.newaxis], 
        b=particles.astype(log_weights.dtype), 
        axis=0, return_sign=True)

    # L1 edge error
    p_edge = log_edge_belief_sgn * jnp.exp(log_edge_belief)
    p_no_edge = 1 - p_edge
    err_connected = jnp.sum(g * p_no_edge)
    err_notconnected = jnp.sum(
        jnp.triu((1 - g) * (1 - g).T * (1 - p_no_edge * p_no_edge.T), k=0))

    err = err_connected + err_notconnected
    return err


def expected_shd(*, dist, g, use_cpdag=False, interv_targets=None):
    """
    Expected structural hamming distance. Note that we drop cyclic graphs!
    Defined as 
        expected_shd = sum_G p(G | D)  SHD(G, G*)

    Args:
        dist: log distribution tuple
        g: ground truth graph [d, d]

    Returns: 
        [1, ]
    """
    n_vars = g.shape[0]

    # convert graph ids to adjacency matrices
    id_particles, log_weights = dist
    particles = id2bit(id_particles, n_vars)
    if use_cpdag:
        # DAG particles should have used intervention information already,
        # i.e. the particles are CPDAG of I-MEC
        # use I-MEC of GT
        if interv_targets is not None:
            gt = causaldag.classes.dag.DAG.from_amat(g)
            g = gt.interventional_cpdag(interv_targets,
                                        cpdag=gt.cpdag()).to_amat()[0]
        # use MEC of GT
        else:
            g = onp.array(get_CPDAG(onp.array(g))).astype(g.dtype)
    else:
        # select acyclic graphs
        is_dag = elwise_acyclic_constr_nograd(particles, n_vars) == 0
        if is_dag.sum() == 0:
            # score as "wrong on every edge"
            return n_vars * (n_vars - 1) / 2
        
        particles = particles[is_dag, :, :]
        log_weights = log_weights[is_dag] - logsumexp(log_weights[is_dag])

    # compute shd for each graph
    shds = pairwise_structural_hamming_distance(x=particles, y=g[None]).squeeze(1)

    # expected SHD = sum_G p(G) SHD(G)
    log_expected_shd, log_expected_shd_sgn = logsumexp(
        log_weights, b=shds.astype(log_weights.dtype), axis=0, return_sign=True)

    expected_shd = log_expected_shd_sgn * jnp.exp(log_expected_shd)
    return expected_shd


def expected_sid(*, dist, g, use_cpdag=False, interv_targets=None):
    """
    Expected structural _interventional_ distance. Note that we drop cyclic graphs!
    Defined as 
        expected_sid = sum_G p(G | D)  SID(G, G*)

    Args:
        dist: log distribution tuple
        g: ground truth graph [d, d]

    Returns: 
        [1, ]
    """
    n_vars = g.shape[0]

    # convert graph ids to adjacency matrices
    id_particles, log_weights = dist
    particles = id2bit(id_particles, n_vars)

    if use_cpdag:
        # the particles already are CPDAGs of I-MEC
        # use I-MEC of GT graph
        if interv_targets is not None:
            gt = causaldag.classes.dag.DAG.from_amat(g)
            g = gt.interventional_cpdag(interv_targets,
                                        cpdag=gt.cpdag()).to_amat()[0]
        # use MEC of GT graph
        else:
            g = onp.array(get_CPDAG(onp.array(g))).astype(g.dtype)
        sids = []
        for g_ in onp.array(particles):
            sid_ = cdt.metrics.SID_CPDAG(target=g, pred=g_)
            # mean between upper and lower bound
            sids.append((float(sid_[0]) + float(sid_[1])) / 2)
        sids = jnp.array(sids)
    else:
        # select acyclic graphsinterv_target_list
        is_dag = elwise_acyclic_constr_nograd(particles, n_vars) == 0
        if is_dag.sum() == 0:
            # score as "wrong completely"
            return n_vars * (n_vars - 1)
        
        particles = particles[is_dag, :, :]
        log_weights = log_weights[is_dag] - logsumexp(log_weights[is_dag])
        g = onp.array(g)
        # compute sid for each graph
        sids = jnp.array([
            float(cdt.metrics.SID(target=g, pred=g_))
            for g_ in onp.array(particles)
        ])

    # expected SID = sum_G p(G) SID(G)
    log_expected_sid, log_expected_sid_sgn = logsumexp(
        log_weights, b=sids.astype(log_weights.dtype), axis=0, return_sign=True)

    expected_shd = log_expected_sid_sgn * jnp.exp(log_expected_sid)
    return expected_shd


def kl_divergence(n_vars, dist_p, gt_dist_q):
    """
    Kullback-Leibler divergence between two posterior distributions over DAGs.
    !! NOTE: for now assume dist_q is the ground truth, i.e. includes all DAGs
    !! NOTE: we drop cyclic graphs, for which the KL-div would be ill-defined compared to the GT
    Defined as 
        expected_shd = sum_G [ p(G | D)  log(p(G | D) / q(G | D))

    Args:
        n_vars (int): number of variables
        dist_p: log distribution tuple
        dist_q: log distribution tuple

    Returns: 
        [1, ]
    """

    # convert graph ids to adjacency matrices
    id_particles_p, log_weights_p = dist_p
    id_particles_q, log_weights_q = gt_dist_q
    n_graphs_p = id_particles_p.shape[0]
    n_graphs_q = id_particles_q.shape[0]

    particles_p = id2bit(id_particles_p, n_vars)

    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(particles_p, n_vars) == 0

    if is_dag.sum() == 0: # ill-defined, can only return infinity
        return jnp.inf
    
    id_particles_p = id_particles_p[is_dag]
    log_weights_p = log_weights_p[is_dag] - logsumexp(log_weights_p[is_dag])
    
    # need to match graphs of q so that they have same index of that of p
    ind = jnp.argwhere(
        jnp.all(id_particles_p[:, None] == id_particles_q, axis=-1))

    if n_graphs_p > n_graphs_q:
        matched_log_p = log_weights_p[ind[:, 0]]
    else:
        matched_log_p = log_weights_p

    matched_log_q = log_weights_q[ind[:, 1]]
    p = jnp.exp(matched_log_p)

    return jnp.sum(p * (matched_log_p - matched_log_q))


def expected_edges(*, dist, g):
    """
    Expected number of edges
    Defined as 
        expected_edges = sum_G p(G | D)  #edges(G)

    Args:
        dist: log distribution tuple
        g: ground truth graph [d, d]

    Returns: 
        [1, ]
    """
    n_vars = g.shape[0]

    # convert graph ids to adjacency matrices
    id_particles_cyc, log_weights_cyc = dist
    particles_cyc = id2bit(id_particles_cyc, n_vars)

    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(particles_cyc, n_vars) == 0
    if is_dag.sum() == 0:
        # if no acyclic graphs, count the edges of the cyclic graphs; more consistent 
        n_edges_cyc = particles_cyc.sum(axis=(-1, -2))
        log_expected_edges_cyc, log_expected_edges_cyc_sgn = logsumexp(
            log_weights_cyc, b=n_edges_cyc.astype(log_weights_cyc.dtype), axis=0, return_sign=True)

        expected_edges_cyc = log_expected_edges_cyc_sgn * jnp.exp(log_expected_edges_cyc)
        return expected_edges_cyc
    
    particles = particles_cyc[is_dag, :, :]
    log_weights = log_weights_cyc[is_dag] - logsumexp(log_weights_cyc[is_dag])
    
    # count edges for each graph
    n_edges = particles.sum(axis=(-1, -2))

    # expected edges = sum_G p(G) edges(G)
    log_expected_edges, log_expected_edges_sgn = logsumexp(
        log_weights, b=n_edges.astype(log_weights.dtype), axis=0, return_sign=True)

    expected_edges = log_expected_edges_sgn * jnp.exp(log_expected_edges)
    return expected_edges


def threshold_metrics(*,
                      dist,
                      g,
                      is_graph_distr=True,
                      undirected_cpdag_oriented_correctly=False):
    """
    Various threshold metrics (e.g. AUROC) 

    Args:
        dist: log distribution tuple
        g: ground truth matrix [k, d] (e.g. graph for k == d)
        is_graph_distr (bool): if the dist describes graphs. Determines if
            we check for acyclicity, otherwise treat as standard binary classification.
        undirected_cpdag_oriented_correctly (bool): 
            if True, uses CPDAGs instead of DAGs

    Returns: 
        [1, ]
    """
    k = g.shape[0]
    target_flat = g.reshape(-1)
    if is_graph_distr:
        target_flat = g.reshape(-1)
        # convert graph ids to adjacency matrices
        n_vars = k
        id_particles, log_weights = dist
        particles = id2bit(id_particles, n_vars)

        if undirected_cpdag_oriented_correctly:
            # find undirected edges of inferred CPDAGs
            particle_cpdag_undir_edge = ((particles == 1) &
                                         (particles.transpose(
                                             (0, 2, 1)) == 1))

            # direct them according to the ground truth IF the ground truth has an edge there
            particle_cpdag_undir_edge_correct = particle_cpdag_undir_edge & (
                (g[None] == 1) | (g[None].transpose((0, 2, 1)) == 1))
            particles = jnp.where(particle_cpdag_undir_edge_correct, g,
                                  particles)

            # direct them one way IF the ground truth does not have an edge here (to only count the mistake once)
            particle_cpdag_undir_edge_incorrect = particle_cpdag_undir_edge & (
                (g[None] == 0) & (g[None].transpose((0, 2, 1)) == 0))
            particles = jnp.where(particle_cpdag_undir_edge_incorrect,
                                  jnp.triu(jnp.ones_like(g, dtype=g.dtype)),
                                  particles)
        else:
            # select acyclic graphs
            is_dag = elwise_acyclic_constr_nograd(particles, n_vars) == 0
            if is_dag.sum() == 0:
                # score as random/junk classifier
                # for AUROC: 0.5
                # for precision-recall: no. true edges/ no. possible edges
                return {
                    'roc_auc': 0.5,
                    'prc_auc': (g.sum() / (n_vars * (n_vars - 1))).item(),
                    'ave_prec': (g.sum() / (n_vars * (n_vars - 1))).item(),
                }

            particles = particles[is_dag, :, :]

            log_weights = log_weights[is_dag] - logsumexp(
                log_weights[is_dag])
    else:
        # particles represent boolean masks already (e.g. interv. targets)
        particles, log_weights = dist

    # P(G_ij = 1) = sum_G w_G 1[G = G] in log space
    log_p, log_p_sign = logsumexp(log_weights[..., jnp.newaxis, jnp.newaxis],
                                  b=particles.astype(log_weights.dtype),
                                  axis=0,
                                  return_sign=True)

    # L1 edge error
    p_edge = log_p_sign * jnp.exp(log_p)
    p_edge_flat = p_edge.reshape(-1)

    # threshold metrics
    fpr_, tpr_, _ = sklearn_metrics.roc_curve(target_flat, p_edge_flat)
    roc_auc_ = sklearn_metrics.auc(fpr_, tpr_)
    precision_, recall_, _ = sklearn_metrics.precision_recall_curve(
        target_flat, p_edge_flat)
    prc_auc_ = sklearn_metrics.auc(recall_, precision_)
    ave_prec_ = sklearn_metrics.average_precision_score(
        target_flat, p_edge_flat)

    return {
        'fpr': fpr_.tolist(),
        'tpr': tpr_.tolist(),
        'roc_auc': roc_auc_,
        'precision': precision_.tolist(),
        'recall': recall_.tolist(),
        'prc_auc': prc_auc_,
        'ave_prec': ave_prec_,
    }


def neg_ave_log_marginal_likelihood(*, dist, eltwise_log_target, x, unknown_interv=False):
    """
    Computes neg. ave log marginal likelihood.

    Args:
        dist:  log distribution tuple
        eltwise_log_target: function satisfying ([:, d, d], x) -> [:, ]
            and computing P(D | G) for held-out D
        x: [..., d]

    Returns: 
        [1, ]
    """
    n_vars = x.shape[-1]

    if len(dist) >= 3:
        id_particles_cyc = dist[0]
        log_weights_cyc = dist[-1]
    else:
        id_particles_cyc, log_weights_cyc = dist
    # convert graph ids to adjacency matrices
    particles_cyc = id2bit(id_particles_cyc, n_vars)

    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(particles_cyc, n_vars) == 0
    if is_dag.sum() == 0:
        # score as empty graph only
        particles = jnp.zeros((1, n_vars, n_vars), dtype=particles_cyc.dtype)
        log_weights = jnp.array([0.0], dtype=log_weights_cyc.dtype)

    else:
        particles = particles_cyc[is_dag, :, :]
        log_weights = log_weights_cyc[is_dag] - logsumexp(log_weights_cyc[is_dag])
        
    log_likelihood = eltwise_log_target(particles, x) / (x.shape[0] * x.shape[1])

     # - sum_G p(G | D) log(p(x | G))
    log_score, log_score_sgn = logsumexp(
        log_weights, b=log_likelihood, axis=0, return_sign=True)
    score = - log_score_sgn * jnp.exp(log_score)
    return score


#
# joint posterior p(G, theta | D) metrics
#

def neg_ave_log_likelihood(*, dist, eltwise_log_target, x, unknown_interv=False):
    """
    Computes neg. ave log marginal likelihood.

    Args:
        dist:  log distribution 3-tuple (joint distribution)
        eltwise_log_target: function satisfying [:, n_vars, n_vars], [:, n_vars, n_vars], [..., n_vars] -> [:, ]
            and computing p(D | G, theta) for held-out D=x
        x: [N, d]

    Returns: 
        [1, ]
    """

    n_vars = x.shape[-1]

    ids_cyc, theta_cyc, log_weights_cyc = dist[0], dist[1], dist[-1]
    hard_g_cyc = id2bit(ids_cyc, n_vars)

    # select acyclic graphs
    is_dag = elwise_acyclic_constr_nograd(hard_g_cyc, n_vars) == 0
    if is_dag.sum() == 0:
        # score as empty graph only
        hard_g = tree_mul(hard_g_cyc, 0.0)
        theta = tree_mul(theta_cyc, 0.0)
        log_weights = tree_mul(log_weights_cyc, 0.0)

    else:
        hard_g = hard_g_cyc[is_dag, :, :]
        theta = tree_select(theta_cyc, is_dag)
        log_weights = log_weights_cyc[is_dag] - logsumexp(log_weights_cyc[is_dag])
    
    if unknown_interv:
        # drop the theta for interventions
        theta = theta[:-1]
    log_likelihood = eltwise_log_target(hard_g, theta, x) / (x.shape[0] * x.shape[1])

    # - sum_G p(G, theta | D) log(p(x | G, theta))
    log_score, log_score_sgn = logsumexp(
        log_weights, b=log_likelihood, axis=0, return_sign=True)
    score = - log_score_sgn * jnp.exp(log_score)
    return score


