import math
import functools
import numpy as onp
import copy
from collections import defaultdict
from sergio.sergio_mod import Sergio as SergioSimulator


def grn_sergio(
    *,
    spec,
    rng,
    g,
    effect_sgn,
    toporder,
    n_vars,
    # specific inputs required in config
    b,  # function
    k_param,  # function
    k_sign_p,  # function
    hill,  # float
    decays,  # float?
    noise_params,  # float?
    cell_types,  # int
    noise_type='dpd',
    sampling_state=15,
    dt=0.01,
    # technical noise NOTE: not used for BaCaDI
    tech_noise_config=None,
    add_outlier_effect=False,
    add_lib_size_effect=False,
    add_dropout_effect=False,
    return_count_data=False,
    # interventions
    n_ko_genes=0,  # unique genes knocked out in all of data collected; -1 indicates all genes
):
    """
    SERGIO simulator for GRNs
    """

    # sample interaction terms K
    # NOTE k_param = uniform [1,5]
    k = onp.abs(k_param(rng, shape=(n_vars, n_vars)))
    # NOTE if we knew real graph, could force sign
    if effect_sgn is None:
        effect_sgn = rng.binomial(
            1, k_sign_p(rng, shape=(n_vars, 1)), size=g.shape) * 2.0 - 1.0
    else:
        assert onp.array_equal(g != 0, effect_sgn != 0)

    k = k * effect_sgn.astype(onp.float32)
    assert onp.array_equal(k != 0, effect_sgn != 0)
    assert set(onp.unique(effect_sgn)).issubset({-1.0, 0.0, 1.0})

    # sample number of cell types to be sampled (i.e. number of unique master regulator reproduction rates)
    n_cell_types = cell_types
    assert n_cell_types > 0, "Need at least 1 cell type to be simulated"

    # master regulator basal reproduction rate
    basal_rates = b(rng,
                    shape=(n_vars,
                           n_cell_types))  # assuming 1 cell type is simulated

    # hill coeff
    hills = hill * onp.ones((n_vars, n_vars))

    # sample targets for experiments performed (wild-type, knockout)
    ko_targets = []

    simulate_observ_data = spec['n_observations_obs'] > 0
    if simulate_observ_data:
        ko_targets += [None]

    simulate_interv_data = spec['n_observations_int'] > 0
    if simulate_interv_data:
        assert n_ko_genes != 0, f"Need n_ko_genes != 0 to have interventional data for SERGIO"
        if n_ko_genes == -1:
            n_ko_genes = n_vars
        ko_targets += sorted(
            rng.choice(n_vars, size=min(n_vars, n_ko_genes),
                       replace=False).tolist())

    # simulate for wild type and each ko target
    data = defaultdict(lambda: defaultdict(list))
    # collect interv_targets
    interv_targets = []
    for e, ko_target in enumerate(ko_targets):

        if ko_target is None:
            # observational/wild type
            data_type = "obs"
            kout = onp.zeros(n_vars).astype(bool)
            number_sc = math.ceil(spec['n_observations_obs'] / n_cell_types)

        else:
            # interventional/knockout
            data_type = "int"
            kout = onp.eye(n_vars)[ko_target].astype(bool)
            number_sc = math.ceil(spec['n_observations_int'] /
                                  (n_cell_types * n_ko_genes))

        # setup simulator
        sim = SergioSimulator(
            rng=rng,
            number_genes=n_vars,  # d
            number_bins=n_cell_types,  # c
            number_sc=number_sc,  # M
            noise_params=noise_params,
            noise_type=noise_type,
            decays=decays,
            sampling_state=sampling_state,
            kout=kout,
            dt=dt,
            safety_steps=10,
        )

        sim.custom_graph(
            g=g,
            k=k,
            b=basal_rates,  # def of cell type
            hill=hills,
        )

        # run steady-state simulations
        assert number_sc >= 1, f"Need to have number_sc >= 1: number_sc {number_sc} data_type {data_type}"
        sim.simulate()

        # Get the clean simulated expression matrix after steady_state simulations
        # shape: [number_bins (#cell types), number_genes, number_sc (#cells per type)]
        expr = sim.getExpressions()

        # Aggregate by concatenating gene expressions of all cell types into 2d array
        # [number_genes (#genes), number_bins * number_sc (#cells  = #cell_types * #cells_per_type)]
        expr_agg = onp.concatenate(expr, axis=1)

        # Now each row represents a gene and each column represents a simulated single-cell
        # Gene IDs match their row in this expression matrix
        # [number_bins * number_sc, number_genes]
        x = expr_agg.T
        x = rng.permutation(x, axis=0)

        # generate intervention mask
        # [number_bins * number_sc, number_genes] with True/False depending on whether gene was knocked out
        ko_mask = onp.tile(kout, (x.shape[0], 1)).astype(onp.float32)

        # environment indicator
        env = onp.array([e] * x.shape[0], dtype=int)
        # advance rng outside for faithfullness/freshness of data in for loop
        rng = copy.deepcopy(sim.rng)

        data[data_type]["x"].append(x)
        data[data_type]["ko_mask"].append(ko_mask)
        data[data_type]["env"].append(env)
        interv_targets.append(kout)

    ## NOTE OWN CODE HERE
    # concatenate interventional data by interweaving rows to have balanced intervention target counts
    if simulate_observ_data:
        x_obs = onp.stack(data["obs"]["x"]).reshape(-1, n_vars, order="F")
        x_obs_msk = onp.stack(data["obs"]["ko_mask"]).reshape(-1,
                                                              n_vars,
                                                              order="F")
        x_obs_env = onp.zeros(x_obs.shape[0], dtype=int)
    else:
        x_obs = onp.zeros((0, n_vars))  # dummy
        x_obs_msk = onp.zeros((0, n_vars))  # dummy
        x_obs_env = onp.zeros(0, dtype=int)  # dummy

    if simulate_interv_data:
        x_int = onp.stack(data["int"]["x"]).reshape(-1, n_vars, order="F")
        x_int_msk = onp.stack(data["int"]["ko_mask"]).reshape(-1,
                                                              n_vars,
                                                              order="F")
        x_int_env = onp.stack(data["int"]["env"]).reshape(-1, order="F")
    else:
        x_int = onp.zeros((0, n_vars))  # dummy
        x_int_msk = onp.zeros((0, n_vars))  # dummy
        x_int_env = onp.zeros(0, dtype=int)
    # clip number of observations to be invariant to n_cell_types due to rounding
    # [n_observations, n_vars]
    x_obs = x_obs[:spec['n_observations_obs']]
    x_int = x_int[:spec['n_observations_int']]
    x_obs_env = x_obs_env[:spec['n_observations_obs']]
    x_int_env = x_int_env[:spec['n_observations_int']]

    assert x_obs.size != 0 or x_int.size != 0, f"Need to sample at least some observations; " \
                                               f"got shapes x_obs {x_obs.shape} x_int {x_int.shape}"

    # collect data
    # boolean [n_env, d]
    interv_targets = onp.stack(interv_targets)
    return dict(
        g=g,
        n_vars=n_vars,
        x_obs=x_obs,
        x_int=x_int,
        int_envs=x_int_env,
        interv_targets=interv_targets,
    )


#####
sergio_clean = functools.partial(
    grn_sergio,
    add_outlier_effect=False,
    add_lib_size_effect=False,
    add_dropout_effect=False,
    return_count_data=False,
)
#####
"""
THESE BELOW AREN'T USED
"""

sergio_clean_count = functools.partial(
    grn_sergio,
    add_outlier_effect=False,
    add_lib_size_effect=False,
    add_dropout_effect=False,
    return_count_data=True,
    n_ko_genes=0,
)

sergio_noisy = functools.partial(
    grn_sergio,
    add_outlier_effect=True,
    add_lib_size_effect=True,
    add_dropout_effect=True,
    return_count_data=False,
    n_ko_genes=0,
)

sergio_noisy_count = functools.partial(
    grn_sergio,
    add_outlier_effect=True,
    add_lib_size_effect=True,
    add_dropout_effect=True,
    return_count_data=True,
    n_ko_genes=0,
)

# interventional versions
kosergio_clean = functools.partial(
    grn_sergio,
    add_outlier_effect=False,
    add_lib_size_effect=False,
    add_dropout_effect=False,
    return_count_data=False,
)

kosergio_clean_count = functools.partial(
    grn_sergio,
    add_outlier_effect=False,
    add_lib_size_effect=False,
    add_dropout_effect=False,
    return_count_data=True,
)

kosergio_noisy = functools.partial(
    grn_sergio,
    add_outlier_effect=True,
    add_lib_size_effect=True,
    add_dropout_effect=True,
    return_count_data=False,
)

kosergio_noisy_count = functools.partial(
    grn_sergio,
    add_outlier_effect=True,
    add_lib_size_effect=True,
    add_dropout_effect=True,
    return_count_data=True,
)
