import json
import os
import sys

from util import Logger, NumpyArrayEncoder

os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # avoids jax gpu warning
os.environ['JAX_ENABLE_X64'] = 'True'  # float64 precision
# os.environ['JAX_DEBUG_NANS'] = 'True'  # debugs NaNs
# os.environ['JAX_DISABLE_JIT'] = 'True'  # disables jit for debugging

import warnings

warnings.filterwarnings("ignore", message="CUDA_ERROR_NO_DEVICE")
warnings.filterwarnings("ignore", message="No GPU/TPU found")

import jax.numpy as jnp

from eval import eval_single_target
from parser import make_evaluation_parser

# from config.svgd import marginal_config, joint_config

import wandb


def main():

    parser = make_evaluation_parser()
    kwargs = parser.parse_args()

    jnp.set_printoptions(precision=4, suppress=True)
    exp_hash = str(abs(json.dumps(kwargs.__dict__, sort_keys=True).__hash__()))
    if kwargs.exp_result_folder is not None:
        os.makedirs(kwargs.exp_result_folder, exist_ok=True)
        log_file_path = os.path.join(kwargs.exp_result_folder,
                                     '%s.log' % exp_hash)
        logger = Logger(log_file_path)
        sys.stdout = logger
        sys.stderr = logger

    init_target_seed = kwargs.seed
    init_model_seed = kwargs.model_seed

    # if we use sergio simulator, force prior to be scale-free
    if kwargs.simulator == 'sergio':
        kwargs.graph_prior = 'sf'

    # smoke test NOT USED
    if kwargs.smoke_test:
        n_particles_loop = jnp.array([2])
        kwargs.n_rollouts = 1
        kwargs.n_variants = 1

    # Creates targets
    eval_single_target(method='gt',
                       graph_prior_str=kwargs.graph_prior,
                       target_seed=init_target_seed,
                       model_seed=init_model_seed,
                       group_id="None",
                       kwargs=kwargs,
                       load=False)

    if kwargs.create_targets_only:
        exit()

    group_id = wandb.util.generate_id()

    results = eval_single_target(method=kwargs.method,
                                 graph_prior_str=kwargs.graph_prior,
                                 target_seed=init_target_seed,
                                 model_seed=init_model_seed,
                                 group_id=group_id,
                                 kwargs=kwargs)

    if kwargs.exp_result_folder is None:
        from pprint import pprint
        pprint(results)
    else:
        exp_result_file = os.path.join(kwargs.exp_result_folder,
                                       '%s.json' % exp_hash)
        with open(exp_result_file, 'w') as f:
            json.dump(results, f, indent=4, cls=NumpyArrayEncoder)
        print('Dumped results to %s' % exp_result_file)


if __name__ == '__main__':
    main()
