from util import generate_base_command, generate_run_commands, hash_dict

from config import RESULT_DIR

import run_experiment
import argparse
import numpy as np
import copy
import os
import itertools
from parser import str2bool

applicable_configs = {
    "bacadi": [
        "bacadi_h_latent", "bacadi_h_interv", "joint_bacadi_h_latent",
        "joint_bacadi_h_theta", "joint_bacadi_h_interv", "joint_bacadi_n_steps",
        "bacadi_n_steps", "joint_bacadi_beta_linear", "joint_bacadi_alpha_linear",
        "bacadi_lambda_regul",
        #"interv_prior_mean", "interv_prior_std", "intervention_noise",
        #"intervention_val",
    ],
    "DCDI": [
        "dcdi_train_batch_size",
        "dcdi_coeff_interv_sparsity",
        "dcdi_h_threshold",
        "dcdi_reg_coeff",
        # "dcdi_num_train_iter"
    ],
    "JCI-PC": ["jci_indep_test_alpha", "jci_indep_test"],
    "IGSP": [
        "igsp_indep_test_alpha",
        "igsp_indep_test_alpha_inv",
        "igsp_indep_test",
    ],
}

default_configs = {
    # BaCaDI
    "bacadi_h_latent": 5,
    "bacadi_h_interv": 5,
    "joint_bacadi_h_latent": 5,
    "joint_bacadi_h_interv": 5,
    "joint_bacadi_h_theta": 500,
    "joint_bacadi_n_steps": 3000,
    "joint_bacadi_beta_linear": 1.,
    "bacadi_n_steps": 2000,
    "joint_bacadi_alpha_linear": 0.01,
    "bacadi_lambda_regul": 1.0,
    # for sergio:
    #"interv_prior_mean": -1,
    #"interv_prior_std": 1.,
    #"intervention_noise": 0.01,
    #"intervention_val": -1,
    # DCDI
    "dcdi_train_batch_size": 64,
    "dcdi_coeff_interv_sparsity": 1e-08,
    "dcdi_h_threshold": 1e-08,
    "dcdi_reg_coeff": 0.1,
    # JCI-PC
    "jci_indep_test_alpha": 0.01,
    "jci_indep_test": "gaussCItest",
    # IGSP
    "igsp_indep_test_alpha": 0.001,
    "igsp_indep_test_alpha_inv": 0.001,
    "igsp_indep_test": "gaussian",
}

search_ranges = {
    # BaCaDI
    "joint_bacadi_beta_linear": ['uniform', [1, 3]],
    "joint_bacadi_alpha_linear": ['loguniform', [-3, -1]],
    "joint_bacadi_h_latent": ['loguniform', [-1, 1.7]],
    "joint_bacadi_h_interv": ['loguniform', [-1, 1.7]],
    "joint_bacadi_h_theta": ['loguniform', [1.2, 5]],
    "bacadi_lambda_regul": ['loguniform', [-1, 2]],
    # # DCDI
    'dcdi_train_batch_size': ['log2uniform', [4, 7]],
    'dcdi_coeff_interv_sparsity': ['loguniform', [-8, -1]],
    'dcdi_reg_coeff': ['loguniform', [-3, 1]],
    "dcdi_h_threshold": ['loguniform', [-8, -6]],
    # # JCI-PC
    # 'jci_indep_test_alpha': ['loguniform', [-5, -1]],
    # # IGSP
    'igsp_indep_test_alpha': ['loguniform', [-5, -1]],
    'igsp_indep_test_alpha_inv': ['loguniform', [-5, -1]],
}

# check consistency of configuration dicts
assert set(itertools.chain(*list(applicable_configs.values()))) == {
    *default_configs.keys(), *search_ranges.keys()
}, "number of config params do not match"


def sample_flag(sample_spec, rds=None):
    if rds is None:
        rds = np.random
    assert len(sample_spec) == 2

    sample_type, range = sample_spec
    if sample_type == 'log2uniform':
        assert len(range) == 2
        return 2**rds.randint(*range)
    if sample_type == 'loguniform':
        assert len(range) == 2
        return 10**rds.uniform(*range)
    elif sample_type == 'uniform':
        assert len(range) == 2
        return rds.uniform(*range)
    elif sample_type == 'choice':
        return rds.choice(range)
    else:
        raise NotImplementedError


def main(args):
    rds = np.random.RandomState(args.seed)
    assert args.num_seeds_per_hparam < 50
    init_seeds = list(rds.randint(0, 10**6, size=(100, )))

    # determine name of experiment
    exp_base_path = os.path.join(RESULT_DIR, args.exp_name)
    method = args.method
    if method == 'bacadi' and args.joint:
        method += '_%s' % args.joint_bacadi_inference_model
    if args.infer_interv:
        setting = 'unknown_interv'
    elif args.interv_data:
        setting = 'known_interv'
    else:
        setting = 'obs'
    exp_path = os.path.join(exp_base_path,
                            '%s_%s_%s' % (args.simulator, method, setting))

    command_list = []
    for _ in range(args.num_hparam_samples):
        # transfer flags from the args
        flags = copy.deepcopy(args.__dict__)
        [
            flags.pop(key) for key in [
                'seed', 'num_hparam_samples', 'num_seeds_per_hparam',
                'exp_name', 'num_cpus', 'dry', 'mode', 'long', 'yes'
            ]
        ]

        # randomly sample flags
        for flag in default_configs:
            if flag in search_ranges:
                flags[flag] = sample_flag(sample_spec=search_ranges[flag],
                                          rds=rds)
            else:
                flags[flag] = default_configs[flag]

        # determine subdir which holds the repetitions of the exp
        flags_hash = hash_dict(flags)
        flags['exp_result_folder'] = os.path.join(exp_path, flags_hash)

        for j in range(args.num_seeds_per_hparam):
            seed = init_seeds[j]
            model_seed = init_seeds[50 + j]
            cmd = generate_base_command(
                run_experiment,
                flags=dict(**flags, **{
                    'seed': seed,
                    'model_seed': model_seed
                }))
            command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list,
                          num_cpus=args.num_cpus,
                          mode=args.mode,
                          dry=args.dry,
                          output_file=os.path.join(exp_path, flags_hash),
                          long=args.long,
                          promt=not args.yes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BaCaDI run')
    parser.add_argument(
        '--method',
        type=str,
        default='bacadi',
        choices=["bacadi", "DCDI-G", "DCDI-DSF", "JCI-PC", "IGSP"],
        help="method")
    parser.add_argument("--n_vars",
                        type=int,
                        default=10,
                        help="number of variables in graph")
    parser.add_argument(
        "--joint",
        default=True,
        type=str2bool,
        help=
        "If true, tunes evaluation of /joint/ posterior p(G, theta | D) methods"
    )
    parser.add_argument(
        "--infer_interv",
        default=True,
        type=str2bool,
        help=
        "If true, considers learning of _interventions_ from environments with unknown interventions"
    )
    parser.add_argument(
        "--interv_data",
        default=True,
        type=str2bool,
        help="if True, use interventional data instead of observational")

    parser.add_argument(
        "--simulator",
        default="synthetic",
        choices=["synthetic", "sergio"],
        help="what simulator to use. either our own synthetic or sergio for GRN"
    )

    parser.add_argument("--graph_prior",
                        type=str,
                        default="er",
                        choices=["er", "sf"],
                        help="prior over graphs")

    parser.add_argument("--joint_bacadi_inference_model",
                        default="lingauss",
                        choices=["lingauss", "fcgauss", "sobolevgauss"],
                        help="joint inference model for bacadi")

    parser.add_argument("--joint_inference_model",
                        default="lingauss",
                        choices=["lingauss", "fcgauss", "sobolevgauss"],
                        help="joint inference model for creation of data")

    parser.add_argument("--mode",
                        default="local_async",
                        choices=["local_async", "local"],
                        help="mode")

    parser.add_argument('--exp_name', type=str, required=True, default=None)

    parser.add_argument('--num_cpus',
                        type=int,
                        default=2,
                        help='number of cpus to use')

    parser.add_argument('--seed',
                        type=int,
                        default=7,
                        help='random number generator seed')
    parser.add_argument('--num_hparam_samples', type=int, default=20)
    parser.add_argument('--num_seeds_per_hparam', type=int, default=30)
    parser.add_argument('--dry', action='store_true')
    parser.add_argument('--long', action='store_true')
    parser.add_argument('--yes', action='store_true')
    
    parser.add_argument("--n_observations", type=int, default=100, help="number of observations defining the ground truth posterior")
    parser.add_argument("--n_interv_obs", type=int, default=10, help="number of observations per interv environment")
    parser.add_argument("--verbose", type=int, default=0)
    

    args = parser.parse_args()
    main(args)
