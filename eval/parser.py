import argparse

from parso import parse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def make_evaluation_parser():
    """
    Returns argparse parser to control evaluation from command line
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_result_folder", type=str, default=None, help="folder")
    parser.add_argument("--method", type=str, default='bacadi', choices=["bacadi", "DCDI-G", "DCDI-DSF", "JCI-PC", "IGSP", "empty"], help="method")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--rel_cpu_usage", type=float, default=1.0, help="fraction of available CPUs allocated. if > 1, treated as integer for number of cpus")
    parser.add_argument("--smoke_test", action="store_true", help="If passed, minimal iterations to see if something breaks")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--callback", type=int, default=0, help="Whether or not to use a callback function when using bacadi")
    parser.add_argument("--callback_every", type=int, default=50, help="callback after every `n` steps")
    # parser.add_argument("--descr", required=True, help="set experiment filename; keep the same to resume in case interrupted")
    parser.add_argument("--create_targets_only", type=str2bool, default=False, help="Only create targets, do not evaluate.")
    parser.add_argument("--timeout", type=float, help="in seconds")
    parser.add_argument("--n_particles", default=20, type=int, help="number of posterior samples per Method")
    parser.add_argument("--use_wandb", type=int, default=0, help="Whether or not to use a wandb for logging")
    parser.add_argument("--use_wandb_final", type=int, default=0, help="Whether or not to log final metrics to wandb")

    '''Simulator'''
    parser.add_argument("--simulator", default="synthetic", choices=["synthetic", "sergio"], help="what simulator to use. either our own synthetic or sergio for GRN")
    parser.add_argument("--sergio_k_lower_lim", default=1, type=int, help="sergio simulator. k=unif[1,upper_lim]")
    parser.add_argument("--sergio_k_upper_lim", default=5, type=int, help="sergio simulator. k=unif[1,upper_lim]")
    parser.add_argument("--sergio_hill", default=2, type=int, help="sergio simulator")
    parser.add_argument("--sergio_cell_types", default=10, type=int, help="sergio simulator")
    parser.add_argument("--sergio_decay", default=0.8, type=float, help="sergio simulator")
    parser.add_argument("--sergio_noise_params", default=1.0, type=float, help="sergio simulator")

    '''Bootstrap'''
    parser.add_argument("--bootstrap_n_error_restarts", type=int, default=10, help="number of restarts for bootstrap")
    parser.add_argument("--no_bootstrap", action="store_true", help="If true, uses no bootstrapping but just runs once on full data")
    parser.add_argument("--n_bootstrap_samples", type=int, default=20, help="number of samples for bootstrap")

    '''Target'''
    parser.add_argument("--joint", action="store_true", help="If true, tunes evaluation of /joint/ posterior p(G, theta | D) methods")
    parser.add_argument("--infer_interv", action="store_true", help="If true, considers learning of _interventions_ from environments with unknown interventions")

    parser.add_argument("--n_variants", type=int, default=1, help="number of different targets optimized and averaged; equally assigned to batch rollouts")
    parser.add_argument("--n_rollouts", type=int, default=1, help="number of rollouts done per method for a single target")

    parser.add_argument("--target_graph", default="random", choices=["random", "linear_chain", "binary_tree", "dibs_paper"], help="what graph to use as target. if `dibs_paper`, uses overwrites n_vars to 4 nodes")
    parser.add_argument("--random_theta", type=bool, default=True, help="if the graph is not random (e.g. linear chain), whether to use random edge weights or default 1")

    # generative model
    parser.add_argument("--graph_prior", type=str, default="er", help="prior over graphs")
    parser.add_argument("--graph_prior_edges_per_node", type=int, default=2, help="parameter for prior over graphs")
    parser.add_argument("--inference_model", default="newbge", choices=["bge", "lingauss"], help="inference model")
    parser.add_argument("--joint_inference_model", default="lingauss", choices=["lingauss", "fcgauss", "sobolevgauss"], help="joint inference model")

    parser.add_argument("--n_vars", type=int, default=10, help="number of variables in graph")
    parser.add_argument("--n_observations", type=int, default=100, help="number of observations defining the ground truth posterior")
    parser.add_argument("--n_ho_observations", type=int, default=100, help="number of held out observations for validation")
    parser.add_argument("--n_intervention_sets", type=int, default=10, help="number of sets of observations sampled with random interventions")
    parser.add_argument("--n_interv_obs", type=int, default=10, help="number of observations sampled with interventions")
    parser.add_argument("--perc_intervened", type=float, default=0.2, help="percentage of nodes intervened upon")
    parser.add_argument("--intervention_type", default="random_all", choices=["random", "all_nodes", "perfect", "random_all"], help="what kind of interventional datasets to collect")
    parser.add_argument("--intervention_val", type=float, default=5.0, help="what value to set variables to in case of perfect interventions")
    parser.add_argument("--intervention_noise", type=float, default=0.5, help="the noise of the interventions in case of gaussian")
    parser.add_argument("--interv_prior_mean", type=float, default=0., help="prior mean for inferring interv. mean")
    parser.add_argument("--interv_prior_std", type=float, default=10., help="prior std for inferring interv mean")
    
    parser.add_argument("--n_posterior_g_samples", type=int, default=100, help="number of ground truth graph samples")

    # inference model
    parser.add_argument("--gbn_lower", type=float, default=1.0, help="GBN Sampler")
    parser.add_argument("--gbn_upper", type=float, default=3.0, help="GBN Sampler")
    parser.add_argument("--gbn_node_mean", type=float, default=0.0, help="GBN Sampler")
    parser.add_argument("--gbn_node_sig", type=float, default=1.0, help="GBN Sampler")
    parser.add_argument("--gbn_obs_sig", type=float, default=0.1, help="GBN Sampler")

    parser.add_argument("--lingauss_obs_noise", type=float, default=0.1, help="linear Gaussian")
    parser.add_argument("--lingauss_random_noise", type=bool, default=True, help="if noise for each node should be randomly sampled once at the beginning")
    parser.add_argument("--lingauss_mean_edge", type=float, default=0.0, help="linear Gaussian")
    parser.add_argument("--lingauss_sig_edge", type=float, default=1.0, help="linear Gaussian")
    parser.add_argument("--lingauss_init_sig_edge", type=float, default=1.0, help="linear Gaussian")

    parser.add_argument("--fcgauss_obs_noise", type=float, default=0.1, help="fully-connected NN Gaussian")
    parser.add_argument("--fcgauss_random_noise", type=bool, default=True, help="if noise for each node should be randomly sampled once at the beginning")
    parser.add_argument("--fcgauss_sig_param", type=float, default=1.0, help="fully-connected NN Gaussian")
    parser.add_argument("--fcgauss_init_sig_param", type=float, default=1.0, help="fully-connected NN Gaussian")
    parser.add_argument("--fcgauss_hidden_layers", type=int, default=1, help="fully-connected NN Gaussian")
    parser.add_argument("--fcgauss_n_neurons", type=int, default=5, help="fully-connected NN Gaussian")
    parser.add_argument("--fcgauss_activation", type=str, default="sigmoid", help="fully-connected NN Gaussian")
    parser.add_argument("--fcgauss_init_param", type=str, default="normal", help="fully-connected NN Gaussian")
    parser.add_argument("--fcgauss_bias", type=str2bool, default=True, help="fully-connected NN Gaussian")

    parser.add_argument("--bge_alpha_mu", type=float, default=1.0, help="BGe")
    parser.add_argument("--bge_alpha_lambd_add", type=float, default=2.0, help="BGe")

    parser.add_argument("--sobolevgauss_n_exp", type=int, default=10, help="Sobolev basis")
    parser.add_argument("--sobolevgauss_random_noise", type=bool, default=True, help="if noise for each node should be randomly sampled once at the beginning")
    parser.add_argument("--sobolevgauss_obs_noise", type=float, default=0.1, help="Sobolev basis")
    parser.add_argument("--sobolevgauss_sig_param", type=float, default=1.0, help="Sobolev basis")
    parser.add_argument("--sobolevgauss_init_sig_param", type=float, default=1, help="Sobolev basis")
    parser.add_argument("--sobolevgauss_mean_param", type=float, default=0.0, help="Sobolev basis")
    parser.add_argument("--sobolevgauss_init_param", type=str, default="uniform", help="fully-connected NN Gaussian")
    

    '''
    #
    # Shared args for marginal + joint bacadi
    #
    '''
    # inference model
    parser.add_argument("--bacadi_lingauss_obs_noise", type=float, default=0.1, help="linear Gaussian")
    parser.add_argument("--bacadi_lingauss_mean_edge", type=float, default=0.0, help="linear Gaussian")
    parser.add_argument("--bacadi_lingauss_sig_edge", type=float, default=1.0, help="linear Gaussian")
    parser.add_argument("--bacadi_lingauss_init_sig_edge", type=float, default=0.3, help="linear Gaussian")

    parser.add_argument("--bacadi_fcgauss_obs_noise", type=float, default=0.1, help="fully-connected NN Gaussian")
    parser.add_argument("--bacadi_fcgauss_sig_param", type=float, default=1.0, help="fully-connected NN Gaussian")
    parser.add_argument("--bacadi_fcgauss_init_sig_param", type=float, default=0.1, help="fully-connected NN Gaussian")
    parser.add_argument("--bacadi_fcgauss_hidden_layers", type=int, default=1, help="fully-connected NN Gaussian")
    parser.add_argument("--bacadi_fcgauss_n_neurons", type=int, default=5, help="fully-connected NN Gaussian")
    parser.add_argument("--bacadi_fcgauss_activation", type=str, default="sigmoid", help="fully-connected NN Gaussian")
    parser.add_argument("--bacadi_fcgauss_init_param", type=str, default="xavier_normal", help="fully-connected NN Gaussian")
    parser.add_argument("--bacadi_fcgauss_bias", type=str2bool, default=True, help="fully-connected NN Gaussian")

    parser.add_argument("--bacadi_bge_alpha_mu", type=float, default=1.0, help="BGe")
    parser.add_argument("--bacadi_bge_alpha_lambd_add", type=float, default=2.0, help="BGe")
    
    parser.add_argument("--bacadi_sobolevgauss_n_exp", type=int, default=10, help="Sobolev basis")
    parser.add_argument("--bacadi_sobolevgauss_obs_noise", type=float, default=0.1, help="Sobolev basis")
    parser.add_argument("--bacadi_sobolevgauss_sig_param", type=float, default=1.0, help="Sobolev basis")
    parser.add_argument("--bacadi_sobolevgauss_init_sig_param", type=float, default=0.1, help="Sobolev basis")
    parser.add_argument("--bacadi_sobolevgauss_mean_param", type=float, default=0.0, help="Sobolev basis")
    parser.add_argument("--bacadi_sobolevgauss_init_param", type=str, default="normal", help="fully-connected NN Gaussian")
    
    # optimizer
    parser.add_argument("--bacadi_optimizer",  default="rmsprop", choices=["gd", "momentum", "adagrad", "adam", "rmsprop"], help="optimizer for bacadi")
    parser.add_argument("--bacadi_optimizer_stepsize",  type=float, default=0.005, help="optimizer stepsize for bacadi")

    parser.add_argument("--model_seed", type=int, default=42, help="random seed for inference")
    
    parser.add_argument("--interv_data", action="store_true", help="if True, use interventional data instead of observational")
        
    parser.add_argument("--bacadi_interv_per_env", type=int, default=0, help="assume a certain number of interventions per environment")
    parser.add_argument("--bacadi_lambda_regul", type=float, default=1, help="lambda parameter in the sparsity regulariser for interventions")
    

    '''
    #
    # Marginal inference methods
    # 
    '''

    # bacadi
    parser.add_argument("--skip_bacadi", action="store_true")
    parser.add_argument("--bacadi_n_steps", type=int, default=2000, help="svgd maximum steps")
    parser.add_argument("--bacadi_latent_dim", type=int, help="svgd particles latent dim")
    parser.add_argument("--bacadi_rel_init_scale", type=float, default=1.0, help="svgd particles")
    parser.add_argument("--bacadi_opt_stepsize", type=float, default=0.005, help="learning rate")
    parser.add_argument("--bacadi_constraint_prior_graph_sampling", default="soft", choices=[None, "soft", "hard"], help="acyclicity constraint sampling")
    parser.add_argument("--bacadi_n_grad_mc_samples", type=int, default=128, help="svgd score function grad estimator samples")
    parser.add_argument("--bacadi_n_acyclicity_mc_samples", type=int, default=32, help="svgd score function grad estimator samples")
    parser.add_argument("--bacadi_score_function_baseline", type=float, default=0.0, help="gradient estimator baseline; 0.0 corresponds to not using a baseline")
    parser.add_argument("--bacadi_kernel", default="frob", choices=["frob"], help="marginal kernel")
    parser.add_argument("--bacadi_interv_kernel", default="frob-interv-add", choices=["frob-interv-add"], help="marginal kernel")
    parser.add_argument("--bacadi_grad_estimator_z", default="reparam", choices=["score", "reparam"], help="gradient estimator for z in marginal inference")
    
    parser.add_argument("--bacadi_alpha_linear", type=float, default=0.01, help="alpha linear default")
    parser.add_argument("--bacadi_beta_linear", type=float, default=1.0, help="beta linear default")
    parser.add_argument("--bacadi_tau_linear", type=float, default=1.0, help="tau linear default")

    parser.add_argument("--bacadi_alpha_expo", type=float, default=0.0, help="alpha expo default")
    parser.add_argument("--bacadi_beta_expo", type=float, default=0.0, help="beta expo default")
    parser.add_argument("--bacadi_tau_expo", type=float, default=0.0, help="tau expo default")

    parser.add_argument("--bacadi_ceil_alpha", type=float, default=1e9, help="maximum value for alpha")
    parser.add_argument("--bacadi_ceil_beta", type=float, default=1e9, help="maximum value for beta")
    parser.add_argument("--bacadi_ceil_tau", type=float, default=1e9, help="maximum value for tau")

    # priors of model
    parser.add_argument("--bacadi_graph_prior", nargs="+", default=["er", "sf"], help="graph prior")
    parser.add_argument("--bacadi_graph_prior_edges_per_node", type=int, default=2, help="parameter for prior over graphs")
    parser.add_argument("--bacadi_inference_model", default="newbge", choices=["bge", "newbge"], help="inference model")

    # kernel parameters
    parser.add_argument("--bacadi_h_latent", type=float, default=5, help="h parameter for latent z in kernel")
    parser.add_argument("--bacadi_h_interv", type=float, default=5, help="h parameter for latent gamma in kernel for interventions")

    '''
    #
    # Joint inference methods
    # 
    '''

    # bacadi
    parser.add_argument("--skip_joint_bacadi", action="store_true")
    parser.add_argument("--joint_bacadi_n_steps", type=int, default=3000, help="svgd maximum steps")
    parser.add_argument("--joint_bacadi_latent_dim", type=int, help="svgd particles latent dim")
    parser.add_argument("--joint_bacadi_rel_init_scale", type=float, default=1.0, help="svgd particles")
    parser.add_argument("--joint_bacadi_opt_stepsize", type=float, default=0.005, help="learning rate")
    parser.add_argument("--joint_bacadi_constraint_prior_graph_sampling", default="soft", choices=[None, "soft", "hard"], help="acyclicity constraint sampling")
    parser.add_argument("--joint_bacadi_n_grad_mc_samples", type=int, default=128, help="svgd score function grad estimator samples")
    parser.add_argument("--joint_bacadi_n_grad_batch_size", type=int, help="svgd observation minibatch size; if not specificed, uses the whole dataset")
    parser.add_argument("--joint_bacadi_n_acyclicity_mc_samples", type=int, default=32, help="svgd score function grad estimator samples")
    parser.add_argument("--joint_bacadi_score_function_baseline", type=float, default=0.0, help="gradient estimator baseline; 0.0 corresponds to not using a baseline")
    parser.add_argument("--joint_bacadi_kernel", default="frob-joint-add", choices=["frob-joint-add", "frob-joint-mul"], help="joint kernel")
    parser.add_argument("--joint_bacadi_interv_kernel", default="frob-joint-interv-add", choices=["frob-joint-interv-add"], help="joint kernel")
    parser.add_argument("--joint_bacadi_grad_estimator_z", default="reparam", choices=["score", "reparam"], help="gradient estimator for x in joint inference")
    
    # priors of model
    parser.add_argument("--joint_bacadi_graph_prior", default="er", choices=["er", "sf"], help="graph prior")
    parser.add_argument("--joint_bacadi_graph_prior_edges_per_node", type=int, default=2, help="parameter for prior over graphs")
    parser.add_argument("--joint_bacadi_inference_model", default="lingauss", choices=["lingauss", "fcgauss", "sobolevgauss"], help="joint inference model")

    # bacadi parameters
    parser.add_argument("--joint_bacadi_alpha_linear", type=float, default=0.01, help="alpha linear default")
    parser.add_argument("--joint_bacadi_beta_linear", type=float, default=1.0, help="beta linear default")
    parser.add_argument("--joint_bacadi_tau_linear", type=float, default=1.0, help="tau linear default")

    parser.add_argument("--joint_bacadi_alpha_expo", type=float, default=0.0, help="alpha expo default")
    parser.add_argument("--joint_bacadi_beta_expo", type=float, default=0.0, help="beta expo default")
    parser.add_argument("--joint_bacadi_tau_expo", type=float, default=0.0, help="tau expo default")

    parser.add_argument("--joint_bacadi_ceil_alpha", type=float, default=1e9, help="maximum value for alpha")
    parser.add_argument("--joint_bacadi_ceil_beta", type=float, default=1e9, help="maximum value for beta")
    parser.add_argument("--joint_bacadi_ceil_tau", type=float, default=1e9, help="maximum value for tau")

    # kernel parameters
    parser.add_argument("--joint_bacadi_h_latent", type=float, default=5, help="h parameter for latent z in kernel")
    parser.add_argument("--joint_bacadi_h_theta", type=float, default=500, help="h parameter for theta in kernel")
    parser.add_argument("--joint_bacadi_h_interv", type=float, default=5, help="h parameter for latent gamma in kernel for interventions")



    ## dcdi
    parser.add_argument("--skip_dcdi", action="store_true", help="if True, skip dcdi")
    # experiment
    parser.add_argument('--dcdi_exp_path', type=str, default='./store/dcdi/',
                        help='Path to experiments')
    parser.add_argument('--dcdi_train', action="store_true",
                        help='Run `train` function, get /train folder')
    parser.add_argument('--dcdi_retrain', action="store_true",
                        help='after to-dag or pruning, retrain model from scratch before reporting nll-val')
    parser.add_argument('--dcdi_dag_for_retrain', default=None, type=str, help='path to a DAG in .npy \
                        format which will be used for retrainig. e.g.  /code/stuff/DAG.npy')

    parser.add_argument('--dcdi_n_bootstrap_samples', type=int, default=5,
                        help='n bootstrap samples. lower for dcdi bc of comput. complexity')
    # data
    parser.add_argument('--dcdi_train_batch_size', type=int, default=64,
                        help='number of samples in a minibatch')
    parser.add_argument('--dcdi_num_train_iter', type=int, default=1000000,
                        help='number of meta gradient steps')
    parser.add_argument('--dcdi_normalize_data', action="store_true",
                        help='(x - mu) / std')
    
    # model
    parser.add_argument('--dcdi_model', type=str, default='DCDI-G',
                        help='model class (DCDI-G or DCDI-DSF)')
    parser.add_argument('--dcdi_num_layers', type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument('--dcdi_hid_dim', type=int, default=5,
                        help="number of hidden units per layer")
    parser.add_argument('--dcdi_nonlin', type=str, default='leaky-relu',
                        help="leaky-relu | sigmoid")
    parser.add_argument("--dcdi_flow_num_layers", type=int, default=1,
                        help='number of hidden layers of the DSF')
    parser.add_argument("--dcdi_flow_hid_dim", type=int, default=5,
                        help='number of hidden units of the DSF')


    # intervention
    parser.add_argument('--dcdi_intervention', action="store_true",
                        help="Use data with intervention")
    parser.add_argument('--dcdi_dcd', action="store_true",
                        help="Use DCD (DCDI with a loss not taking into account the intervention)")
    parser.add_argument('--dcdi_intervention_type', type=str, default="perfect",
                        help="Type of intervention: perfect or imperfect")
    parser.add_argument('--dcdi_intervention_knowledge', type=str, default="known",
                        help="If the targets of the intervention are known or unknown")
    parser.add_argument('--dcdi_coeff_interv_sparsity', type=float, default=1e-8,
                        help="Coefficient of the regularisation in the unknown \
                        interventions case (lambda_R)")

    # optimization
    parser.add_argument('--dcdi_optimizer', type=str, default="rmsprop",
                        help='sgd|rmsprop')
    parser.add_argument('--dcdi_lr', type=float, default=1e-3,
                        help='learning rate for optim')
    parser.add_argument('--dcdi_lr_reinit', type=float, default=None,
                        help='Learning rate for optim after first subproblem. Default mode reuses --dcdi_lr.')
    parser.add_argument('--dcdi_lr_schedule', type=str, default=None,
                        help='Learning rate for optim, change initial lr as a function of mu: None|sqrt_mu|log-mu')
    parser.add_argument('--dcdi_stop_crit_win', type=int, default=100,
                        help='window size to compute stopping criterion')
    parser.add_argument('--dcdi_reg_coeff', type=float, default=0.1,
                        help='regularization coefficient (lambda)')

    # Augmented Lagrangian options
    parser.add_argument('--dcdi_omega_gamma', type=float, default=1e-4,
                        help='Precision to declare convergence of subproblems')
    parser.add_argument('--dcdi_omega_mu', type=float, default=0.9,
                        help='After subproblem solved, h should have reduced by this ratio')
    parser.add_argument('--dcdi_mu_init', type=float, default=1e-8,
                        help='initial value of mu')
    parser.add_argument('--dcdi_mu_mult_factor', type=float, default=2,
                        help="Multiply mu by this amount when constraint not sufficiently decreasing")
    parser.add_argument('--dcdi_gamma_init', type=float, default=0.,
                        help='initial value of gamma')
    parser.add_argument('--dcdi_h_threshold', type=float, default=1e-8,
                        help='Stop when |h|<X. Zero means stop AL procedure only when h==0')

    # misc
    parser.add_argument('--dcdi_patience', type=int, default=10,
                        help='Early stopping patience in --dcdi_retrain.')
    parser.add_argument('--dcdi_train_patience', type=int, default=5,
                        help='Early stopping patience in --dcdi_train after constraint')
    parser.add_argument('--dcdi_train_patience-post', type=int, default=5,
                        help='Early stopping patience in --dcdi_train after threshold')

    # logging
    parser.add_argument('--dcdi_plot_freq', type=int, default=10000,
                        help='plotting frequency')
    parser.add_argument('--dcdi_no_w_adjs_log', action="store_true",
                        help='do not log weighted adjacency (to save RAM). One plot will be missing (A_\phi plot)')
    parser.add_argument('--dcdi_plot_density', action="store_true",
                        help='Plot density (only implemented for 2 vars)')
    parser.add_argument("--dcdi_callback_every", type=int, default=1000, help="callback after every `n` steps")
    
    # device and numerical precision
    parser.add_argument('--dcdi_gpu', action="store_true",
                        help="Use GPU")
    parser.add_argument('--dcdi_float', action="store_true",
                        help="Use Float precision")

    parser.add_argument('--dcdi_save_to_files', action="store_true",
                        help="if yes, store to pkl and files")
    
    ## JCI and others
    parser.add_argument("--skip_jci", action="store_true", help="if True, skip jci")
    
    parser.add_argument('--jci_indep_test_alpha', type=float, default=1e-2,
                        help='Cutoff value for independence tests')
    parser.add_argument('--jci_indep_test', type=str, default="gaussCItest",
                        help='Independence test used (gaussCItest or kernelCItest)')
    parser.add_argument('--jci_normalize_data', action="store_true",
                        help='(x - mu) / std')
    parser.add_argument('--jci_max_cond_set', type=int, default=None,
                        help='max size of cond. set for jci_pc')
    

    ##
    parser.add_argument("--skip_igsp", action="store_true", help="if True, skip igsp")
    
    parser.add_argument('--igsp_indep_test_alpha', type=float, default=1e-3,
                        help='Threshold for conditional indep tests')
    parser.add_argument('--igsp_indep_test_alpha_inv', type=float, default=1e-3,
                        help='Threshold for invariance tests')
    parser.add_argument('--igsp_indep_test', type=str, default='gaussian', choices=['gaussian', 'hci', 'kci'],
                        help='Type of conditional independance test to use \
                        (gaussian, hsic, kci)')
    parser.add_argument('--igsp_normalize_data', action="store_true",
                        help='(x - mu) / std')
    
    return parser