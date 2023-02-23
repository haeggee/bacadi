from jax import config
import jax.numpy as jnp
from jax.ops import index, index_update
import numpy as onp
import pandas as pd
import torch
import wandb
import random as rnd
from jax import random
import causaldag

import warnings

from baselines.dcdi.models.flows import DeepSigmoidalFlowModel
from baselines.igsp.igsp import run_igsp, run_ut_igsp
from bacadi.eval.target import make_inference_model, options_to_str
from bacadi.models.nonlinearGaussian import DenseNonlinearGaussianJAX

warnings.filterwarnings("ignore", message="No GPU automatically detected")

from bacadi.utils.graph import *

from baselines.dcdi.models.learnables import LearnableModel_NonLinGaussANM
from baselines.dcdi.data import DataManagerBaCaDI
from baselines.dcdi.train import compute_loss, train as train_dcdi
from baselines.jci.pc import PC


def compute_linear_gaussian_mle_params(x, envs, graph, interv_targets=None):
    """
    Computes MLE parameters for linear GBN
    See Hauser et al
    https://arxiv.org/pdf/1303.3216.pdf 
    Page 17
    Based on https://github.com/agrawalraj/active_learning
    from the paper Agrawal et al 
    https://arxiv.org/pdf/1902.10347.pdf
    """

    n_vars = x.shape[-1]

    coeffs = jnp.zeros_like(graph, dtype=x.dtype)

    # we can assume that envs is always sorted?
    envs_unique, counts = jnp.unique(envs, return_counts=True)
    sorted_ind = jnp.argsort(envs_unique)
    envs_unique, counts = envs_unique[sorted_ind], counts[sorted_ind]
    # covariances for different intv. distributions

    cov_Is = []
    for e in envs_unique:
        x_e = x[envs == e]
        cov_Is.append(x_e.T @ x_e)
    # [n_envs, d, d]
    cov_Is = jnp.stack(cov_Is)

    # for each node and its parents
    for j in range(n_vars):

        parents = jnp.where(graph[:, j] == 1)[0]
        if len(parents) > 0:
            if interv_targets is None:
                cov_mat = 1 / x.shape[0] * cov_Is[0]
            else:
                # [n_env,]
                not_intervened_on_j = interv_targets[:, j] != 1
                # num of datapoints where j not intervened
                n_j = counts[not_intervened_on_j].sum()
                cov_mat = 1 / n_j * cov_Is[not_intervened_on_j].sum(axis=0)

            cov_j_j = cov_mat[j, j]
            cov_j_pa = cov_mat[j, parents]
            cov_pa_pa = cov_mat[jnp.ix_(parents, parents)]

            if len(parents) > 1:
                inv_cov_pa_pa = jnp.linalg.inv(cov_pa_pa)
            else:
                inv_cov_pa_pa = jnp.array(1 / cov_pa_pa)

            mle_coeffs_pa_j = cov_j_pa.dot(inv_cov_pa_pa)

            # jax.numpy way for: coeffs[parents, j] = mle_coeffs_pa_j
            coeffs = coeffs.at[parents, j].set(mle_coeffs_pa_j)
    return coeffs


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _log_metrics(stage, step, metrics, throttle=None):
    wandb.log(metrics)


def _print_metrics(stage, step, metrics, **kwargs):
    for k, v in metrics.items():
        print("    %s:" % k, v)


def _log_nothing(**kwargs):
    pass


class DAGLearner:
    """
    Class as called by Bootstrap implementing an external DAG learning method
    """
    def __init__(self):
        super().__init__()
        self.final_try = False

    def learn_dag(self, x, envs, cpdag=None):
        """
        Learns DAG from data
            x :         [n_observations, n_vars]
            cpdag :     [n_vars, n_vars] if provided, returns random consistent extension of CPDAG

        Returns 
            dag:        [n_vars, n_vars] 
        """
        raise NotImplementedError

    def get_mle_params(self, x, envs, interv_targets, graph):
        """
        Computes MLE parameters for a given dataset + graph
        """

        if self.config.joint_bacadi_inference_model == 'lingauss':
            return compute_linear_gaussian_mle_params(
                x=x, envs=envs, interv_targets=interv_targets, graph=graph)

        else:
            raise NotImplementedError(
                f'No MLE parameter implementation available for type `{self.config.joint_inference_model}`'
            )

    def get_ave_negll(self,
                      x,
                      envs,
                      graph,
                      inference_model,
                      interv_targets=None,
                      params=None):
        """
            x: [N,d]
            envs: [N,]
            interv_targets: [n_env, d] or None
            graph: [d,d]
        
        returns:
            negll
        """
        if self.config.joint:
            if params is None:
                self.get_mle_params(x=x,
                                    envs=envs,
                                    graph=graph,
                                    interv_targets=interv_targets)

            negll = inference_model.log_likelihood(
                data=jnp.array(x),
                theta=jnp.array(params),
                w=jnp.array(graph),
                interv_targets=jnp.array(interv_targets, dtype=int)
                if interv_targets is not None else None,
                envs=jnp.array(envs))
        else:
            negll = inference_model.log_marginal_likelihood_given_g(
                w=jnp.array(graph),
                data=jnp.array(x),
                interv_targets=jnp.array(interv_targets, dtype=int)
                if interv_targets is not None else None,
                envs=jnp.array(envs))
        # mean
        negll = -negll / (x.shape[0] * x.shape[1])
        return negll

    def get_negll_all(self, x_train, envs_train, interv_targets_pred, graph):
        """
            x: [N,d]
            envs: [N,]
            interv_targets: [n_env - 1, d] or None (i.e. not with zero obs. targets)
            graph: [d,d]
        
        returns:
            negll, negll_ho, negll_interv_ho

        """
        if self.config.infer_interv or self.config.interv_data:
            interv_targets_with_obs = jnp.concatenate(
                (jnp.zeros(shape=(1, self.n_vars)), interv_targets_pred),
                axis=0)
        else:
            interv_targets_with_obs = None
        # get params for training data
        params = self.get_mle_params(x=x_train,
                                     envs=envs_train,
                                     graph=graph,
                                     interv_targets=interv_targets_with_obs)

        if self.config.joint:
            type_ = self.config.joint_bacadi_inference_model
            inference_model = make_inference_model(
                inference_str=type_,
                n_vars=self.config.n_vars,
                obs_noise=self.config.bacadi_lingauss_obs_noise,
                mean_edge=self.config.bacadi_lingauss_mean_edge,
                sig_edge=self.config.bacadi_lingauss_sig_edge,
                init_sig_edge=self.config.bacadi_lingauss_init_sig_edge,
                interv_mean=self.config.intervention_val,
                interv_noise=self.config.intervention_noise)

        else:
            type_ = self.config.inference_model
            inference_model = make_inference_model(
                inference_str=type_,
                n_vars=self.config.n_vars,
                alpha_mu=self.config.bacadi_bge_alpha_mu,
                alpha_lambd=self.config.n_vars +
                self.config.bacadi_bge_alpha_lambd_add,
                interv_mean=self.config.intervention_val,
                interv_noise=self.config.intervention_noise)

        negll = self.get_ave_negll(x=x_train,
                                   envs=envs_train,
                                   interv_targets=interv_targets_pred,
                                   graph=graph,
                                   params=params,
                                   inference_model=inference_model)

        negll_ho = self.get_ave_negll(x=self.target.x_ho,
                                      envs=jnp.zeros(
                                          self.target.n_ho_observations,
                                          dtype=int),
                                      interv_targets=None,
                                      graph=graph,
                                      params=params,
                                      inference_model=inference_model)
        negll_interv_ho = self.get_ave_negll(
            x=self.target.x_ho_interv_data,
            envs=self.target.envs_ho,
            interv_targets=self.target.interv_targets_ho,
            graph=graph,
            params=params,
            inference_model=inference_model)
        return negll, negll_ho, negll_interv_ho


class GIES(DAGLearner):
    '''
    Greedy interventional equivalence search

    'obs' : GaussL0penObsScore corresponds to BIC
        l0-penalized Gaussian MLE estimator. By default,
        score = log(L(D)) - k * log(n)/2
        corresponding exactly to BIC
        Specifically, assumes linear structural equation model with Gaussian noise
    
    
    'int' : GaussL0penIntScore is intended for a mixture of data sources
        i.e. observational and interventional data
    
    https://cran.r-project.org/web/packages/pcalg/vignettes/vignette2018.pdf
    https://rdrr.io/cran/pcalg/api/

    '''
    def __init__(self):
        super().__init__()
        raise NotImplementedError("GIES TBD")


class JCI_PC_Learner(DAGLearner):
    def __init__(self, target, config):
        super().__init__()
        self.target = target
        self.gt = onp.array(target.g)
        self.ci_alpha = config.jci_indep_test_alpha
        self.ci_alpha_init = config.jci_indep_test_alpha
        self.indep_test = config.jci_indep_test

        if config.infer_interv or config.interv_data:
            self.interv_targets = target.interv_targets
        else:
            self.interv_targets = None
        self.known_targets = not config.infer_interv
        self.n_vars = target.n_vars
        self.config = config

    def reinit(self, *, ci_alpha):
        """
        Re-initializes 
        """
        self.ci_alpha = ci_alpha

    def learn_dag(self, key, x, envs, cpdag=None):
        if self.config.jci_normalize_data:
            x = (x - x.mean(0)) / x.std(0)
        train_data_pd = pd.DataFrame(onp.array(x))
        regimes_pd = pd.DataFrame(onp.array(envs))
        obj = PC()
        interv_targets_gt = self.interv_targets if self.interv_targets is not None else None
        # drop obs. case
        if (interv_targets_gt[0] == 0).all():
            interv_targets_gt = interv_targets_gt[1:]
        ext_graph = obj._run_pc(train_data_pd,
                                regimes=regimes_pd,
                                mmax=self.config.jci_max_cond_set,
                                alpha=self.ci_alpha,
                                indep_test=self.indep_test,
                                known=self.known_targets,
                                targets=interv_targets_gt,
                                verbose=self.config.verbose)
        # first submatrix is the actual graph,
        # the lower rows indicate the interventional target
        # because JCI treats them as nodes in the extended graph
        dag_g = ext_graph[:self.n_vars, :self.n_vars]
        if self.interv_targets is not None:
            # interv_case
            interv_targets = ext_graph[self.n_vars:, :self.n_vars]
        else:
            interv_targets = onp.zeros((1, self.n_vars), dtype=int)[1:]

        # possibly direct some edges with I-MEC:
        dag_g_ = causaldag.classes.dag.DAG.from_amat(dag_g)
        interv_target_list = [
            set([i_ for i_ in range(self.n_vars) if t[i_]])
            for t in interv_targets
        ]
        cpdag_g = dag_g_.interventional_cpdag(
            interv_target_list, cpdag=dag_g_.cpdag()).to_amat()[0]
        if self.config.verbose:
            print("JCI-PC estimated CPDAG:")
            print(cpdag_g)
            print("JCI-PC estimated I:")
            print(interv_targets)

        key, subk = random.split(key)
        dag_g = random_consistent_expansion(
            key=subk,
            cpdag=cpdag_g,
            force_expansion_with_new_vee_structure=True)

        negll, negll_ho, negll_interv_ho = self.get_negll_all(
            x_train=x,
            envs_train=envs,
            interv_targets_pred=interv_targets,
            graph=dag_g)
        return cpdag_g, interv_targets, -negll, negll_ho, negll_interv_ho


class IGSPLearner(DAGLearner):
    def __init__(self, target, config):
        super().__init__()
        self.target = target
        self.gt = onp.array(target.g)
        self.ci_alpha = config.igsp_indep_test_alpha
        self.ci_alpha_init = config.igsp_indep_test_alpha
        self.ci_alpha_inv = config.igsp_indep_test_alpha_inv
        self.indep_test = config.igsp_indep_test

        if config.infer_interv or config.interv_data:
            self.interv_targets = target.interv_targets
        else:
            self.interv_targets = onp.zeros((1, target.n_vars), dtype=int)
        self.known_targets = not config.infer_interv
        self.n_vars = target.n_vars
        self.config = config

    def reinit(self, *, ci_alpha=None, ci_alpha_inv=None):
        """
        Re-initializes 
        """
        self.ci_alpha = ci_alpha or self.ci_alpha
        self.ci_alpha_inv = ci_alpha or self.ci_alpha_inv

    def learn_dag(self, key, x, envs, cpdag=None):
        key, subk = random.split(key)
        onp.random.seed(key[0])
        rnd.seed(key[0])
        if self.config.igsp_normalize_data:
            x = (x - x.mean(0)) / x.std(0)

        train_data_pd = pd.DataFrame(onp.array(x))
        regimes_pd = onp.array(envs)
        # masks: list of length [n_obs], where each element is a
        # list of intervened nodes
        masks = [[i for i in range(self.n_vars) if self.interv_targets[e][i]]
                 for e in envs]
        mask_pd = pd.DataFrame(masks)
        # run model
        if self.known_targets:
            cpdag, est_dag, targets_list = run_igsp(
                train_data_pd,
                targets=mask_pd,
                regimes=regimes_pd,
                alpha=self.ci_alpha,
                alpha_inv=self.ci_alpha_inv,
                ci_test=self.indep_test)
            if self.config.interv_data:
                interv_targets = jnp.array(self.interv_targets, dtype=int)
                # drop obs case
                if (interv_targets[0] == 0).all():
                    interv_targets = interv_targets[1:]
            else:
                interv_targets = None
        else:
            cpdag, est_dag, targets_list, est_targets = run_ut_igsp(
                train_data_pd,
                targets=mask_pd,
                regimes=regimes_pd,
                alpha=self.ci_alpha,
                alpha_inv=self.ci_alpha_inv,
                ci_test=self.indep_test)
            interv_targets = jnp.array(
                [[i in est_targets[e] for i in range(self.n_vars)]
                 for e in range(len(self.interv_targets) - 1)],
                dtype=int)

        key, subk = random.split(key)
        dag = random_consistent_expansion(
            key=subk, cpdag=cpdag, force_expansion_with_new_vee_structure=True)

        if self.config.verbose:
            print("(UT)-IGSP estimated CPDAG:")
            print(cpdag)
            print("(UT)-IGSP one DAG:")
            print(dag)
            print("(UT)-IGSP estimated I:")
            print(interv_targets)

        ### NEGLL ESTIMATION
        negll, negll_ho, negll_interv_ho = self.get_negll_all(
            x_train=x,
            envs_train=envs,
            interv_targets_pred=interv_targets,
            graph=dag)
        return cpdag, interv_targets, -negll, negll_ho, negll_interv_ho


class DCDILearner(DAGLearner):
    '''
    DCDI

    '''
    def __init__(self, target, config):
        super(DAGLearner, self).__init__()
        config_dict = {}
        # take dcdi config from argparse but drop dcdi prefix
        for k, v in vars(config).items():
            if k.startswith("dcdi_"):
                new_k = k[5:]
                config_dict[new_k] = v
        config_dict['num_vars'] = target.n_vars
        config_dict[
            'intervention_knowledge'] = "known" if not config.infer_interv else "unknown"
        config_dict['intervention'] = config.interv_data or config.infer_interv
        config_dict['method'] = config.method
        config_dict['verbose'] = config.verbose
        self.config = dotdict(config_dict)
        self.config_orig = config
        self.target = target
        self.n_vars = target.n_vars
        if config.use_wandb:
            self.callback = _log_metrics
        elif not config.use_wandb and config.verbose:
            self.callback = _print_metrics
        else:
            self.callback = _log_nothing

        if config.interv_data or config.infer_interv:
            # [n_env, d]
            self.interv_targets = onp.array(target.interv_targets)
            self.num_regimes = len(target.interv_targets)
        else:
            self.interv_targets = onp.zeros(target.n_vars)[None]
            self.num_regimes = 1

        self.test_data = DataManagerBaCaDI(
            adjacency=onp.array(self.target.g),
            data=onp.array(target.x_ho),
            interv_targets=onp.zeros(target.n_vars)[None],
            regimes=onp.array([0] * target.x_ho.shape[0]),
            num_regimes=1,
            train=False,
            normalize=config.dcdi_normalize_data,
            random_seed=0,  # does not matter here
            intervention=False,
            intervention_knowledge="known")

        # not used for now
        self.test_data_interv = DataManagerBaCaDI(
            onp.array(self.target.g),
            onp.array(target.x_ho_interv_data),
            interv_targets=onp.concatenate((
                onp.zeros(
                    (1,
                     target.n_vars)),  # add obs. since not included in heldout
                onp.array(target.interv_targets_ho))),
            regimes=onp.array(target.envs_ho),
            num_regimes=len(target.x_ho_interv) + 1,
            train=False,
            normalize=config.dcdi_normalize_data,
            random_seed=0,  # does not matter here
            intervention=True,
            intervention_knowledge="known")

        self.filename = config.dcdi_exp_path + config.dcdi_model + '_' + \
            options_to_str(
                d=target.n_vars,
                graph=config.graph_prior,
                edges_per_node=config.graph_prior_edges_per_node,
                n_obs=config.n_observations,
                n_interv_obs=config.n_interv_obs,
                n_ho=config.n_ho_observations,
                interventions=config.intervention_type,
                intervention_val=config.intervention_val,
                intervention_noise=config.intervention_noise,
            )

        # other methods have it, not used here
        self.ci_alpha_init = 0

    def reinit(self, **kwargs):
        """
        Re-initializes 
        """
        if self.config.verbose:
            print("Reinit for DCDI Learner does called, but does nothing")

    def learn_dag(self, *, key, x, envs):

        # Control as much randomness as possible
        torch.manual_seed(key[0])
        onp.random.seed(key[0])

        # change filename
        # -- otherwise DCDI will always reload previous model,
        # -- independent of the bootstrap or key
        filename = self.filename + "__" + str(key[0])
        self.config.exp_path = filename

        train_data = DataManagerBaCaDI(
            onp.array(self.target.g),
            x,
            self.interv_targets,
            regimes=envs,
            num_regimes=self.num_regimes,
            train=True,
            normalize=self.config.dcdi_normalize_data,
            random_seed=key[0],
            intervention=True,
            intervention_knowledge="known"
            if not self.config.infer_interv else "unknown")

        # create learning model and ground truth model
        if self.config.method == "DCDI-G":
            model = LearnableModel_NonLinGaussANM(
                self.n_vars,
                self.config.num_layers,
                self.config.hid_dim,
                nonlin=self.config.nonlin,
                intervention=self.config.intervention,
                intervention_type=self.config.intervention_type,
                intervention_knowledge=self.config.intervention_knowledge,
                num_regimes=train_data.num_regimes)
        elif self.config.method == "DCDI-DSF":
            model = DeepSigmoidalFlowModel(
                num_vars=self.n_vars,
                cond_n_layers=self.config.num_layers,
                cond_hid_dim=self.config.hid_dim,
                cond_nonlin=self.config.nonlin,
                flow_n_layers=self.config.flow_num_layers,
                flow_hid_dim=self.config.flow_hid_dim,
                intervention=self.config.intervention,
                intervention_type=self.config.intervention_type,
                intervention_knowledge=self.config.intervention_knowledge,
                num_regimes=train_data.num_regimes)
        model = train_dcdi(model,
                           train_data.adjacency.detach().cpu().numpy(),
                           train_data.gt_interv,
                           train_data,
                           self.test_data,
                           self.config,
                           metrics_callback=self.callback,
                           plotting_callback=None)
        model.eval()
        with torch.no_grad():
            weights, biases, extra_params = model.get_parameters(mode="wbx")
            weights_ = [weight.detach().cpu().numpy() for weight in weights]
            biases_ = [b.detach().cpu().numpy() for b in biases]
            # this is basically sigma for gaussians
            extra_params_ = jnp.array(
                [b.detach().cpu().numpy() for b in extra_params])[..., 0]
            extra_params_ = jnp.exp(extra_params_)
            print("NOISE:", extra_params_)
            # drop all weights not corresponding to obs. regime
            if self.config.intervention_knowledge == 'unknown':
                weights_ = [w[..., 0] for w in weights_]
                biases_ = [b[..., 0] for b in biases_]

            # transpose to put in same form as for bacadi
            weights_ = [w.transpose((0, 2, 1)) for w in weights_]
            theta = []
            for w, b in zip(weights_, biases_):
                theta.append((w, b))
                theta.append(())
            theta.pop()

            # get dag and intervention targets
            dag = model.adjacency.detach().cpu().numpy()

            if self.config.intervention_knowledge == 'known':
                interv_targets = self.interv_targets
            else:
                interv_targets = model.gumbel_interv_w.get_proba().detach(
                ).cpu().numpy()
                # drop environment/regime 0 since that is observational
                interv_targets = ((1 - interv_targets.T) > 0.5).astype(int)

            inference_model = DenseNonlinearGaussianJAX(
                obs_noise=self.config_orig.fcgauss_obs_noise,
                sig_param=self.config_orig.fcgauss_sig_param,
                hidden_layers=[self.config.hid_dim] * self.config.num_layers,
                activation='leakyrelu')  # ignore DSF params

            loglik_train = inference_model.log_likelihood(
                data=x,
                theta=theta,
                w=dag,
                interv_targets=interv_targets,
                envs=envs) / (x.shape[0] * x.shape[1])

            loglik_test = inference_model.log_likelihood(
                data=self.target.x_ho,
                theta=theta,
                w=dag,
                interv_targets=None,
                envs=None) / (self.target.x_ho.shape[0] *
                              self.target.x_ho.shape[1])

            loglik_test_interv = inference_model.log_likelihood(
                data=self.target.x_ho_interv_data,
                theta=theta,
                w=dag,
                interv_targets=self.target.interv_targets_ho,
                envs=self.target.envs_ho) / (
                    self.target.x_ho_interv_data.shape[0] *
                    self.target.x_ho_interv_data.shape[1])

        if self.config.verbose:
            print(f"{self.config.method} estimated DAG:")
            print(dag)
            print(f"{self.config.method} estimated I:")
            print(interv_targets)

        return dag, \
            interv_targets[1:], \
            loglik_train, \
            -loglik_test, \
            -loglik_test_interv
