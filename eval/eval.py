import copy
import functools
import wandb
import pandas as pd
import torch
from timeit import default_timer as timer
import time
from datetime import timedelta

from class_maker import make_bacadi, make_target
from baselines.bootstrap import run_bootstrap
from bacadi.utils.func import expected_graph, expected_interv
from result import get_metrics
from jax import random


def callback(target, config, **kwargs):
    if config.method == "bacadi":
        zs = kwargs["zs"]
        gs = kwargs["model"].particle_to_g_lim(zs)
        probs = kwargs["model"].edge_probs(zs, kwargs["t"])
        # TODO find some way to properly save images with wandb, not overwrite them?
        # visualize(probs, save_path=logger.fig_dir, t=kwargs["t"], show=False)

        bacadi_empirical = kwargs["model"].particle_empirical()
        bacadi_mixture = kwargs["model"].particle_mixture()

        metrics = {}
        metrics.update(get_metrics("bacadi_", bacadi_empirical, target, config))
        metrics.update(get_metrics("bacadi+_", bacadi_mixture, target, config))
        if config.use_wandb:
            wandb.log(metrics)
    else:
        raise ValueError(f"{config.method} not yet implemented in callback")
    return


def eval_single_target(*,
                       method,
                       graph_prior_str,
                       target_seed,
                       model_seed,
                       group_id,
                       kwargs,
                       load=True):
    config = copy.deepcopy(kwargs)
    config.method = method
    config.seed = target_seed
    config.model_seed = model_seed
    config.graph_prior = graph_prior_str
    config.bacadi_graph_prior = graph_prior_str
    config.joint_bacadi_graph_prior = graph_prior_str

    target, filename = make_target(config, load)

    if method == "gt":
        print("For GT method, no fitting done. Just created target")
        return
    if config.use_wandb:
        run = wandb.init(project=config.meta_descr,
                         name=f"{group_id}_t{target_seed}_r{model_seed}",
                         config=config,
                         group=group_id,
                         settings=wandb.Settings(start_method="fork"))

    data_type = 'interventional' if config.interv_data or config.infer_interv else 'observational'
    print('------------------------------------------------------------')
    print(
        f'Fitting and running {method} to get {"joint" if config.joint else "marginal"} model with {data_type} data.'
    )
    print(f'Seed: {model_seed}')
    if config.verbose: print(f'Target Graph:\n{target.g}')

    key = random.PRNGKey(model_seed)

    metrics = {}
    start = timer()
    t_before = time.time()

    if method == "empty":
        raise NotImplementedError("empty graph metrics tbd")
    #####################
    elif method == "bacadi":
        bacadi = make_bacadi(target=target,
                         config=config,
                         callback=functools.partial(callback,
                                                    target=target,
                                                    config=config),
                         key=key)

        if config.infer_interv:
            # data and datapoint -> environment mappin
            bacadi.fit(target.x_interv_data, target.envs)
        else:
            if config.interv_data:
                interv_targets = target.interv_targets
                # drop obs. setting
                if (interv_targets[0] == 0).all():
                    interv_targets = interv_targets[1:]
                bacadi.fit(target.x_interv_data, interv_targets, target.envs)
            else:
                bacadi.fit(target.x)
        # ... eval
        dist_empirical = bacadi.particle_empirical()
        dist_mixture = bacadi.particle_mixture()
        metrics.update(
            get_metrics("empirical_",
                        dist_empirical,
                        target,
                        config,
                        bacadi=bacadi,
                        final=True))
        metrics.update(
            get_metrics("mixture_",
                        dist_mixture,
                        target,
                        config,
                        bacadi=bacadi,
                        final=True))
    #####################
    elif method == "DCDI-G" or method == "DCDI-DSF":
        # adjust some default hparams, taken from dcdi/main
        if config.dcdi_lr_reinit is None:
            config.dcdi_lr_reinit = config.dcdi_lr

        # Use GPU
        if config.dcdi_gpu:
            if config.dcdi_float:
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.cuda.DoubleTensor')
        else:
            if config.dcdi_float:
                torch.set_default_tensor_type('torch.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.DoubleTensor')

        dist_empirical, dist_mixture, metrics_bootstrap = run_bootstrap(
            key, config, target, learner_str='dcdi')
        metrics.update(metrics_bootstrap)
        metrics.update(
            get_metrics("empirical_",
                        dist_empirical,
                        target,
                        config,
                        final=True))
        metrics.update(
            get_metrics("mixture_", dist_mixture, target, config, final=True))
    #####################
    elif method == "JCI-PC":
        dist_empirical, dist_mixture, metrics_bootstrap = run_bootstrap(
            key, config, target, learner_str='jci-pc')
        metrics.update(metrics_bootstrap)
        metrics.update(
            get_metrics("empirical_",
                        dist_empirical,
                        target,
                        config,
                        use_cpdag=True,
                        final=True))
        metrics.update(
            get_metrics("mixture_",
                        dist_mixture,
                        target,
                        config,
                        use_cpdag=True,
                        final=True))
    #####################
    elif method == "IGSP":
        dist_empirical, dist_mixture, metrics_bootstrap = run_bootstrap(
            key, config, target, learner_str='igsp')
        metrics.update(metrics_bootstrap)
        metrics.update(
            get_metrics("empirical_",
                        dist_empirical,
                        target,
                        config,
                        use_cpdag=True,
                        final=True))
        metrics.update(
            get_metrics("mixture_",
                        dist_mixture,
                        target,
                        config,
                        use_cpdag=True,
                        final=True))
    #####################
    else:
        raise ValueError(f"{method} method not implemented yet")
    #####
    t_after = time.time()
    end = timer()
    delta = timedelta(seconds=end - start)
    print(
        f'Fit and eval done for {method}. Time: {delta} --------------------------------------'
    )
    if not config.use_wandb and config.use_wandb_final:
        run = wandb.init(project=config.meta_descr,
                         name=f"{group_id}_t{target_seed}_r{model_seed}",
                         config=config,
                         group=group_id,
                         settings=wandb.Settings(start_method="fork"))
    if config.use_wandb or config.use_wandb_final:
        wandb.log(metrics)
        run.finish()

    results = dict(
        params=config.__dict__,
        target_filename=filename,
        duration=t_after - t_before,
        evals=metrics,
        g_gt=target.g,
        interv_targets_gt=target.interv_targets.astype(int),
        g_empirical=expected_graph(dist_empirical, config.n_vars),
        g_mixture=expected_graph(dist_mixture, config.n_vars),
        I_empirical=expected_interv(dist_empirical) if config.infer_interv else 0,
        I_mixture=expected_interv(dist_mixture) if config.infer_interv else 0,
    )

    return results
