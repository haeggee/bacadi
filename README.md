# BaCaDI: Bayesian Causal Discovery with Unknown Interventions
Published at AISTATS 2023. Link: https://arxiv.org/abs/2206.01665

This is the full Python code for *BaCaDI*, a fully differentiable method for joint Bayesian inference of causal Bayesian networks and unknown interventions, as well as all baselines.

## Installation

The provided code uses `python` and various libraries, in particular `JAX`. The code was run on Linux and using `anaconda`. Note that it might not work on Mac because of an older Python version (that is not compatible with e.g. Apple Silicon).

To get Conda:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

To reproduce all results from the paper and run the code, install the following:

1. Inside this folder/repository, run the following to create a conda environment

   ```bash
   conda env create --name bacadi --file environment.yml
   conda activate bacadi
   pip install -e .
   pip install -r requirements.txt
   ```

2. Next, we need install the `CausalDiscoveryToolbox` which contains code for metrics as well as GPUtil

   ```bash
   pip install ./CausalDiscoveryToolbox/
   pip install ./gputil
   ```

   (The package `CausalDiscoveryToolbox` (`cdt 0.5.22`) is installed manually this way to allow fixing a bug inside their package. The same goes for GPUtil, even though no GPUs were used for our experiments.)

3. Lastly, we need to install all packages required for the `R` scripts underlying the JCI-PC baseline as well as `CausalDiscoveryToolbox`. For this, it should suffice to run

   ```bash
   chmod +x ./install_R.sh
   sudo ./install_R.sh
   ```
   (This is essentially the same code as the instructions given in the repository of [cdt](https://github.com/FenTechSolutions/CausalDiscoveryToolbox).
   
## Experiments
The experiments are launched with the scripts in `eval/`. You can e.g. run 
```bash
python eval/run_experiment.py --n_vars 10 --infer_interv --verbose 1
```
to launch BaCaDI with unknown interventions on a 10 node linear Gaussian BN with verbose output.

All results in the paper were achieved by a hyperparameter search that was launched via `eval/launch_experiments.py`. For example, to launch BaCaDI with 5 hyperparameter samples (with the search range as defined in the `launch_experiments.py` file) for 10 node linear Gaussian BNs, do

```bash
python eval/launch_experiments.py --exp_name <exp_name> --num_seeds_per_hparam 10 --n_vars 20 --num_hparam_samples 5 ---num_cpus 2
```
Note that this will launch 50 different async processes in parallel. Results will be written to `results/<exp_name>/`.

## Plots and Reproducibility

The folder in `results/` already contains all results that are shown in the paper, including hyperparameters and configs. The corresponding Figures as shown in our submission are included in the folder `plots/`. To reproduce these Figures, simply run 
```bash
python visualization/plot_from_path.py
```
Note that the code currently uses Latex which should be installed on your machine (this can be turned off in the code).

If you wish to reproduce the Figures in the Appendix, simply edit the `plot_from_path.py` file and replace `PLOT_DICTS` with `PLOT_DICTS_APPENDIX` at the beginning of the main loop. These dictionaries defined in `config.py` give the folder in which the corresponding experimental results are contained. By following the directory it is possible to inspect configurations and hyperparameters inside the JSON files.

## NOTE:
This code was extended starting from the open source implementation of [DiBS](https://github.com/larslorch/dibs).
