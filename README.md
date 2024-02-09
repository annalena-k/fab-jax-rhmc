# Flow Annealed Importance Sampling Bootstrap (FAB) with Jax and Reflection Hamiltonian Monte Carlo (RHMC)

This fork of [`fabjax`](https://github.com/lollcat/FAB-JAX/) based on the [[FAB paper]](https://arxiv.org/abs/2208.01893) extends the existing code
to include reflections in the Hamiltonian Monte Carlo (HMC) update steps.

The reason for this is that HMC has a low efficiency on limited intervals as explained in the 
[`rhmc-jax` repository](https://github.com/annalena-k/rhmc-jax). As a result, HMC chains that move outside the limited
interval on which e.g. a neural spline flow is defined, are rejected in `fab-jax`.
To solve this problem, we include the `rhmc-jax` implementation based on
[reflection HMC (RHMC)](https://papers.nips.cc/paper_files/paper/2015/hash/8303a79b1e19a194f1875981be5bdb6f-Abstract.html) 
in the `fab-jax` package.

See `experiments` for training runs on various common problems using FAB.

## Installation
After cloning the `fab-jax` code locally, the package can be installed in editable mode via
```shell
pip install -e .
```
Since the `rhmc-jax` package is included in this fork via a git submodule, it has to be initialized with
```shell
git submodule init
```
and updated with
```shell
git submodule update
```
Alternatively, it can be initialized directly during the `git clone` by including `--recurse-submodules`
```shell
git clone --recurse-submodules https://github.com/annalena-k/fab-jax-rhmc.git
```

To install the submodule, enter the `rhmc-jax` folder in the command line (`cd rhmc-jax`) and run
```shell
pip install -e .
```


## Key tips
 - Please reach out to us if you would like us to help apply FAB to a problem of interest!
 - To pick hyperparameters begin with the defaults inside `experiments/configs` - these should give a solid starting point.
The most important hyper-parameters to tune are the number of iterations, the batch size, the number of intermediate distributions, the flow architecture, the MCMC transition operator, the learning rate schedule. 
 - For FAB to work well we need SMC to preform reasonably well, where by reasonble we just mean that it produces samples that are
better than samples from the flow by a noticeable margin.
If applying FAB to a new problem, make sure that transition operator works well (e.g. has a well tuned step size).
Having good plotting tools for visualising samples from the flow and SMC can be very helpful for diagnosing performance.
 - For getting started with a new problem we recommend starting with a small toy version of the problem, getting that to work
well, and then to move onto more challenging versions of the problem.


## Library
Key components of FAB:
- `sampling`: Running AIS/SMC with a trainable base q, and target p, targetting the optimal distribution for estimating alpha-divergence.
  - `Transition operators`: Currently the below MCMC transition operators are implemented.
     - `metropolis`: Propose step by adding Gaussian noise to sample and then accept/reject. Includes step size tuning.
     - `hmc`: Hamiltonean Monte Carlo. Simple step size tuning.
  - `point_is_valid_fn`: Provides users ability to specify which points are valid (invalid points are rejected within AIS/SMC). This can improve the efficiency of MCMC, and training stability.
    - `default_point_is_valid_fn`: Default setting. Rejects points with Nan values, or NaN density under the base/target.
    - `point_is_valid_if_in_bounds_fn`: Allows specification of bounds for a problem. Points that fall outside the bounds are rejected.
    - Write your own: The `point_is_valid_fn` is flexible to any criterion - so custom problem specific versions can be easily implemented.
- `buffer`: Prioritised replay buffer. 

these are written to be self-contained such that they can be easily ported into an existing code base.

Additionally, we have
 - `flow`: Create minimal real-nvp/spline normalizing flow for the gmm problem (using distrax).
 - `targets`: Target distributions to be fit.
 - `train`: Training script for fab (not modular, but can be copy and pasted and adapted).


## Experiments
Current problems include `cox`, `funnel` `gmm_v0` `gmm_v1` and `many_well`.

The performance of FAB for these problems in terms of accuracy in estimation of the 
log normalizing constant are shown in the below table. 
These results are estimated using 5 seeds (with standard error reported across seeds). For each seed we measure
the mean absolute error (MAE) in the estimation of the log normalizing constant using 2000 samples per estimate, 
where the MAE is averaged over 10 batches.

| GMM  (`gmm_v1`)        | Funnel | Cox | Many Well |
|------------------------|---|-----|--------|
| 0.00269 $\pm$ 0.000538 |  0.00218 $\pm$ 0.000506 | 0.194 $\pm$ 0.0394 |  0.0316 $\pm$ 0.00417  |

These problems may be run using the command
```shell
python experiments/gmm_v0.py 
```
When running the above command, ensure that you are in the repo's root directory with the PYTHONPATH environment variable
set to the root directory (`export PYTHONPATH=$PWD`). 


Additionally we have a quickstart notebook:

<a href="https://colab.research.google.com/github/lollcat/fab-jax/blob/master/experiments/fabjax_quickstart.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


We use the WANDB logger for training. 
If you have a WANDB account then simply change the config inside `experiments/config/{problem_name}.yaml` 
to match your WANDB project. Alternatively a `list_logger` or `pandas_logger` is available if you do not 
use WANDB (the list logger is used inside the Quickstart notebook). 



## Citation

If you use this code in your research, please cite the original `fab-jax` paper as:

> Laurence I. Midgley, Vincent Stimper, Gregor N. C. Simm, Bernhard Schölkopf, José Miguel Hernández-Lobato.
> Flow Annealed Importance Sampling Bootstrap. The Eleventh International Conference on Learning Representations. 2023.

**Bibtex**

```
@inproceedings{
midgley2023flow,
title={Flow Annealed Importance Sampling Bootstrap},
author={Laurence Illing Midgley and Vincent Stimper and Gregor N. C. Simm and Bernhard Sch{\"o}lkopf and Jos{\'e} Miguel Hern{\'a}ndez-Lobato},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=XCTVFJwS9LJ}
}
```