from typing import NamedTuple, Callable

import chex
import optax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from molboil.train.train import TrainConfig, Logger, ListLogger
from molboil.train.train import train

from fabjax.train.fab import build_fab_no_buffer_init_step_fns, LogProbFn, TrainStateNoBuffer
from fabjax.flow import build_flow, Flow, FlowDistConfig
from fabjax.sampling import build_ais, build_blackjax_hmc, AnnealedImportanceSampler, simple_resampling, \
    build_metropolis
from fabjax.targets.gmm import GMM
from fabjax.utils.plot import plot_marginal_pair, plot_contours_2D

class FABTrainConfig(NamedTuple):
    dim: int
    n_iteration: int
    batch_size: int
    plot_batch_size: int
    n_eval: int
    flow: Flow
    ais: AnnealedImportanceSampler
    plotter: Callable
    log_p_fn: LogProbFn
    optimizer: optax.GradientTransformation
    n_checkpoints: int = 0
    logger: Logger = ListLogger()
    seed: int = 0
    save: bool = False



def setup_plotter(flow, ais, log_p_fn, plot_batch_size, plot_bound: float):

    @jax.jit
    @chex.assert_max_traces(3)
    def get_data_for_plotting(state: TrainStateNoBuffer, key: chex.PRNGKey):
        x0 = flow.sample_apply(state.flow_params, key, (plot_batch_size,))

        def log_q_fn(x: chex.Array) -> chex.Array:
            return flow.log_prob_apply(state.flow_params, x)

        point, log_w, ais_state, ais_info = ais.step(x0, state.ais_state, log_q_fn, log_p_fn)
        x_ais = point.x
        _, x_ais_resampled = simple_resampling(key, log_w, x_ais)

        return x0, x_ais, x_ais_resampled

    def plot(state: TrainStateNoBuffer, key: chex.PRNGKey):
        x0, x_ais, x_ais_resampled = get_data_for_plotting(state, key)

        fig, axs = plt.subplots(3, figsize=(5, 15))
        plot_marginal_pair(x0, axs[0], bounds=(-plot_bound, plot_bound))
        plot_marginal_pair(x_ais, axs[1], bounds=(-plot_bound, plot_bound))
        plot_marginal_pair(x_ais_resampled, axs[2], bounds=(-plot_bound, plot_bound))
        plot_contours_2D(log_p_fn, axs[0], bound=plot_bound, levels=50)
        plot_contours_2D(log_p_fn, axs[1], bound=plot_bound, levels=50)
        plot_contours_2D(log_p_fn, axs[2], bound=plot_bound, levels=50)
        axs[0].set_title("flow samples")
        axs[1].set_title("ais samples")
        axs[2].set_title("resampled ais samples")
        plt.tight_layout()
        plt.show()
    return plot


def setup_fab_config():
    # Setup params

    # Train
    easy_mode = True
    alpha = 2.  # alpha-divergence param
    dim = 2
    n_iterations = int(2e4)
    n_eval = 10
    batch_size = 128
    plot_batch_size = 1000
    lr = 1e-4
    max_global_norm = 1.

    # Flow
    n_layers = 8
    conditioner_mlp_units = (64, 64)

    # AIS.
    use_hmc = False
    hmc_n_outer_steps = 1
    hmc_init_step_size = 1e-3
    metro_n_outer_steps = 1
    metro_init_step_size = 5.

    target_p_accept = 0.65
    tune_step_size = False
    n_intermediate_distributions = 4
    spacing_type = 'linear'


    # Setup flow and target.
    flow_config = FlowDistConfig(dim=dim, n_layers=n_layers, conditioner_mlp_units=conditioner_mlp_units)
    flow = build_flow(flow_config)

    if easy_mode:
        target_loc_scaling = 10
        n_mixes = 4
    else:
        target_loc_scaling = 30
        n_mixes = 40
    gmm = GMM(dim, n_mixes=n_mixes, loc_scaling=target_loc_scaling)

    # Setup AIS.
    if use_hmc:
        transition_operator = build_blackjax_hmc(dim=2, n_outer_steps=hmc_n_outer_steps,
                                                     init_step_size=hmc_init_step_size, target_p_accept=target_p_accept,
                                                     adapt_step_size=tune_step_size)
    else:
        transition_operator = build_metropolis(dim, metro_n_outer_steps, metro_init_step_size)
    ais = build_ais(transition_operator=transition_operator,
                    n_intermediate_distributions=n_intermediate_distributions, spacing_type=spacing_type,
                    alpha=alpha)

    # Optimizer
    optimizer = optax.chain(optax.zero_nans(), optax.clip_by_global_norm(max_global_norm), optax.adam(lr))

    # Plotter
    plotter = setup_plotter(flow=flow, ais=ais, log_p_fn=gmm.log_prob, plot_batch_size=plot_batch_size,
                            plot_bound=target_loc_scaling * 1.5)

    config = FABTrainConfig(dim=dim, n_iteration=n_iterations, batch_size=batch_size, flow=flow,
                            log_p_fn=gmm.log_prob, ais=ais, optimizer=optimizer, plot_batch_size=plot_batch_size,
                            n_eval=n_eval, plotter=plotter)

    return config


def setup_molboil_train_config(fab_config: FABTrainConfig) -> TrainConfig:
    """Convert fab_config into what we need for running the molboil training loop."""

    init, step = build_fab_no_buffer_init_step_fns(
        fab_config.flow, log_p_fn=fab_config.log_p_fn,
        ais = fab_config.ais, optimizer = fab_config.optimizer,
        batch_size = fab_config.batch_size)

    def eval_and_plot_fn(state, subkey, iteration, save, plots_dir) -> dict:
        fab_config.plotter(state, subkey)
        return {}

    train_config = TrainConfig(n_iteration=fab_config.n_iteration,
                n_checkpoints=fab_config.n_checkpoints,
                logger = fab_config.logger,
                seed = fab_config.seed,
                n_eval = fab_config.n_eval,
                init_state = init,
                update_state = step,
                eval_and_plot_fn = eval_and_plot_fn,
                save = fab_config.save
                )
    return train_config


if __name__ == '__main__':
    fab_config = setup_fab_config()
    train_config = setup_molboil_train_config(fab_config)
    train(train_config)