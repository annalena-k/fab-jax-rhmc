from typing import NamedTuple, Sequence, Union

import chex
import distrax
import tensorflow_probability.substrates.jax as tfp
import jax.numpy as jnp

from fabjax.flow.flow import FlowRecipe, Flow, create_flow
from fabjax.flow.distrax_with_extra import ChainWithExtra
from fabjax.flow.build_coupling_bijector import build_split_coupling_bijector

class FlowDistConfig(NamedTuple):
    dim: int
    n_layers: int
    conditioner_mlp_units: Sequence[int]
    type: Union[str, Sequence[str]] = 'split_coupling'
    identity_init: bool = True
    compile_n_unroll: int = 2
    permute: bool = True


def build_flow(config: FlowDistConfig) -> Flow:
    recipe = create_flow_recipe(config)
    flow = create_flow(recipe)
    return flow


def create_flow_recipe(config: FlowDistConfig) -> FlowRecipe:
    flow_type = [config.type] if isinstance(config.type, str) else config.type
    for flow in flow_type:
        assert flow in ['split_coupling']

    def make_base() -> distrax.Distribution:
        base = distrax.MultivariateNormalDiag(loc=jnp.zeros(config.dim), scale_diag=jnp.ones(config.dim))
        return base

    def make_bijector():
        # Note that bijector.inverse moves through this forwards, and bijector.fowards reverses the bijector order
        bijectors = []

        if 'split_coupling' in flow_type:
            bijector = build_split_coupling_bijector(
                dim=config.dim,
                identity_init=config.identity_init,
                mlp_units=config.conditioner_mlp_units
            )
            bijectors.append(bijector)

        if config.permute:
            permutation = jnp.roll(jnp.arange(config.dim), 1)
            bijectors.append(tfp.bijectors.Permute(permutation, axis=-1))

        return ChainWithExtra(bijectors)


    definition = FlowRecipe(
        make_base=make_base,
        make_bijector=make_bijector,
        n_layers=config.n_layers,
        config=config,
        dim=config.dim,
        compile_n_unroll=config.compile_n_unroll,
        )
    return definition
