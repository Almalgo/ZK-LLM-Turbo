"""Helpers for synchronous and async encrypted layer execution."""

from __future__ import annotations

import asyncio

import numpy as np


def get_encrypted_layer_params(components, model_config, num_encrypted_layers: int):
    params = []
    for layer_idx in range(num_encrypted_layers):
        layer = components["layers"][layer_idx]
        params.append(
            (
                layer_idx,
                layer.input_layernorm.weight.detach().numpy(),
                layer.post_attention_layernorm.weight.detach().numpy(),
                model_config.rms_norm_eps,
            )
        )
    return params


def run_encrypted_layers(
    hidden_states: np.ndarray,
    protocol,
    encrypted_layer_params,
    position_offset: int,
) -> np.ndarray:
    for layer_idx, input_ln_w, post_attn_ln_w, eps in encrypted_layer_params:
        hidden_states = protocol.process_layer(
            hidden_states,
            layer_idx,
            input_ln_w,
            post_attn_ln_w,
            eps,
            position_offset=position_offset,
        )
    return hidden_states


async def run_encrypted_layers_async(
    hidden_states: np.ndarray,
    protocol,
    encrypted_layer_params,
    position_offset: int,
) -> np.ndarray:
    for layer_idx, input_ln_w, post_attn_ln_w, eps in encrypted_layer_params:
        hidden_states = await protocol.process_layer_async(
            hidden_states,
            layer_idx,
            input_ln_w,
            post_attn_ln_w,
            eps,
            position_offset=position_offset,
        )
    return hidden_states
