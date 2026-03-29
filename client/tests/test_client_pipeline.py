import asyncio

import numpy as np

from client.inference.pipeline import run_encrypted_layers, run_encrypted_layers_async


class DummyProtocol:
    def process_layer(
        self,
        hidden_states,
        layer_idx,
        input_layernorm_weight,
        post_attn_layernorm_weight,
        eps,
        position_offset=0,
    ):
        delta = layer_idx + input_layernorm_weight[0] + post_attn_layernorm_weight[0] + eps + position_offset
        return hidden_states + delta

    async def process_layer_async(
        self,
        hidden_states,
        layer_idx,
        input_layernorm_weight,
        post_attn_layernorm_weight,
        eps,
        position_offset=0,
    ):
        await asyncio.sleep(0)
        return self.process_layer(
            hidden_states,
            layer_idx,
            input_layernorm_weight,
            post_attn_layernorm_weight,
            eps,
            position_offset=position_offset,
        )


def test_async_pipeline_matches_sync_runner():
    protocol = DummyProtocol()
    hidden_states = np.ones((2, 4), dtype=np.float32)
    params = [
        (0, np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32), 0.1),
        (1, np.array([3.0], dtype=np.float32), np.array([4.0], dtype=np.float32), 0.1),
    ]

    sync_result = run_encrypted_layers(hidden_states.copy(), protocol, params, position_offset=5)
    async_result = asyncio.run(
        run_encrypted_layers_async(hidden_states.copy(), protocol, params, position_offset=5)
    )

    np.testing.assert_allclose(async_result, sync_result)
