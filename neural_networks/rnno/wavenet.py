from types import SimpleNamespace
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import tree_utils
import x_xy
from x_xy.maths import safe_normalize


class Conv1D(nn.Module):
    n_channels: int
    dilation: int = 1
    kernel_size: int = 2

    @nn.compact
    def __call__(self, X):
        "X.shape = (bs, T, features)"
        field = self.kernel_size + self.dilation - 1
        causal_shift = (field - 1) / 2
        if (field % 2) == 0:
            causal_shift += 1
        causal_shift = int(causal_shift)
        X = jnp.pad(X, ((0, 0), (causal_shift, 0), (0, 0)))
        X = nn.Conv(
            self.n_channels,
            kernel_size=[self.kernel_size],
            kernel_dilation=self.dilation,
        )(X)
        if causal_shift > 0:
            X = X[:, :-causal_shift]
        return X


class _WaveNet(nn.Module):
    dilations: Sequence[int]
    residual_channels: int
    skip_channels: int
    final_channels: int
    skip_depth: int = 1
    filter_width: int = 2
    initial_filter_width: int = 32

    @nn.compact
    def __call__(self, X):
        X = tree_utils.batch_concat(X, num_batch_dims=2)
        assert X.ndim == 3

        X = Conv1D(self.residual_channels, kernel_size=self.initial_filter_width)(X)
        out = 0
        res = X
        for dilation in self.dilations:
            f = jnp.tanh(
                Conv1D(self.residual_channels, dilation, kernel_size=self.filter_width)(
                    res
                )
            )
            g = jax.nn.sigmoid(
                Conv1D(self.residual_channels, dilation, kernel_size=self.filter_width)(
                    res
                )
            )
            p = f * g
            out += Conv1D(self.residual_channels, kernel_size=1)(p)
            res += Conv1D(self.residual_channels, kernel_size=1)(p)
        for _ in range(self.skip_channels):
            out = Conv1D(self.skip_channels, kernel_size=1)(jax.nn.relu(out))
        out = Conv1D(self.final_channels, kernel_size=1)(jax.nn.relu(out))
        return out


def wavenet(
    sys: x_xy.base.System,
    n_residual_layers: int = 4,
    n_conv_layers_in_residual_layer: int = 11,
    residual_channels: int = 32,
    skip_channels: int = 512,
    skip_depth: int = 1,
    filter_width: int = 2,
    initial_filter_width: int = 32,
):
    dilations = [
        2**i for i in range(n_conv_layers_in_residual_layer)
    ] * n_residual_layers
    _wavenet = _WaveNet(
        dilations,
        residual_channels,
        skip_channels,
        _num_links_parent_not_root(sys.link_parents) * 4,
        skip_depth,
        filter_width,
        initial_filter_width,
    )

    def init(key, X):
        params = _wavenet.init(key, X)
        # unused state, only such that vmap operations work
        state = {"unused": jnp.zeros((1,))}
        return params, state

    def apply(params, state, X):
        X = _wavenet.apply(params, X)

        quats = {}
        idx = 0

        def build_quaternion_output(_, __, name: str, p: int):
            nonlocal idx
            if p == -1:
                return
            quats[name] = safe_normalize(X[..., idx * 4 : (idx + 1) * 4])
            idx += 1

        x_xy.scan.tree(
            sys,
            build_quaternion_output,
            "ll",
            sys.link_names,
            sys.link_parents,
        )
        assert idx * 4 == X.shape[-1]
        return quats, state

    return SimpleNamespace(init=init, apply=apply)


def _num_links_parent_not_root(parent_array: list[int]):
    return len([p for p in parent_array if p != -1])
