#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
# Vendored verbatim from
#   https://github.com/apple/ml-ane-transformers/blob/main/ane_transformers/reference/layer_norm.py
# at upstream HEAD as of 2026-05-05. We do not pip-depend on `ane-transformers`
# because its setup.py strict-pins `torch<=1.11`, which is incompatible with
# torch 2.x. The actual code below is plain PyTorch and runs unchanged on torch 2.x+.
#

import torch
import torch.nn as nn


class LayerNormANE(nn.Module):
    """LayerNorm optimized for Apple Neural Engine (ANE) execution

    Note: This layer only supports normalization over the final dim. It expects `num_channels`
    as an argument and not `normalized_shape` which is used by `torch.nn.LayerNorm`.
    """

    def __init__(self, num_channels, clip_mag=None, eps=1e-5, elementwise_affine=True):
        """
        Args:
            num_channels:       Number of channels (C) where the expected input data format is BC1S. S stands for sequence length.
            clip_mag:           Optional float value to use for clamping the input range before layer norm is applied.
                                If specified, helps reduce risk of overflow.
            eps:                Small value to avoid dividing by zero
            elementwise_affine: If true, adds learnable channel-wise shift (bias) and scale (weight) parameters
        """
        super().__init__()
        # Principle 1: Picking the Right Data Format (machinelearning.apple.com/research/apple-neural-engine)
        self.expected_rank = len("BC1S")

        self.num_channels = num_channels
        self.eps = eps
        self.clip_mag = clip_mag
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels))

        self._reset_parameters()

    def _reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, inputs):
        input_rank = len(inputs.size())

        # Principle 1: Picking the Right Data Format (machinelearning.apple.com/research/apple-neural-engine)
        # Migrate the data format from BSC to BC1S (most conducive to ANE)
        if input_rank == 3 and inputs.size(2) == self.num_channels:
            inputs = inputs.transpose(1, 2).unsqueeze(2)
            input_rank = len(inputs.size())

        assert input_rank == self.expected_rank
        assert inputs.size(1) == self.num_channels

        if self.clip_mag is not None:
            inputs.clamp_(-self.clip_mag, self.clip_mag)

        channels_mean = inputs.mean(dim=1, keepdims=True)

        zero_mean = inputs - channels_mean

        zero_mean_sq = zero_mean * zero_mean

        denom = (zero_mean_sq.mean(dim=1, keepdims=True) + self.eps).rsqrt()

        out = zero_mean * denom

        if self.elementwise_affine:
            out = (out + self.bias.view(1, self.num_channels, 1, 1)) * self.weight.view(1, self.num_channels, 1, 1)

        return out


# Note: torch.nn.LayerNorm and LayerNormANE apply scale and bias terms in
# opposite orders. To accurately restore a state_dict trained using
# torch.nn.LayerNorm into LayerNormANE, we adjust the bias term: bias /= weight.
# This pre-hook can be registered on subclasses that load HF weights.
def correct_for_bias_scale_order_inversion(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    state_dict[prefix + "bias"] = state_dict[prefix + "bias"] / state_dict[prefix + "weight"]
    return state_dict


class LayerNormANELoadable(LayerNormANE):
    """LayerNormANE that auto-corrects HF LayerNorm bias/scale order on load."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_load_state_dict_pre_hook(correct_for_bias_scale_order_inversion)
