from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.ops import Conv2dNormActivation

from ..builder import NECKS


@NECKS.register_module()
class AdjacentFeatureFusion(nn.Module):
    """
    Implements AdjacentFeatureFusion from the 'Ship Detection in SAR Images Based on Feature Enhancement Swin
    Transformer and Adjacent Feature Fusion' - <https://doi.org/10.3390/rs14133186>

    Module that adds a FPN from on top of a set of feature maps. This is based on `"Feature Pyramid Network for
    Object Detection" <https://arxiv.org/abs/1612.03144>.

    The input to the model is expected to be a List[Tensor], containing the feature maps on top of which the FPN will
    be added. The feature maps are currently supposed to be in increasing depth order.

    The following code is inspired by the PyTorch version of the FeaturePyramidNetwork.

    Args:
        in_channels_list (list[int]): number of channels for each feature map that is passed to the module
        out_channels (int): number of channels of the FPN representation
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    """

    _version = 2

    def __init__(
            self,
            in_channels_list: List[int],
            out_channels: int,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(Tensor(torch.rand(1))) for i in range(len(in_channels_list) - 1)])
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            inner_block_module = Conv2dNormActivation(
                in_channels, out_channels, kernel_size=1, padding=0, norm_layer=norm_layer, activation_layer=None
            )
            layer_block_module = Conv2dNormActivation(
                out_channels, out_channels, kernel_size=3, norm_layer=norm_layer, activation_layer=None
            )
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _load_from_state_dict(self,
                              state_dict,
                              prefix,
                              local_metadata,
                              strict,
                              missing_keys,
                              unexpected_keys,
                              error_msgs,
                              ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            num_blocks = len(self.inner_blocks)
            for block in ["inner_blocks", "layer_blocks"]:
                for i in range(num_blocks):
                    for type in ["weight", "bias"]:
                        old_key = f"{prefix}{block}.{i}.{type}"
                        new_key = f"{prefix}{block}.{i}.0.{type}"
                        if old_key in state_dict:
                            state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.inner_blocks):
            if i == idx:
                out = module(x)
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.layer_blocks):
            if i == idx:
                out = module(x)
        return out

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        """
        Computes the FPN for a set of feature maps.

        Args:
            x (List[Tensor]): feature maps for each feature level. The feature maps are currently supposed to be in
        increasing depth order.

        Returns:
            results (List[Tensor]): feature maps after FPN layers. They are ordered from the highest resolution first.
        """
        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        out = [self.get_result_from_layer_blocks(last_inner, -1)]

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")     # up sampling
            last_inner = inner_lateral + inner_top_down
            out.insert(0, self.get_result_from_layer_blocks(last_inner, idx))       # insert element in front

        out2 = [out[0]]
        for idx in range(0, len(out)-1):
            feat_shape = out[idx+1].shape[-2:]
            inner_up = F.interpolate(out[idx], size=feat_shape, mode='nearest')
            out2.append(inner_up * (1-self.weights[idx]) + out[idx+1] * self.weights[idx])

        out2.append(F.interpolate(out2[-1], scale_factor=0.5, mode='nearest'))
        return out2

