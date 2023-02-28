from functools import partial
from typing import Callable, List, Optional

from mmcv.utils import to_2tuple
from torch import nn, Tensor
from torchvision.models.swin_transformer import SwinTransformerBlock, PatchMerging
from torchvision.ops import Permute

from mmdet.models.builder import BACKBONES


class FeatureEnhancement(nn.Module):
    """
    Implements of the Feature Enhancement module from the article 'Ship Detection in SAR Images Based on Feature
    Enhancement Swin Transformer and Adjacent Feature Fusion' - https://doi.org/10.3390/rs14133186

    Args:
        dim (int): Number of input channels
        img_size (int | tuple[int]): The size of input image
    """

    def __init__(self, dim: int, img_size: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        self.channel_information_integration = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim // 4, kernel_size=(1, 1)),
            norm_layer([dim // 4, img_size, img_size]),
            nn.ReLU(),
            nn.Conv2d(in_channels=dim // 4, out_channels=dim, kernel_size=(1, 1)),
            norm_layer([dim, img_size, img_size]),
        )

        self.spatial_information_integration = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim // 4, kernel_size=(3, 3), padding=1),
            norm_layer([dim // 4, img_size, img_size]),
            nn.ReLU(),
            nn.Conv2d(in_channels=dim // 4, out_channels=dim, kernel_size=(3, 3), padding=1),
            norm_layer([dim, img_size, img_size]),
        )

        self.sig = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with expected layout of [*, H, W, C]. It is already the result of the feature fusion

        Returns:
            Tensor with layout of [*, H/2, W/2, 2*C]
        """
        x = x.permute([0, 3, 1, 2])
        c = self.channel_information_integration(x)  # channel information integration
        s = self.spatial_information_integration(x)  # spatial information integration
        f = self.sig(x)  # result of the sigmoid function
        x = f * c + (1 - f) * s
        return x.permute([0, 2, 3, 1])


@BACKBONES.register_module()
class FESwin(nn.Module):
    """
    Implements FESwin from the 'Ship Detection in SAR Images Based on Feature Enhancement Swin Transformer and
    Adjacent Feature Fusion' - <https://doi.org/10.3390/rs14133186>

    The following code is inspired by the PyTorch version of the Swin Transformer.

    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 1000.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        downsample_layer (nn.Module, optional): Downsample layer (patch merging). Default: None.
    """
    def __init__(
            self,
            patch_size: List[int],
            embed_dim: int,
            depths: List[int],
            num_heads: List[int],
            window_size: List[int],
            img_size: int = 256,
            mlp_ratio: float = 4.0,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            stochastic_depth_prob: float = 0.1,
            init_cfg=None,
            num_classes: int = 1000,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            block: Optional[Callable[..., nn.Module]] = None,
            downsample_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes

        if block is None:
            block = SwinTransformerBlock
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)
        if downsample_layer is None:
            downsample_layer = PatchMerging

        # split image into non-overlapping patches
        self.pretreatment = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])),
            Permute([0, 2, 3, 1]),
            norm_layer(embed_dim),
        )

        # build SwinTransformer blocks
        total_stage_blocks = sum(depths)  # number of Swin block
        stage_block_id = 0
        self.stages = nn.ModuleList()  # list of stages
        self.featureEnhancements = nn.ModuleList()  # list of Feature Enhancement modules
        self.patchMerging = nn.ModuleList()  # list of patch merging layers

        for i_stage in range(len(depths)):  # for each stage
            dim = embed_dim * 2 ** i_stage  # calculate the feature dimension
            stage: List[nn.Module] = []  # list of layers in the current stage

            # add patch merging layer
            if i_stage >= 1:
                self.patchMerging.append(downsample_layer(embed_dim * 2 ** (i_stage - 1), norm_layer))

            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1

            self.stages.append(nn.Sequential(*stage))
            # add Feature Enhancement layer
            self.featureEnhancements.append(FeatureEnhancement(dim=dim, img_size=img_size // 2 ** (i_stage + 2)))

        self.permute = Permute([0, 3, 1, 2])  # B H W C -> B C H W

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Computes the FESwin.

        Args:
            x (Tensor): image

        Returns:
            results (List[Tensor]): feature maps. They are ordered from the highest resolution first.
        """
        x = self.pretreatment(x)
        x_stages: List[Tensor] = []
        for k in range(len(self.stages)):
            if k >= 1:
                x = self.patchMerging[k - 1](x)
            y = self.stages[k](x)
            xy = x + y
            x = self.featureEnhancements[k](xy)
            x_stages.append(self.permute(x))
            # x_stages.append(self.permute(x))
        return x_stages

    # def init_weights(self, pretrained=None):
    #     raise (NotImplementedError("No weights found"))
