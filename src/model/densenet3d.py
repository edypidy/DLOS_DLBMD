import torch
from torch import nn


class _DenseLayer(nn.Sequential):
    def __init__(self, in_channels: int, growth_rate: int, bn_size: int):
        super().__init__()
        out_channels = bn_size * growth_rate
        self.layers = nn.Sequential()
        self.layers.add_module("norm1", nn.BatchNorm3d(in_channels))
        self.layers.add_module("relu1", nn.ReLU(inplace=True))
        self.layers.add_module("conv1", nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False))
        self.layers.add_module("norm2", nn.BatchNorm3d(out_channels))
        self.layers.add_module("relu2", nn.ReLU(inplace=True))
        self.layers.add_module(
            "conv2", nn.Conv3d(out_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )


class _DenseBlock(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, bn_size: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size)
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            new_features = layer(x)
            features.append(new_features)
            x = torch.cat(features, dim=1)
        return x


class _Transition(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.add_module("norm", nn.BatchNorm3d(in_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool3d(kernel_size=2, stride=2))


class _TransitionMaintainSize(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.add_module("norm", nn.BatchNorm3d(in_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False))


class SpatialBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, depth: int = 4):
        super().__init__()
        for i in range(1, depth + 1):
            self.add_module(
                f"conv{i}",
                nn.Conv3d(
                    in_channels // (2 ** (i - 1)),
                    in_channels // (2**i),
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
            )
            self.add_module(f"norm{i}", nn.BatchNorm3d(in_channels // (2**i)))
            self.add_module(f"relu{i}", nn.ReLU(inplace=True))
        self.add_module(
            "conv_final",
            nn.Conv3d(in_channels // (2**depth), out_channels, kernel_size=1, stride=1, bias=False),
        )


class DLBMD(nn.Module):
    def __init__(
        self,
        growth_rate: int = 32,
        block_config=(6, 12, 24, 16),
        inverse_attention: bool = False,
        attentive_regularization: bool = False,
        split_denominator: int = 2,
        num_classes: int = 3,
        regression: bool = False,
    ):
        super().__init__()
        bn_size = 4
        in_channels = 64
        self.first_conv = nn.Sequential(
            nn.Conv3d(1, in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1, dilation=1),
        )

        self.dense_block1 = _DenseBlock(in_channels, growth_rate, bn_size, block_config[0])
        in_channels += block_config[0] * growth_rate
        self.transition1 = _TransitionMaintainSize(in_channels, in_channels // 2)
        in_channels = in_channels // 2

        self.dense_block2 = _DenseBlock(in_channels, growth_rate, bn_size, block_config[1])
        in_channels += block_config[1] * growth_rate

        self.inverse_attention = inverse_attention
        if inverse_attention:
            self.split_size = int(in_channels // split_denominator)
            self.attentive_regularization = attentive_regularization
            self.spatial_block = SpatialBlock(self.split_size, out_channels=1)
            self.sigmoid = nn.Sigmoid()

        self.transition2 = _Transition(in_channels, in_channels // 2)
        in_channels = in_channels // 2

        self.dense_block3 = _DenseBlock(in_channels, growth_rate, bn_size, block_config[2])
        in_channels += block_config[2] * growth_rate
        self.transition3 = _Transition(in_channels, in_channels // 2)
        in_channels = in_channels // 2

        self.dense_block4 = _DenseBlock(in_channels, growth_rate, bn_size, block_config[3])
        in_channels += block_config[3] * growth_rate

        self.norm5 = nn.BatchNorm3d(in_channels)
        self.relu5 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(in_channels, num_classes, bias=False)

        self.regression = regression
        if regression:
            self.regressor = nn.Linear(in_channels, 1, bias=False)

    def forward(self, x: torch.Tensor):
        x = self.first_conv(x)
        x = self.dense_block1(x)
        x = self.transition1(x)
        x = self.dense_block2(x)

        if self.inverse_attention:
            x_split = x[:, : self.split_size, ...]
            x_non_split = x[:, self.split_size :, ...]
            spatial_map = self.sigmoid(self.spatial_block(x_split))
            if not self.attentive_regularization:
                x_split = spatial_map * x_split
                x = torch.cat([x_split, x_non_split], dim=1)

        x = self.transition2(x)
        x = self.dense_block3(x)
        x = self.transition3(x)
        x = self.dense_block4(x)
        x = self.norm5(x)
        x = self.relu5(x)
        x = nn.functional.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.view(x.size(0), -1)
        out = self.classifier(x)

        if self.inverse_attention:
            if self.regression:
                reg = self.regressor(x)
                return out, reg, spatial_map
            return out, spatial_map

        if self.regression:
            reg = self.regressor(x)
            return out, reg
        return out


# Backward compatibility for existing imports/checkpoints.
DenseNet3D = DLBMD
