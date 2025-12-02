import torch
import torch.nn as nn
from timm.layers import trunc_normal_, DropPath, to_2tuple



def calculate_padding(kernel_size, padding=None, dilation=1):
    """Calculate padding size to ensure convolution output keeps the same dimensions"""
    if dilation > 1:
        # Calculate effective kernel size after dilation
        if isinstance(kernel_size, int):
            effective_kernel = dilation * (kernel_size - 1) + 1
        else:
            effective_kernel = [dilation * (k - 1) + 1 for k in kernel_size]
    else:
        effective_kernel = kernel_size

    # Automatically calculate padding size to ensure input and output dimensions are the same
    if padding is None:
        if isinstance(effective_kernel, int):
            padding = effective_kernel // 2
        else:
            padding = [k // 2 for k in effective_kernel]
    return padding


class ConvolutionLayer(nn.Module):
    """Standard convolution layer, including convolution, Batch Normalization, and activation function"""
    default_activation = nn.SiLU()  # Default activation function

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None,
                 groups=1, dilation=1, activation=True):
        """Initialize convolution layer"""
        super().__init__()
        # Create convolution layer, automatically calculate padding
        self.convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            calculate_padding(kernel_size, padding, dilation),
            groups=groups, dilation=dilation, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # Set activation function
        if activation is True:
            self.activation = self.default_activation
        elif isinstance(activation, nn.Module):
            self.activation = activation
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        """Forward propagation: Convolution -> Batch Normalization -> Activation"""
        return self.activation(self.batch_norm(self.convolution(x)))

    def forward_fused(self, x):
        """Forward propagation (fused version): Convolution -> Activation"""
        return self.activation(self.convolution(x))


class LEGR(nn.Module):
    def __init__(self, in_channels):
        # Multi-feature extraction module initialization
        super(LEGR, self).__init__()
        out_channels = in_channels // 2
        self.depthwise_conv_7x7 = nn.Conv2d(
            in_channels, in_channels, kernel_size=7, stride=1, padding=3, groups=in_channels
        )

        self.pointwise_conv_7x7 = ConvolutionLayer(in_channels, in_channels, kernel_size=1, stride=1)

        # 3x3 depthwise separable convolution branch
        self.depthwise_conv_3x3 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels
        )
        self.pointwise_conv_3x3 = ConvolutionLayer(out_channels, out_channels, kernel_size=1, stride=1)

        # 7x7 depthwise separable convolution branch placeholder

        # 5x5 depthwise separable convolution branch
        self.depthwise_conv_5x5 = nn.Conv2d(
            out_channels, out_channels, kernel_size=5, stride=1, padding=2, groups=out_channels
        )
        self.pointwise_conv_5x5 = ConvolutionLayer(out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.depthwise_conv_7x7(x)
        x = self.pointwise_conv_7x7(x)
        in_channels = x.size(1)
        assert in_channels % 2 == 0, f"Input channels {in_channels} must be divisible by 2"
        split_size = in_channels // 2
        x1, x2, = torch.split(x, split_size, dim=1)
        x1 = self.depthwise_conv_3x3(x1)
        x1 = self.pointwise_conv_3x3(x1)
        x2 = self.depthwise_conv_5x5(x2)
        x2 = self.pointwise_conv_5x5(x2)
        x = torch.cat([x1, x2], 1)
        return x


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class MRFA(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        # Use multi-layer small convolutions to replace large convolutions (Legacy comment)

        self.dwconv = LEGR(dim)

        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = LEGR(dim)
        self.act = nn.ReLU6(inplace=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class Backbone(nn.Module):
    def __init__(self, base_dim=16, depths=[1, 1, 3, 1], mlp_ratio=3, drop_path_rate=0.0, num_classes=1000, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 16
        # stem layer
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.GELU())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth
        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [MRFA(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]

            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))

        self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        features = []

        x = self.stem(x)
        features.append(x)
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


if __name__ == '__main__':
    model = Backbone()
    inputs = torch.randn((1, 3, 640, 640))
    res = model(inputs)
    for i in res:
        print(i.size())
    from torchstat import stat

    stat(model, (3, 224, 224))