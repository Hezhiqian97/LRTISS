import torch
import torch.nn as nn

from torchstat import stat
from nets.Backbone import Backbone





class VMAF(nn.Module):
    def __init__(self, in_size, out_size, eps: float = 1e-4):
        super(VMAF, self).__init__()


        self.up = nn.UpsamplingBilinear2d(scale_factor=2)


        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True)
                                   )



        self.fuse_conv = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size=1, bias=False),
                                       nn.ReLU(inplace=True))
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=7, dilation=7, groups=out_size),
            nn.ReLU(inplace=True),

            )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=3, dilation=3, groups=out_size),
            nn.ReLU(inplace=True),

            )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=5, dilation=5, groups=out_size),
            nn.ReLU(inplace=True),

            )
        self.relu = nn.ReLU(inplace=True)

        self.sigmoid = nn.Sigmoid()
        self.eps = eps
        self.scale = nn.Parameter(torch.abs(torch.tensor(4.0)))

        self.shift_raw = nn.Parameter(torch.tensor(0.5))  # 内部参数，可以自由变化

    @property
    def shift(self):
        """通过一个sigmoid函数将内部参数约束到(0,1)区间"""
        return self.sigmoid(self.shift_raw)

    def forward(self, x_high, x_low):
        x_low = self.up(x_low)
        x0 = torch.cat([x_high, x_low], dim=1)

        x1 = self.conv1(x0)


        x2_1 = self.dwconv2(x1)
        x2_2 = self.conv_3(x1)
        x2_3 = self.conv_5(x1)
        x2 = x2_1 + x2_2 + x2_3
        x2 = self.fuse_conv(x2)
        B, C, H, W = x2.shape
        spatial_var = torch.var(x2, dim=(-2, -1), keepdim=True).pow(2)
        global_var_norm = spatial_var.sum(dim=[2, 3], keepdim=True) / (H * W - 1)
        attention_coef = (spatial_var / (self.scale * (global_var_norm + self.eps))) + self.shift
        attention_weight = self.sigmoid(attention_coef)
        spattention = x2 * attention_weight
        output = x1 + spattention

        return output



class EdgeAwareFeatureEnhancer(nn.Module):
    def __init__(self, in_channels):
        super(EdgeAwareFeatureEnhancer, self).__init__()
        self.edge_extractor = nn.AvgPool2d(kernel_size=11, stride=1, padding=5)
        self.weight_generator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        edge_features = x - self.edge_extractor(x)
        edge_weights = self.weight_generator(edge_features)
        enhanced_features = edge_weights * x+ x


        return enhanced_features


class LRTIIS(nn.Module):
    def __init__(self, num_classes=8):
        super(LRTIIS, self).__init__()
        self.barbk = Backbone()

        in_filters = [48, 80, 160, 192]
        out_filters = [32, 32, 64, 128]
        self.final = nn.Conv2d(out_filters[0], num_classes, 3, 1, 1, )

        self.up_conv = nn.Sequential(

            nn.UpsamplingBilinear2d(scale_factor=2),

            nn.Conv2d(out_filters[0], out_filters[0]//2, kernel_size=3, padding=1,groups=2),
            nn.ReLU(),
            nn.Conv2d(out_filters[0]//2, out_filters[0], kernel_size=3, padding=1,groups=2),
            nn.ReLU(),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=1),
           # nn.Conv2d(out_filters[0], out_filters[0], kernel_size=1),
        )
        self.up_concat4 = VMAF(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = VMAF(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = VMAF(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = VMAF(in_filters[0], out_filters[0])
        self.c1 = EdgeAwareFeatureEnhancer(out_filters[0])
        self.c2 = EdgeAwareFeatureEnhancer(out_filters[1])
        self.c3 = EdgeAwareFeatureEnhancer(out_filters[2])
        self.c4 = EdgeAwareFeatureEnhancer(out_filters[3])


    def forward(self, x):
        [feat1, feat2, feat3, feat4, feat5] = self.barbk.forward(x)
       # print(feat1.size(), feat2.size(), feat3.size(), feat4.size(), feat5.size())
        # 40*40
        up4 = self.up_concat4(feat4, feat5)

        up4 = self.c4(up4)

        up3 = self.up_concat3(feat3, up4)
        up3 = self.c3(up3)


        up2 = self.up_concat2(feat2, up3)
        up2 = self.c2(up2)

        # print("up2",up2.size())
        up1 = self.up_concat1(feat1, up2)
        up1 = self.c1(up1)

        up1 = self.up_conv(up1)

        final = self.final(up1)

        return final


if __name__ == '__main__':
    net = LRTIIS(num_classes=8)
    x = torch.randn((1, 3, 224, 224))

    with torch.no_grad():
        feats = net.barbk.forward(x)
        print("Backbone outputs shapes:")
        for i, f in enumerate(feats, 1):
            if f is None:
                print(f" feat{i}: None")
            else:
                print(f" feat{i}: {tuple(f.shape)}")

        try:
            out = net(x)
            print("Final output shape:", tuple(out.shape))
        except Exception as e:
            print("Forward 出错:", e)



    stat(net, (3, 224, 224))