import torch
import torch.nn as nn
import torch.nn.functional as F
    
"""
3D Residual Block，3x3x3 conv ==> 3 smaller 3D conv, refered from DDRNet
"""

class Bottleneck3D(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        norm_layer,
        stride=1,
        dilation=[1, 1, 1],
        expansion=4,
        downsample=None,
        fist_dilation=1,
        multi_grid=1,
        bn_momentum=0.0003,
    ):
        super(Bottleneck3D, self).__init__()

        self.expansion = expansion
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(1, 1, 3),
            stride=(1, 1, stride),
            dilation=(1, 1, dilation[0]),
            padding=(0, 0, dilation[0]),
            bias=False,
        )
        self.bn2 = norm_layer(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(1, 3, 1),
            stride=(1, stride, 1),
            dilation=(1, dilation[1], 1),
            padding=(0, dilation[1], 0),
            bias=False,
        )
        self.bn3 = norm_layer(planes, momentum=bn_momentum)
        self.conv4 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(3, 1, 1),
            stride=(stride, 1, 1),
            dilation=(dilation[2], 1, 1),
            padding=(dilation[2], 0, 0),
            bias=False,
        )
        self.bn4 = norm_layer(planes, momentum=bn_momentum)
        self.conv5 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=(1, 1, 1), bias=False
        )
        self.bn5 = norm_layer(planes * self.expansion, momentum=bn_momentum)

        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

        self.downsample2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, stride, 1), stride=(1, stride, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample3 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )

    def forward(self, x):
        residual = x

        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out2_relu = self.relu(out2)

        out3 = self.bn3(self.conv3(out2_relu))
        if self.stride != 1:
            out2 = self.downsample2(out2)
        out3 = out3 + out2
        out3_relu = self.relu(out3)

        out4 = self.bn4(self.conv4(out3_relu))
        if self.stride != 1:
            out2 = self.downsample3(out2)
            out3 = self.downsample4(out3)
        out4 = out4 + out2 + out3

        out4_relu = self.relu(out4)
        out5 = self.bn5(self.conv5(out4_relu))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out5 + residual
        out_relu = self.relu(out)

        return out_relu

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, bn_momentum):
        super(Upsample, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                output_padding=1,
            ),
            norm_layer(out_channels, momentum=bn_momentum),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.main(x)


class Downsample(nn.Module):
    def __init__(self, feature, norm_layer, bn_momentum, expansion=8):
        super(Downsample, self).__init__()
        self.main = Bottleneck3D(
            feature,
            feature // 4,
            bn_momentum=bn_momentum,
            expansion=expansion,
            stride=2,
            downsample=nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(
                    feature,
                    int(feature * expansion / 4),
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                norm_layer(int(feature * expansion / 4), momentum=bn_momentum),
            ),
            norm_layer=norm_layer,
        )

    def forward(self, x):
        return self.main(x)
    
class ASPP(nn.Module):
    """
    ASPP 3D
    Adapt from https://github.com/cv-rits/LMSCNet/blob/main/LMSCNet/models/LMSCNet.py#L7
    """

    def __init__(self, planes, dilations_conv_list):
        super().__init__()

        # ASPP Block
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList(
            [
                nn.Conv3d(
                    planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False
                )
                for dil in dilations_conv_list
            ]
        )
        self.bn1 = nn.ModuleList(
            [nn.BatchNorm3d(planes) for dil in dilations_conv_list]
        )
        self.conv2 = nn.ModuleList(
            [
                nn.Conv3d(
                    planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False
                )
                for dil in dilations_conv_list
            ]
        )
        self.bn2 = nn.ModuleList(
            [nn.BatchNorm3d(planes) for dil in dilations_conv_list]
        )
        self.relu = nn.ReLU()

    def forward(self, x_in):

        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
        x_in = self.relu(y + x_in)  # modified

        return x_in

class MPAC(nn.Module): # Anisotropic convolution
    def __init__(self, channel, kernel=(3, 5, 7), dilation=(1, 1, 1), residual=False):
        super().__init__()
        self.channel = channel
        self.residual = residual
        self.n = len(kernel)  # number of kernels
        self.conv_mx = nn.Conv3d(channel, 3 * self.n, (1, 1, 1), stride=1, padding=0, bias=False, dilation=1)
        self.softmax = nn.Softmax(dim=2)  # Applies the Softmax function in each axis

        # ---- Convs of each axis
        self.conv_1x1xk = nn.ModuleList()
        self.conv_1xkx1 = nn.ModuleList()
        self.conv_kx1x1 = nn.ModuleList()

        c = channel
        for _idx in range(self.n):
            k = kernel[_idx]
            d = dilation[_idx]
            p = k // 2 * d
            self.conv_1x1xk.append(nn.Conv3d(c, c, (1, 1, k), stride=1, padding=(0, 0, p), bias=True, dilation=(1, 1, d)))
            self.conv_1xkx1.append(nn.Conv3d(c, c, (1, k, 1), stride=1, padding=(0, p, 0), bias=True, dilation=(1, d, 1)))
            self.conv_kx1x1.append(nn.Conv3d(c, c, (k, 1, 1), stride=1, padding=(p, 0, 0), bias=True, dilation=(d, 1, 1)))

    def forward(self, x):
        mx = self.conv_mx(x)  # (BS, 3n, D, H, W)
        _bs, _, _d, _h, _w = mx.size()
        mx = mx.view(_bs, 3, -1, _d, _h, _w)  # (BS, 3, n, D, H, W)

        mx = self.softmax(mx)  # dim=2

        mx_c = torch.unsqueeze(mx, dim=3)  # (BS, 3, n, 1, D, H, W)
        mx_c = mx_c.expand(-1, -1, -1, self.channel, -1, -1, -1)  # (BS, 3, n, c, D, H, W)
        mx_list = torch.split(mx_c, 1, dim=2)  # n x (BS, 3, 1, c, D, H, W)

        mx_z_list = []
        mx_y_list = []
        mx_x_list = []
        for i in range(self.n):
            mx_z, mx_y, mx_x = torch.split(torch.squeeze(mx_list[i], dim=2), 1, dim=1)  # 3 x (BS, 1, c, D, H, W)
            mx_z_list.append(torch.squeeze(mx_z, dim=1))  # (BS, c, D, H, W)
            mx_y_list.append(torch.squeeze(mx_y, dim=1))  # (BS, c, D, H, W)
            mx_x_list.append(torch.squeeze(mx_x, dim=1))  # (BS, c, D, H, W)

        # ------ x ------
        y_x = None
        for _idx in range(self.n):
            y1_x = self.conv_1x1xk[_idx](x)
            y1_x = F.relu(y1_x, inplace=True)
            y1_x = torch.mul(mx_x_list[_idx], y1_x)
            y_x = y1_x if y_x is None else y_x + y1_x

        # ------ y ------
        y_y = None
        for _idx in range(self.n):
            y1_y = self.conv_1xkx1[_idx](y_x)
            y1_y = F.relu(y1_y, inplace=True)
            y1_y = torch.mul(mx_y_list[_idx], y1_y)
            y_y = y1_y if y_y is None else y_y + y1_y

        # ------ z ------
        y_z = None
        for _idx in range(self.n):
            y1_z = self.conv_kx1x1[_idx](y_y)
            y1_z = F.relu(y1_z, inplace=True)
            y1_z = torch.mul(mx_z_list[_idx], y1_z)
            y_z = y1_z if y_z is None else y_z + y1_z

        # y = F.relu(y_z + x, inplace=True) if self.residual else F.relu(y_z, inplace=True)
        y = y_z + x if self.residual else y_z
        return y
