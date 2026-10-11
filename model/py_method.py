import torch.nn as nn
import torch
import torch.nn.functional as F

# self.local = FAA(1024, 1024)
# self.PRA = PRA(channel=512, h=24, w=12)
# PRA  通过捕捉输入特征图中的水平和垂直方向的空间依赖关系来增强特征表示
class PRA(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(PRA, self).__init__()

        self.h = h
        self.w = w
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)
        self.SFT_shift_conv0 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1,
                                         stride=1, bias=False)
        self.SFT_shift_conv1 = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1,
                                         stride=1, bias=False)
        self.SFT_scale_conv0 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1,
                                         stride=1, bias=False)
        self.SFT_scale_conv1 = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1,
                                         stride=1, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)
        avg_pool = self.avg_pool(x)

        # x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv = self.conv_1x1(torch.cat((x_h, x_w), 3))
        x_cat_conv_bn = self.bn(x_cat_conv)
        x_cat_conv_relu = self.relu(x_cat_conv_bn)

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(avg_pool), 0.1, inplace=True))
        scale = torch.sigmoid(scale)
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x), 0.1, inplace=True))
        shift = (x + shift) * scale

        out = (x * s_h.expand_as(x) * s_w.expand_as(x)) * shift
        return out

# FAA (simple vision)  捕捉图像特征的全局关系  增强输入特征  提高模型的全局感受野
class FAA(nn.Module):

    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 sub_sample=True,
                 bn_layer=True):
        super(FAA, self).__init__()
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            # 进行压缩得到channel个数
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c,  h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # [bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)

        # print(f.shape)

        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z
