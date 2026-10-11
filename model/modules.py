import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Softmax, init



class SimAM(nn.Module):
    def __init__(self, lamda=1e-5):
        super().__init__()
        self.lamda = lamda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 获取输入张量的形状信息
        b, n, c = x.shape  # b: 批次大小, n: 补丁数量, c: 嵌入维度
        # 计算每个补丁的均值
        mean = torch.mean(x, dim=1, keepdim=True)
        # 计算每个补丁的方差
        var = torch.sum((x - mean) ** 2, dim=1, keepdim=True) / (n - 1)
        # 计算特征图的激活值
        e_t = (x - mean) ** 2 / (4 * (var + self.lamda)) + 0.5
        # 使用 Sigmoid 函数进行归一化
        out = self.sigmoid(e_t) * x
        return out

class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super().__init__()
        self.channel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, patches, C = x.size()
        proj_query = x.reshape(m_batchsize, C, -1)
        proj_key = x.reshape(m_batchsize, C, -1).permute(0, 2, 1)

        # proj_query = proj_query / torch.norm(proj_query, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        # proj_key = proj_key / torch.norm(proj_key, p=2, dim=1, keepdim=True).clamp(min=1e-6)

        energy = torch.bmm(proj_query, proj_key)
        max_energy_0 = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)
        energy_new = max_energy_0 - energy
        attention = self.softmax(energy_new)
        proj_value = x.reshape(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.reshape(m_batchsize, patches, C)

        gamma = self.gamma.to(out.device)
        out = gamma * out + x
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()
        self.gamma = Parameter(torch.zeros(1))


    def forward(self, x):
        x = x.permute(0, 2, 1)
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = output.permute(0, 2, 1)
        output = self.sigmoid(output)
        x = x.permute(0, 2, 1)


        return self.gamma * output + x

class SpatialFeatureAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SpatialFeatureAttention, self).__init__()
        self.conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, noise_x):

        # x 和 noise_x 形状：[batch_size, seq_len, embed_dim]
        # 将数据 reshape 为 [batch_size, embed_dim, height, width]
        # 这里需要知道 seq_len 对应的空间维度，可以将 seq_len 转换为 (H, W)
        batch_size, seq_len, embed_dim = x.shape
        height = width = int(seq_len ** 0.5)  # 假设是方形

        x_reshaped = x.transpose(1, 2).view(batch_size, embed_dim, height, width)
        noise_x_reshaped = noise_x.transpose(1, 2).view(batch_size, embed_dim, height, width)

        # 计算差异
        diff = x_reshaped - noise_x_reshaped

        # 通过卷积层
        attention = self.conv(diff)
        attention_weights = self.sigmoid(attention)

        # 调整特征
        adjusted_x = x_reshaped - attention_weights * diff

        # 恢复形状
        adjusted_x = adjusted_x.view(batch_size, embed_dim, seq_len).transpose(1, 2)

        return adjusted_x

class StripeLocal(nn.Module):

    def __init__(self, stripe_dim, **kwargs):
        super().__init__()
        self.stripe_dim = stripe_dim

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_pool = nn.MaxPool2d(2, 2)
        self.local_pool_2 = nn.MaxPool2d(4, 4)

    def forward(self, x):
        B, C, H, W = x.size()
        seps = x.chunk(C // self.stripe_dim, dim=1)
        result = []
        result_1 = []
        for sep in seps:
            t = self.local_pool(sep)
            result.append(t)
            t_1 = self.local_pool_2(sep)
            result_1.append(t_1)

        result = torch.cat(result, dim=1).contiguous()
        result_1 = torch.cat(result_1, dim=1).contiguous()

        x = self.global_pool(x)

        result = self.global_pool(result)
        result_1 = self.global_pool(result_1)

        return torch.cat((x, result, result_1), dim=1).contiguous()

# FAA (simple vision)  捕捉图像特征的全局关系  增强输入特征  提高模型的全局感受野
class FAA_ViT(nn.Module):
    def __init__(self, dim, reduction=2):
        super().__init__()
        self.dim = dim
        self.inter_dim = dim // reduction

        self.g = nn.Linear(dim, self.inter_dim)
        self.theta = nn.Linear(dim, self.inter_dim)
        self.phi = nn.Linear(dim, self.inter_dim)
        self.W = nn.Linear(self.inter_dim, dim)

        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)


    def forward(self, x):  # x: (B, P, C)
        B, P, C = x.shape

        g_x = self.g(x)               # (B, P, C')
        theta_x = self.theta(x)       # (B, P, C')
        phi_x = self.phi(x)           # (B, P, C')

        f = torch.matmul(theta_x, phi_x.transpose(1, 2))  # (B, P, P)
        f_div_C = F.softmax(f, dim=-1)                    # attention map

        y = torch.matmul(f_div_C, g_x)                    # (B, P, C')
        y = self.W(y)                                     # (B, P, C)

        return y + x  # 残差连接

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

class AGCA(nn.Module):
    def __init__(self, in_channel, ratio=3):
        super(AGCA, self).__init__()
        hide_channel = in_channel // ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channel, hide_channel, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(2)
        # Choose to deploy A0 on GPU or CPU according to your needs
        self.A0 = torch.eye(hide_channel).to('cuda')
        # self.A0 = torch.eye(hide_channel)
        # A2 is initialized to 1e-6
        self.A2 = nn.Parameter(torch.FloatTensor(torch.zeros((hide_channel, hide_channel))), requires_grad=True)
        init.constant_(self.A2, 1e-6)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(hide_channel, in_channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv1(y)
        B, C, _, _ = y.size()
        y = y.flatten(2).transpose(1, 2)
        A1 = self.softmax(self.conv2(y))
        A1 = A1.expand(B, C, C)
        A = (self.A0 * A1) + self.A2
        y = torch.matmul(y, A)
        y = self.relu(self.conv3(y))
        y = y.transpose(1, 2).view(-1, C, 1, 1)
        y = self.sigmoid(self.conv4(y))

        return x * y

class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)



