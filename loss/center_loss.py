from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F

class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()
        return loss


class ContrastiveCenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, margin=1.0, lambda_c=1.0):
        super(ContrastiveCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.margin = margin
        self.lambda_c = nn.Parameter(torch.tensor(lambda_c))
        # 初始化类别中心
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        """
        features: [batch_size, feat_dim]，模型输出的特征向量
        labels: [batch_size]，对应的标签
        """
        batch_size = features.size(0)
        # 计算类内损失
        centers_batch = self.centers.index_select(0, labels)
        intra_loss = F.mse_loss(features, centers_batch)

        # 计算类间对比损失
        centers_norm = F.normalize(self.centers, p=2, dim=1)
        dist_matrix = torch.mm(centers_norm, centers_norm.t())
        # 去掉对角线元素（自身到自身的距离）
        eye = torch.eye(self.num_classes, device=features.device)
        dist_matrix = dist_matrix * (1 - eye)
        # 计算类间损失，确保不同类别中心之间的距离大于 margin
        contrastive_loss = F.relu(self.margin - dist_matrix).mean()

        # 总损失
        loss = intra_loss + self.lambda_c * contrastive_loss
        return loss