import torch


def pairwise_distance(query_features, gallery_features, metric='euclidean'):
    """计算特征之间的成对距离

    Args:
        query_features: 查询特征
        gallery_features: 库特征
        metric: 距离度量方式，'euclidean'或'cosine'
    """
    x = query_features
    y = gallery_features
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)

    if metric == 'cosine':
        # 余弦距离 = 1 - 余弦相似度
        x_norm = torch.nn.functional.normalize(x, dim=1, p=2)
        y_norm = torch.nn.functional.normalize(y, dim=1, p=2)
        dist = 1.0 - torch.mm(x_norm, y_norm.t())
    else:  # 欧氏距离
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist.addmm_(1, -2, x, y.t())
        dist = torch.clamp(dist, min=1e-12)

    return dist


def nfc(feat, strategy="knn", aggregation='sum', k1=5, k2=3, metric='cosine', weight_temp=10.0):
    """改进版的邻居特征协作

    Args:
        feat: 输入特征张量
        strategy: 邻居选择策略，'knn'或'mutual_knn'
        aggregation: 特征聚合方法，'sum'、'mean'或'weighted'
        k1: KNN的k值
        k2: 互近邻判断的k值（仅在mutual_knn策略下使用）
        metric: 距离度量方式，'euclidean'或'cosine'
        weight_temp: 加权聚合的温度参数（仅在weighted聚合方法下使用）
    """
    feat = feat.clone()
    feat = torch.nn.functional.normalize(feat, dim=1, p=2)

    # 计算特征间距离
    device = feat.device
    dist = pairwise_distance(feat, feat, metric=metric)

    # 排除自身
    eye = torch.eye(dist.size(0), device=device)
    dist = dist + eye * 1000

    # 获取最近邻索引
    val, rank = dist.topk(k1, largest=False)

    if strategy == "mutual_knn":
        # 互近邻策略
        neighbor_indices = []
        for i in range(rank.size(0)):
            mutual_list = []
            for j in rank[i]:
                if i in rank[j][:k2]:
                    mutual_list.append(j.item())
            neighbor_indices.append(mutual_list)
    else:  # 普通knn策略
        neighbor_indices = [[j.item() for j in rank[i]] for i in range(rank.size(0))]

    # 特征聚合
    feat_orig = feat.clone()
    for i in range(feat.size(0)):
        if len(neighbor_indices[i]) == 0:
            continue

        if aggregation == 'mean':
            # 平均聚合
            feat[i] = feat[i] + feat_orig[neighbor_indices[i]].mean(dim=0)
            # feat[i] = feat[i] + feat_orig[neighbor_indices[i]].sum(dim=0)
            feat[i] = feat[i] * 0.5 + feat_orig[neighbor_indices[i]].sum(dim=0) * 0.5
            # feat[i] = feat[i] + feat_orig[neighbor_indices[i]].mean(dim=0)*0.5 + feat_orig[neighbor_indices[i]].sum(dim=0)*0.5

        elif aggregation == 'weighted':
            # 加权聚合（基于距离）
            if len(neighbor_indices[i]) > 0:
                weights = torch.exp(-val[i, :len(neighbor_indices[i])] * weight_temp)
                weights = weights / weights.sum()
                weighted_feat = (feat_orig[neighbor_indices[i]] * weights.unsqueeze(1)).sum(dim=0)
                feat[i] = feat[i] + weighted_feat
        else:  # 'sum'
            # 求和聚合
            feat[i] = feat[i] + feat_orig[neighbor_indices[i]].sum(dim=0)

    # 再次归一化
    feat = torch.nn.functional.normalize(feat, dim=1, p=2)
    return feat


# 保持向后兼容性
def NFC(feat: torch.tensor, k1=2, k2=2):
    """原始NFC函数，保持向后兼容"""
    return nfc(feat, strategy="mutual_knn", aggregation='sum', k1=k1, k2=k2)