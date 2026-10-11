import scipy.io
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn as nn


def distance_plot(qfeat, gfeat, qlabel, glabel, dataset_name):
    try:
        query_feature = torch.stack(qfeat)
        query_label = torch.stack(qlabel)
        gallery_feature = torch.stack(gfeat)
        gallery_label = torch.stack(glabel)

        # 将特征转换为numpy数组
        query_feature = query_feature.detach().cpu().numpy()
        gallery_feature = gallery_feature.detach().cpu().numpy()

        query_feature = query_feature / np.linalg.norm(query_feature, axis=1, keepdims=True)
        gallery_feature = gallery_feature / np.linalg.norm(gallery_feature, axis=1, keepdims=True)

        # 创建掩码以区分类内和类间距离
        mask = query_label[:, np.newaxis] == gallery_label

        # 计算余弦距离
        distmat = torch.FloatTensor(1 - np.matmul(query_feature, gallery_feature.T))  # Cosine distance

        # 应用掩码获取类内和类间距离
        intra = distmat[mask].numpy()
        inter = distmat[~mask].numpy()

        # 打印距离统计信息帮助调试
        print(f"类内距离: 最小={np.min(intra):.4f}, 最大={np.max(intra):.4f}, 平均={np.mean(intra):.4f}")
        print(f"类间距离: 最小={np.min(inter):.4f}, 最大={np.max(inter):.4f}, 平均={np.mean(inter):.4f}")

        ######################################################################
        plt.rcParams.update({'font.size': 14})

        fig, ax = plt.subplots(figsize=(8, 6))
        # 修改区间为0-2，更适合余弦距离
        b = np.linspace(0, 0.6, num=1000)

        # 绘制直方图
        n_intra, bins_intra, patches_intra = ax.hist(intra, bins=b, histtype="stepfilled",
                                                     alpha=0.6, color='#2c9678', density=True,
                                                     label='Intra-class')
        n_inter, bins_inter, patches_inter = ax.hist(inter, bins=b, histtype="stepfilled",
                                                     alpha=0.6, color='#f7de98', density=True,
                                                     label='Inter-class')

        ax.set_ylim(0, 9)  # 设置y轴范围
        ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 9])  # 设置y轴刻度
        ax.legend()

        # 设置标签
        ax.set_xlabel('Distance')
        ax.set_ylabel('Density')
        # 转换为大写字母
        ax.set_title(f'{dataset_name.upper()}')
        # 设置字体为Times New Roman
        plt.rc('font', family='Times New Roman')


        # 保存图像
        output_dir = "/home/gf/Downloads/visual/distance/" + dataset_name + "/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig.savefig(os.path.join(output_dir, 'dis1.png'), dpi=1000, format='png')
        plt.savefig(os.path.join(output_dir, 'dis3.png'), dpi=1000, format='png')

        # 关闭图形对象释放内存
        plt.close(fig)

        return {
            "intra_mean": np.mean(intra),
            "inter_mean": np.mean(inter)
        }

    except Exception as e:
        print(f"绘制距离图时出错: {str(e)}")
        return None