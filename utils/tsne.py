import os
import json
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


# t-SNE 可视化函数，并将结果保存为JSON文件
def visualize_tsne(features, labels, dataset_name):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(features.cpu())

    # 创建可视化图
    plt.figure(figsize=(8, 8))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels.cpu(), cmap='tab20', marker='o')
    plt.xlim(-80, 80)
    plt.ylim(-80, 80)

    output_dir = "./visual/distance/" + dataset_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存图像
    plt.savefig(os.path.join(output_dir, "tsne_" + dataset_name + ".png"))

    # 按标签分组数据并保存到JSON文件
    result_array = []
    labels_np = labels.cpu().numpy()
    unique_labels = np.unique(labels_np)

    for label in unique_labels:
        indices = np.where(labels_np == label)[0]
        points = []

        # 将每个点作为{x,y}对象存储
        for idx in indices:
            points.append({
                "x": float(tsne_result[idx, 0]),
                "y": float(tsne_result[idx, 1])
            })

        # 添加到结果数组
        result_array.append({
            "label": int(label),
            "points": points
        })

    # 保存为JSON文件
    json_path = os.path.join(output_dir, "tsne_data_" + dataset_name + ".json")
    with open(json_path, 'w') as f:
        json.dump(result_array, f)

    print(f"t-SNE数据已保存至: {json_path}")