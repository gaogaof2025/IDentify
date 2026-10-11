import os
import random
import shutil

# 路径配置
datasets_path = "/home/iyouth/datasets/bridge_defect"
images_path = os.path.join(datasets_path, "images","val")
labels_path = os.path.join(datasets_path, "labels","val")

# 要保留的图片数量
keep_count = 300

# 获取所有图片文件
all_images = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
total_images = len(all_images)

print(f"总共找到 {total_images} 张图片")

# 如果图片总数小于等于要保留的数量，则不需要删除
if total_images <= keep_count:
    print(f"图片总数 ({total_images}) 小于或等于要保留的数量 ({keep_count})，无需删除。")
else:
    # 随机选择要删除的图片
    random.seed(42)  # 设置随机种子以确保可重复性
    images_to_delete = random.sample(all_images, total_images - keep_count)

    delete_count = 0
    label_delete_count = 0

    # 创建备份目录（可选）
    # backup_dir = os.path.join(datasets_path, "backup")
    # if not os.path.exists(backup_dir):
    #     os.makedirs(os.path.join(backup_dir, "images"), exist_ok=True)
    #     os.makedirs(os.path.join(backup_dir, "labels"), exist_ok=True)

    # 删除选定的图片和对应的标签文件
    for img_file in images_to_delete:
        img_path = os.path.join(images_path, img_file)

        # 计算对应的标签文件名（假设标签文件与图像文件同名但扩展名不同）
        label_name = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_path, label_name)

        # 移动到备份目录而不是直接删除（可选）
        # shutil.move(img_path, os.path.join(backup_dir, "images", img_file))
        # 删除图片文件
        if os.path.exists(img_path):
            os.remove(img_path)
            delete_count += 1

        # 删除对应的标签文件
        if os.path.exists(label_path):
            os.remove(label_path)
            label_delete_count += 1

    print(f"已删除 {delete_count} 张图片和 {label_delete_count} 个标签文件")
    print(f"现在数据集中有 {total_images - delete_count} 张图片")

# 确认剩余文件数量
remaining_images = len([f for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
remaining_labels = len([f for f in os.listdir(labels_path) if f.endswith('.txt')])

print(f"数据集中剩余 {remaining_images} 张图片和 {remaining_labels} 个标签文件")