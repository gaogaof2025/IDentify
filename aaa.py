import numpy as np
import math
import matplotlib.pyplot as plt


def linear_growth(epoch, max_epoch):
    """
    线性增长函数
    Args:
        epoch: 当前的epoch值。
        max_epoch: 总的训练epoch数。
    Returns:
        当前epoch对应的值（范围从0到1）。
    """
    return min(1, epoch / max_epoch)


def sigmoid_growth(epoch, max_epoch):
    """
    Sigmoid增长函数
    Args:
        epoch: 当前的epoch值。
        max_epoch: 总的训练epoch数。
    Returns:
        当前epoch对应的值（范围从0到1）。
    """
    # 调整映射，确保在epoch=0时输出接近0，在epoch=max_epoch时输出接近1
    x = 12 * (epoch / max_epoch - 0.5)
    return 1 / (1 + math.exp(-x))


def cosine_growth(epoch, max_epoch):
    """
    余弦增长函数
    Args:
        epoch: 当前的epoch值。
        max_epoch: 总的训练epoch数。
    Returns:
        当前epoch对应的值（范围从0到1）。
    """
    return 1 - math.cos(math.pi * 0.5 * epoch / max_epoch)


def sine_growth(epoch, max_epoch):
    """
    正弦增长函数
    Args:
        epoch: 当前的epoch值。
        max_epoch: 总的训练epoch数。
    Returns:
        当前epoch对应的值（范围从0到1）。
    """
    return math.sin(math.pi * 0.5 * epoch / max_epoch)


def log_growth(epoch, max_epoch):
    """
    对数增长函数
    Args:
        epoch: 当前的epoch值。
        max_epoch: 总的训练epoch数。
    Returns:
        当前epoch对应的值（范围从0到1）。
    """
    # 确保在epoch=0时精确输出0
    if epoch == 0:
        return 0
    x = epoch / max_epoch
    return math.log(1 + 9 * x) / math.log(10)


def exponential_growth(epoch, max_epoch):
    """
    指数增长函数
    Args:
        epoch: 当前的epoch值。
        max_epoch: 总的训练epoch数。
    Returns:
        当前epoch对应的值（范围从0到1）。
    """
    k = -5  # 控制增长速率的参数
    return (math.exp(k * epoch / max_epoch) - 1) / (math.exp(k) - 1)


if __name__ == '__main__':
    # 使用matplotlib绘制上述函数的图像

    # 生成数据（从0开始，包含最大值）
    epochs = [x for x in range(0, 21)]

    linear_values = [linear_growth(x, 20) for x in epochs]
    sigmoid_values = [sigmoid_growth(x, 20) for x in epochs]
    cosine_values = [cosine_growth(x, 20) for x in epochs]
    sine_values = [sine_growth(x, 20) for x in epochs]
    exponential_values = [exponential_growth(x, 20) for x in epochs]
    log_values = [log_growth(x, 20) for x in epochs]

    # 创建图形
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, linear_values, label='Linear Growth', color='blue', marker='o', markersize=6)
    plt.plot(epochs, sigmoid_values, label='Sigmoid Growth', color='orange', marker='s', markersize=6)
    plt.plot(epochs, cosine_values, label='Cosine Growth', color='green', marker='^', markersize=6)
    plt.plot(epochs, sine_values, label='Sine Growth', color='red', marker='v', markersize=6)
    plt.plot(epochs, exponential_values, label='Exponential Growth', color='purple', marker='d', markersize=6)
    plt.plot(epochs, log_values, label='Log Growth', color='brown', marker='|', markersize=6)

    # 添加标题和标签
    plt.title('Growth Methods')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.savefig('growth_methods.png')

    plt.show()
    # 保存图形
