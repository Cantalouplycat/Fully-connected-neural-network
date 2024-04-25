"""
File: model_test.py
Author: DM
Date: 2024/1/30,
Description: 左侧窗口为绘图区,右侧窗口为显示区,鼠标左键长按移动绘图,鼠标右键按住滑动清空.
"""
import NN_basis as Mnn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 读取模型权重
My_model = np.load('My_model.npz')
# 构建神经网络
Mynet = Mnn.Mnn_net(input_size=784, hidden_size=300, output_size=10)
# 导入模型权重
Mynet.Model_evaluation(v=My_model['v'], r=My_model['r'], w=My_model['w'], c=My_model['c'])
# 创建画布和坐标轴
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# 创建空白图像和变换
canvas = np.zeros((28, 28), dtype=np.float32)


# 定义鼠标事件处理函数
def on_mouse_move(event):
    if (event.button == 1 or event.button == 3) and event.xdata is not None and event.ydata is not None:
        x = int(event.xdata * 28)
        y = int(event.ydata * 28)
        if event.button == 1:  # 左键点击
            canvas[28 - y - 1:28 - y + 1, x - 1:x + 1] = 1
            ax[0].add_patch(Rectangle((x / 28, y / 28), 1 / 28, 1 / 28, edgecolor='none', facecolor='black'))
        elif event.button == 3:  # 右键点击
            canvas[:, :] = 0
            ax[1].clear()
            ax[0].clear()
        fig.canvas.draw_idle()


def on_mouse_release(event):
    if event.button == 1:
        ax[1].imshow(np.stack((canvas * 255,) * 3, axis=-1) / 255)
        x = np.ravel(canvas)
        Mynet.C_output(x)
        digit = np.argmax(Mynet.y)
        ax[1].set_title(f"Predicted Digit: {digit}")
        fig.canvas.draw_idle()


# 注册鼠标事件
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
fig.canvas.mpl_connect('button_release_event', on_mouse_release)

plt.show()
