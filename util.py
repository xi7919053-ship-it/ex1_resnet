"""
util.py —— 显示工具
对应课件 79 页：训练完成后画出 loss / accuracy 随 epoch 变化的曲线，
用于直观判断模型是否收敛（loss该下降、acc该上升）。
"""

import matplotlib.pyplot as plt
import os


def plot_history(history, save_dir='outputs'):
    os.makedirs(save_dir, exist_ok=True)

    # ---- loss 曲线 ----
    plt.figure()
    plt.plot(history['train_loss'], label='train loss')
    plt.plot(history['test_loss'], label='test loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    # ---- accuracy 曲线 ----
    plt.figure()
    plt.plot(history['train_acc'], label='train acc')
    plt.plot(history['test_acc'], label='test acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy (%)')
    plt.title('Accuracy curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'acc_curve.png'))
    plt.close()

    print('曲线已保存至 {}/loss_curve.png 和 {}/acc_curve.png'.format(save_dir, save_dir))
