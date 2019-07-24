#!/usr/bin/python
#  -*- coding:utf-8 -*-

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from pprint import pprint


def restore1(sigma, u, v, K):  # 奇异值、左特征向量、右特征向量
    m = len(u)
    n = len(v[0])
    a = np.zeros((m, n))
    for k in range(K):
        uk = u[:, k].reshape(m, 1)
        vk = v[k].reshape(1, n)
        a += sigma[k] * np.dot(uk, vk)   #内积
        a[a < 0] = 0
    a[a > 255] = 255
    # a = a.clip(0, 255)
    return np.rint(a).astype('uint8')


# def restore2(sigma, u, v, K):  # 奇异值、左特征向量、右特征向量
#     m = len(u)
#     n = len(v[0])
#     a = np.zeros((m, n))
#     for k in range(K+1):
#         for i in range(m):
#             a[i] += sigma[k] * u[i][k] * v[k]
#     a[a < 0] = 0
#     a[a > 255] = 255
#     return np.rint(a).astype('uint8')


if __name__ == "__main__":
    A = Image.open("6.son.png", 'r')
    output_path = r'.Pic'
    if not os.path.exists(output_path):
        os.mkdir(output_path)       #若输出路径不存在新建一个路径
    a = np.array(A)
    K = 50
    u_r, sigma_r, v_r = np.linalg.svd(a[:, :, 0])
    u_g, sigma_g, v_g = np.linalg.svd(a[:, :, 1])   #三通道分别进行SVD
    u_b, sigma_b, v_b = np.linalg.svd(a[:, :, 2])
    plt.figure(figsize=(10,10), facecolor='w')
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    for k in range(1, K+1):          #k是指用前几个奇异值
        print(k)
        R = restore1(sigma_r, u_r, v_r, k)   #做恢复
        G = restore1(sigma_g, u_g, v_g, k)
        B = restore1(sigma_b, u_b, v_b, k)
        I = np.stack((R, G, B), 2)      #三通道连在一起
        Image.fromarray(I).save('%s\\svd_%d.png' % (output_path, k))
        if k <= 30:
            plt.subplot(5,6 , k)
            plt.imshow(I)
            plt.axis('off')
            plt.title(u'sigam是%d' % k)
    plt.suptitle(u'SVD', fontsize=18)
    plt.tight_layout(2)
    plt.subplots_adjust(top=0.9)
    plt.show()
