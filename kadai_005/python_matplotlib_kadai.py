# -*- coding: utf-8 -*-
"""
Created on Sun May 19 16:09:23 2024

@author: ttoh1
"""

import numpy as np
import matplotlib as mlt
import matplotlib.image as mpimg
import japanize_matplotlib

fig, axes = plt.subplots(2, 3, tight_layout = True)

#折れ線グラフ
y0 = [1, 2, -5, 2]
axes[0,0].plot(y0)

#y1 = [1,2,5,1]
#axes[1].plot(y1)

#sin関数
x1 = np.linspace(0, 10, 100)
y1 = 2 + 2*np.sin(2*x1)

axes[0,1].set(xlim = (0, 10), xticks = np.arange(0, 10),
         ylim = (-1, 5), yticks = np. arange(-1, 6))
axes[0,1].plot(x1, y1, linewidth=2.0)

#ヒストグラム
y2 = np.random.normal(15, 5, 2000)
axes[0,2].hist(y2)

#散布図
x3 = 5 + np.random.normal(0, 2, 60)
y3 = 5 + np.random.normal(0, 2, len(x3))
axes[1,0].scatter(x3, y3)

#画像
y4 = mpimg.imread('sample/risu.jpg')
axes[1,1].imshow(y4)

#円グラフ
y5 = np.array([30, 60, 10])
label = ["いぬ", "ネコ", "ねずみ"]
axes[1,2].set_title("好きな動物")
axes[1,2].pie(y5, labels=label)