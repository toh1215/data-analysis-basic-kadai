# -*- coding: utf-8 -*-
"""
Created on Sun May 19 09:21:05 2024

@author: ttoh1
"""
#%%
import numpy as np
narray = np.array([1,2,3])
narray
narray.size
type(narray)

nones = np.ones(10)
nones

type(nones)

multi_array = np.array([[0,1,2],[3,4,5],[6,7,8]])
multi_array

np.random.rand(3,3)
np.random.randint(1,10)
random_multi_array = np.random.randint(1, 10, (5,5))
random_multi_array
random_multi_array.max()
random_multi_array.min()

random_array = np.random.randint(1, 10, (10,))
random_array
random_array[0:3]

a = np.array([1,2,3])
b = np.array([4,5,6])
np.concatenate((a,b))

c = np.array([[1,2,3]])
d = np.array([[4,5,6]])
np.concatenate((c,d), axis=0)

three = np.ones(3)
three
three + 3

six_reshape = np.arange(6).reshape(2,3)
six_reshape
six_reshape + 1
six_reshape + np.array([[1, 0, 2],[1, 0, 2]])

five_full = np.full(5, 5)
five_full - 2
five_full - np.array([1, 2, 3, 4, 5])

ten = np.arange(10)
ten
ten * 3
ten * np.array([1, 2, 2, 2, 3, 3, 3, 4, 4, 5])

div_six = np.arange(6)
div_six
div_six / 2

A = np.array([[4, 7, 2], [1, 2, 1]])
B = np.array([[2, 2, 2], [4, 5, 2], [9, 2, 1]])
np.dot(A, B)

#%%
# 逆行列を求めるinvメソッドをインポート
from numpy.linalg import inv
A = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
ainv = inv(A)
ainv
np.dot(A, ainv)

#%%
a = np.array([1,2,3])
np.sum(a)

a = np.array([[14, 8, 11, 10], [7, 9, 10, 11], [10, 15, 5, 10]])
np.sum(a)
np.sum(a, axis=0)
np.sum(a, axis=1)
#np.sum(a, axis=2)

a = np.array([1,2,3,4])
np.median(a)

a = np.array([1, 2, 3, 4, 5])
np.std(a)

#%%
from PIL import Image
import numpy as np

im = Image.open(r"C:\Users\ttoh1\iCloudDrive\work\15.データサイエンス\04.自己学習\侍エンジニア_教材\python\sample\camera.jpg")
im = im.resize((im.width //2, im.height //2))
im

# PIL形式からNumPy形式に変換
im_np = np.asarray(im)
im_np

#　ネガティブ画像へ変換する（ブロードキャスが機能する）
negative_im_np = 255 -im_np
negative_im = Image.fromarray(negative_im_np)
negative_im





