# -*- coding: utf-8 -*-
"""
Created on Mon May 20 06:36:50 2024

@author: ttoh1
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import japanize_matplotlib


df = pd.read_csv("sample/sample_pandas_6.csv")
category_df = pd.read_csv("sample/category.csv")

df = pd.merge(df, category_df[['商品番号', 'カテゴリー']], how='inner', on='商品番号')
df

bento_counts = df['カテゴリー'].value_counts()
bento_counts.plot(kind='bar')
plt.xlabel('カテゴリー')
plt.ylabel('頻度')
plt.title('カテゴリー別の出現頻度')
plt.show()



df.groupby('商品番号')［'注文数'］.sum()
df.groupby('商品番号')［'注文数'］.describe()