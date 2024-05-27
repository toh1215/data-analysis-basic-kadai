# -*- coding: utf-8 -*-
"""
Created on Sun May 19 09:21:05 2024

@author: ttoh1
"""
#%%
######################## numpy #########################
#%%
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

six_reshape = np.arange(6).reshape(2, 3)
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

#%%
######################## matplotlib #########################
#%%

#%%
#折れ線グラフ
import matplotlib.pyplot as plt

x = [0, 1, 2, 3]
y = [1, 2, -5, 2]
plt.plot(x, y)

#%%
#sin関数の折れ線グラフ
import matplotlib.pyplot as plt
import numpy as np

#データを作成
#0から10までの幅で、配列を等間隔に100個作成する。
x = np.linspace(0, 10, 30)
#切片2で、2sin2x。
y = 2 + 2 * np.sin(2 * x)

#プロット
fig, ax = plt.subplots()

ax.plot(x, y, linewidth=2.0)

ax.set(xlim=(0, 10), xticks=np.arange(0, 10),
       ylim=(-1, 5), yticks=np.arange(-1, 6))

plt.show()

#%%
# 画像読み込みライブラリをインポート
import matplotlib.image as mpimg
# 画像表示ライブラリをインポート
import matplotlib.pyplot as plt

#画像を読み込む
img = mpimg.imread('./sample/risu.jpg')
#画像を表示
plt.imshow(img)


#%%
#画像読み込み、カラー変更（グレーアウト）
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

img = mpimg.imread('sample/risu.jpg')

#RGBのRの値を使って疑似カラーを付与
img = img[:, :, 0]
plt.imshow(img, cmap='gray')
#plt.colorbar()

#%%
#ヒストグラム
import numpy as np
import matplotlib.pyplot as plt

x = np.random.normal(15, 5, 2000)
plt.xlim(-10,40)

plt.hist(x, bins=30)

#%%
#箱ひげ図
import matplotlib.pyplot as plt
import japanize_matplotlib

# モーターから観測した音の周波数
hertz = [300, 200, 200, 500, 600, 2000, 1600, 1800, 1850, 1700, 1500, 1500, 1500, 1400, 5000, 6000, 2000, 3000, 3000]
# 箱ひげ図
plt.boxplot(hertz)
plt.title('モーター音の周波数')
#描画
plt.show()

#%%
#散布図
import matplotlib.pyplot as plt
import numpy as np

# データを生成
np.random.seed(3)
x = 4 + np.random.normal(0, 2, 60)
y = 4 + np.random.normal(0, 2, len(x))

plt.scatter(x, y)
plt.show()

#%%
#円グラフ
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

x = np.array([30, 40, 30])
label = ["いぬ", "ネコ", "その他"]
plt.title("好きな動物")
plt.pie(x, labels=label)

#%%
#棒グラフ
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

x =np.array([1, 2, 3, 4, 5])
y = np.array([100, 120, 150, 200, 160])
label = ["昆布", "うめ", "鮭", "カルビ", "すじこ"]
plt.title("おにぎりの具ごとの値段")
plt.bar(x, y, tick_label=label)

#%%
########################pandas#########################
#%%
import pandas as pd

#CSVを読み込む。データフレームとして呼び出される。
df = pd.read_csv('sample/train.csv')

#先頭5行を表示
df.head()

#データフレーム（表形式）として表示
df.loc[[2]]

#シリーズ（1次元配列）として表示
df.loc[2]

#POSTED_BY項目がOwnerの行だけ取り出す。
df.query('POSTED_BY == "Owner"')

#dfのTARGET(PRICE_IN_LACS)が最大の値の行を取り出す
df.loc[[df['TARGET(PRICE_IN_LACS)'].idxmax()]]

#最大値
df['TARGET(PRICE_IN_LACS)'].max()

#最小値
df['TARGET(PRICE_IN_LACS)'].min()

#物件価格の平均値
df['TARGET(PRICE_IN_LACS)'].mean()

#まとめて求める
df['TARGET(PRICE_IN_LACS)'].describe()

#%%
############# pandas 実践################
#%%
import pandas as pd

#CSVファイルをデータフレームとして読み込み
df = pd.read_csv("sample/sample_pandas_6.csv")

#先頭から5行目を表示
df.head()

df.query('商品番号 == "Z4WOOIYV"')

df.query('単価 == 600')

df.query('在庫 <= 5')

df.query('商品番号 == "8T7D5DQA" and 在庫 <= 5')

def tax(x):
    return x * 1.10

df['単価'].apply(tax)

type(df['発注日'].loc[0])

df['発注日'] = pd.to_datetime(df['発注日'])

type(pd.to_datetime(df['発注日']).loc[0])

df.agg({'発注日':['max', 'min']})

tax_series = df['単価'].apply(tax)
tax_series.name = "単価（税込み)"
pd.concat([df, tax_series], axis=1)

category_df = pd.read_csv('sample/category.csv')
category_df

df = pd.merge(df, category_df[['商品番号', 'カテゴリー']], how='inner', on='商品番号')
df

#%%
############# scikit-learn　################
#%%
from sklearn.datasets import load_wine

dataset = load_wine()
dataset.data
dataset.feature_names

#%%データの取得
import pandas as pd
df = pd.DataFrame(data = dataset.data, columns = dataset.feature_names)
df.head()

dataset.target

df['category'] = dataset.target
df.head()

df.shape

#%%　サンプルデータの分割
X = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split

train_test_split(X, y, test_size=0.3, random_state=5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

print(X.shape, X_train.shape, X_test.shape, y.shape, y_train.shape, y_test.shape)

#%% 予測モデルのインスタンス化
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=3)

#%% 予測モデルの学習
model.fit(X_train, y_train)

#%% 予測モデルの評価
y_pred = model.predict(X_test)
y_pred
y_test

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
#accuracy_score(y_pred, y_test)

model.score(X_test, y_test)

#%% 予測
import numpy as np

X_real = np.array([
    [13, 1.6, 2.2, 16, 118, 2.6, 2.9, 0.21, 1.6, 5.8, 0.92, 3.2, 1011],
    [12, 2.8, 2.2, 18, 100, 2.5, 2.3, 0.25, 2.0, 2.2, 1.15, 3.3, 1000],
    [14, 4.1, 2.7, 24, 101, 1.6, 0.7, 0.53, 1.4, 9.4, 0.61, 1.6, 560]])

model.predict(X_real)

































































