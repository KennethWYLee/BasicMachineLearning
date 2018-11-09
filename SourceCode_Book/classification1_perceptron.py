import numpy as np
import matplotlib.pyplot as plt

# 学習データを読み込む
train = np.loadtxt('images1.csv', delimiter=',', skiprows=1)
train_x = train[:,0:2]
train_y = train[:,2]

# 重みの初期化
w = np.random.rand(2)

# 識別関数
def f(x):
    if np.dot(w, x) >= 0:
        return 1
    else:
        return -1

# 繰り返し回数
epoch = 10

# 更新回数
count = 0

# 重みを学習する
for _ in range(epoch):
    for x, y in zip(train_x, train_y):
        if f(x) != y:
            w = w + y * x

            # ログの出力
            count += 1
            print('{}回目: w = {}'.format(count, w))

# プロットして確認
x1 = np.arange(0, 500)
plt.plot(train_x[train_y ==  1, 0], train_x[train_y ==  1, 1], 'o')
plt.plot(train_x[train_y == -1, 0], train_x[train_y == -1, 1], 'x')
plt.plot(x1, -w[0] / w[1] * x1, linestyle='dashed')
plt.show()
