import numpy as np
import matplotlib.pyplot as plt


# 重みの初期設定のための関数
def rand_weights(layer_in, layer_out):
    # 分散が0だと小さすぎ,1だと大きすぎるのため0.5と設定する
    W = np.random.rand(layer_out, 1 + layer_in) * 0.5
    return W


# 二値分類問題のため活性化関数をロジスティック関数に設定
def logistic_function(x):
    return 1.0 / (1.0 + np.exp(-x))


# ロジスティック関数の導関数
def logistic_derivative_function_(x):
    return logistic_function(x) * (1 - logistic_function(x))


# netural_networkの計算
def processing(w_1, w_2, X, d_n):
    # 訓練データの数が4
    N = 4
    # 誤差逆転播法を使うためのN*partial w
    N_partial_w_1 = np.zeros(w_1.shape)
    N_partial_w_2 = np.zeros(w_2.shape)
    # 出力の配列の指定
    y = np.zeros((N, 1))
    for t in range(N):
        # 入力の列ベクトル,3*1.但しnp.array([[1]])はバイアスを表す.
        z_1 = np.vstack((np.array([[1]]), X[t:t + 1, :].T))
        # 2*1
        u_2 = np.dot(w_1, z_1)
        # 3*1
        z_2 = np.vstack((np.array([[1]]), logistic_function(u_2)))
        # 1*1
        u_3 = np.dot(w_2, z_2)
        # 出力のベクトル.1*1
        z_3 = logistic_function(u_3)
        # 得られた結果をベクトルにする.
        y[t, 0] = z_3
        # 誤差逆転播法を用いるδとN倍のpartial Wを計算
        delta_3 = z_3 - d_n[t, 0]
        delta_2 = np.multiply(np.dot(w_2[:, 1:].T, delta_3), logistic_derivative_function_(u_2))
        N_partial_w_2 = N_partial_w_2 + np.dot(delta_3, z_2.T)
        N_partial_w_1 = N_partial_w_1 + np.dot(delta_2, z_1.T)
    # Nに割り算してpartial Wを得る(平均値)
    partial_w_1 = (1.0 / N) * N_partial_w_1
    partial_w_2 = (1.0 / N) * N_partial_w_2
    # 誤差関数の計算
    E = (1.0 / N) * np.sum(-d_n * np.log(y) - (np.array([[1]]) - d_n) * np.log(1 - y))
    return {'partial_w_1': partial_w_1, 'partial_w_2': partial_w_2, 'E': E, 'y': y}


# XORを実現するためにネットワークの各層のユニット数を設定
input_layer_size = 2
hidden_layer_size = 2
output_layer_size = 1

# 学習係数
epsilon = 0.5

# 訓練データを設定
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
d_n = np.array([[0], [1], [1], [0]])


# 重みの初期設定
w_1 = rand_weights(input_layer_size, hidden_layer_size)
w_2 = rand_weights(hidden_layer_size, output_layer_size)


# 訓練回数の設定
iter_times = 10000


# 誤差関数と出力を用いて評価する
result = {'E': [], 'y': []}

for i in range(iter_times):
    # 結果を獲得
    processing_result = processing(w_1, w_2, X=X, d_n=d_n)
    Delta_w_1 = processing_result.get('partial_w_1')
    Delta_w_2 = processing_result.get('partial_w_2')
    E = processing_result.get('E')
    y = processing_result.get('y')
    # 重みの更新
    w_1 -= epsilon * Delta_w_1
    w_2 -= epsilon * Delta_w_2
    # 獲得したデータの記録
    result['E'].append(E)
    result['y'].append(y)
    # 重みの初期値と最終値を表示
    if i == 0 :
        print('w_1の初期値', w_1 )
        print('\n')
        print('w_2の初期値', w_2)
        print('\n')
    if i == (iter_times - 1):
        print('w_1の最終値', w_1)
        print('\n')
        print('w_2の最終値', w_2)
        print('\n')



# 誤差関数と更新回数を図で示す
plt.plot(result.get('E'))
plt.show()

# 最初と最後の出力を表す
print('最初の出力',result.get('y')[0])
print('\n')
print('最後の出力',result.get('y')[-1])
