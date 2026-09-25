import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suprime logs de INFO e WARNING do C++

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD
from keras import initializers

os.makedirs('graphics', exist_ok=True)

X, Y = datasets.make_moons(100, noise=0.1)

color = ['blue' if k == 0 else 'red' for k in Y]

plt.scatter(X[:, 0], X[:, 1], c=color)
plt.savefig('graphics/duas_luas.svg')
plt.close()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def run_neural_net(x, w0, b0, b1, w1):
    s00 = w0[0, 0] * x[0]
    s01 = w0[0, 1] * x[1]
    s02 = s00 + s01
    v0 = s02 + b0[0]
    y0 = sigmoid(v0)

    s10 = w0[1, 0] * x[0]
    s11 = w0[1, 1] * x[1]
    s12 = s10 + s11
    v1 = s12 + b0[1]
    y1 = sigmoid(v1)

    s20 = y0 * w1[0]
    s21 = y1 * w1[1]
    s22 = s20 + s21
    v2 = s22 + b1[0]
    y2 = sigmoid(v2)
    return 1 if y2 > 0.5 else 0


def neural_net(x, d, w0, b0, b1, w1):
    # forward

    s00 = w0[0, 0] * x[0]
    s01 = w0[0, 1] * x[1]
    s02 = s00 + s01
    v0 = s02 + b0[0]
    y0 = sigmoid(v0)

    s10 = w0[1, 0] * x[0]
    s11 = w0[1, 1] * x[1]
    s12 = s10 + s11
    v1 = s12 + b0[1]
    y1 = sigmoid(v1)

    s20 = y0 * w1[0]
    s21 = y1 * w1[1]
    s22 = s20 + s21
    v2 = s22 + b1[0]
    y2 = sigmoid(v2)
    e = y2 - d
    L = 1/2 * (e ** 2)


    # backward
    grad_w0 = np.zeros(w0.shape)
    grad_w1 = np.zeros(w1.shape)
    grad_b0 = np.zeros(b0.shape)
    grad_b1 = np.zeros(b1.shape)

    grad_L = 1
    grad_e = grad_L * e

    grad_y2 = grad_e
    grad_v2 = grad_y2 * y2 * (1 - y2)
    grad_b1[0] = grad_v2
    grad_s22 = grad_v2
    grad_s21 = grad_s22
    grad_s20 = grad_s22
    grad_w1[1] = grad_s21 * y1
    grad_y1 = grad_s21 * w1[1]

    grad_w1[0] = grad_s20 * y0
    grad_y0 = grad_v2 * w1[0]

    grad_v0 = grad_y0 * y0 * (1 - y0)
    grad_v1 = grad_y1 * y1 * (1 - y1)

    grad_b0[0] = grad_v0
    grad_b0[1] = grad_v1
    grad_s12 = grad_v1
    grad_s02 = grad_v0

    grad_s00 = grad_s02
    grad_s01 = grad_s02
    grad_s10 = grad_s12
    grad_s11 = grad_s12
    grad_w0[0, 0] = grad_s00 * x[0]
    grad_w0[0, 1] = grad_s01 * x[1]
    grad_w0[1, 0] = grad_s10 * x[0]
    grad_w0[1, 1] = grad_s11 * x[1]
    return grad_w0, grad_b0, grad_w1, grad_b1, L

def main():
    np.random.seed(42)

    # inicialização aleatória
    w0 = np.random.rand(2, 2)
    w1 = np.random.rand(2)
    b0 = np.random.rand(2)
    b1 = np.random.rand(1)

    # Guarda pesos iniciais para usar exatamente os mesmos no Keras
    w0_init = w0.copy()
    w1_init = w1.copy()
    b0_init = b0.copy()
    b1_init = b1.copy()

    # taxa de aprendizado
    taxa = 0.1

    print("=================================================================")
    print(" 1. Rede Manual (BackPropagation MANUAL)")
    print("=================================================================")

    acc = 0
    for i in range(100):
        out = run_neural_net(X[i], w0, b0, b1, w1)
        if out == Y[i]:
            acc += 1
    print(f"Acurácia inicial: {acc}%")

    # gradiente descendente manual
    for i in range(10000):
        loss = 0

        grad_w0 = np.zeros(w0.shape)
        grad_w1 = np.zeros(w1.shape)
        grad_b0 = np.zeros(b0.shape)
        grad_b1 = np.zeros(b1.shape)

        for k in range(100):
            g_w0, g_b0, g_w1, g_b1, L = neural_net(X[k], Y[k], w0, b0, b1, w1)

            grad_w0 += g_w0
            grad_w1 += g_w1
            grad_b0 += g_b0
            grad_b1 += g_b1
            loss += L

        w0 -= taxa * grad_w0
        w1 -= taxa * grad_w1
        b0 -= taxa * grad_b0
        b1 -= taxa * grad_b1

        if i % 2000 == 0 or i == 9999:
            print(f"Época {i:5d} | Loss: {loss:.6f}")

    # acurácia após o treinamento manual
    acc_manual = 0
    for i in range(100):
        out = run_neural_net(X[i],  w0, b0, b1, w1)
        if out == Y[i]:
            acc_manual += 1
    print(f"Acurácia final (Manual): {acc_manual}%")

    print("\n=================================================================")
    print(" 2. Rede Keras (Arquitetura equivalente e pesos iniciais iguais)")
    print("=================================================================")

    # Rede Keras exatamente equivalente: 2 entradas -> 2 ocultos (sigmoid) -> 1 saída (sigmoid)
    model = Sequential([
        Input(shape=(2,)),
        Dense(2, activation='sigmoid'),
        Dense(1, activation='sigmoid')
    ])

    # Inicializa o Keras com exatamente os mesmos pesos iniciais da rede manual
    model.layers[0].set_weights([w0_init.T, b0_init])
    model.layers[1].set_weights([w1_init.reshape(2, 1), b1_init])

    # O SGD no Keras divide por N (100) e o MSE tem fator 2 em relação ao half-MSE (1/2 e^2)
    # Logo, learning_rate = taxa * 50 garante o mesmo tamanho de passo de gradiente exato
    opt = SGD(learning_rate=taxa * 50)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    model.fit(X, Y, epochs=10000, batch_size=100, verbose=False)

    loss_k, acc_k = model.evaluate(X, Y, verbose=0)
    print(f"Acurácia final (Keras) : {acc_k * 100:.0f}%")
    print(f"Loss final (Keras MSE) : {loss_k:.6f}  (equiv. sum Loss manual: {loss_k * 50:.6f})")

    # Extrai os pesos aprendidos pelo Keras
    w_k0, b_k0 = model.layers[0].get_weights()
    w_k1, b_k1 = model.layers[1].get_weights()

    print("\n=================================================================")
    print(" 3. Comparação dos Pesos e Biases entre Rede Manual e Keras")
    print("=================================================================")
    print("--- Pesos Camada Oculta (w0) ---")
    print("Manual w0:\n", np.round(w0, 4))
    print("Keras  w0:\n", np.round(w_k0.T, 4))

    print("\n--- Biases Camada Oculta (b0) ---")
    print("Manual b0:", np.round(b0, 4))
    print("Keras  b0:", np.round(b_k0, 4))

    print("\n--- Pesos Camada Saída (w1) ---")
    print("Manual w1:", np.round(w1, 4))
    print("Keras  w1:", np.round(w_k1.flatten(), 4))

    print("\n--- Bias Camada Saída (b1) ---")
    print("Manual b1:", np.round(b1, 4))
    print("Keras  b1:", np.round(b_k1.flatten(), 4))
    print("=================================================================")

    # 4. Gera e salva o gráfico com os dados, as 2 retas e a fronteira não-linear
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    grid = np.c_[xx.ravel(), yy.ravel()]
    # Avalia a rede no grid (pode usar tanto a função manual quanto o Keras)
    Z = model.predict(grid, verbose=0).reshape(xx.shape)

    plt.figure(figsize=(10, 7), dpi=150)
    plt.contourf(xx, yy, Z, levels=50, cmap='RdBu_r', alpha=0.3)
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2.5)

    # Retas dos neurônios da camada oculta (onde v_0 = 0 e v_1 = 0)
    x_vals = np.linspace(x_min, x_max, 200)
    reta0 = -(w0[0, 0] * x_vals + b0[0]) / w0[0, 1]
    reta1 = -(w0[1, 0] * x_vals + b0[1]) / w0[1, 1]

    plt.plot(x_vals, reta0, '--', color='darkorange', linewidth=2.5, label=r'Reta Neurônio 1 ($v_0 = 0$)')
    plt.plot(x_vals, reta1, '--', color='purple', linewidth=2.5, label=r'Reta Neurônio 2 ($v_1 = 0$)')

    plt.scatter(X[Y == 0, 0], X[Y == 0, 1], c='blue', edgecolors='k', s=60, label='Classe 0 (Lua Azul)')
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], c='red', edgecolors='k', s=60, label='Classe 1 (Lua Vermelha)')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title("Classificação das Duas Luas: Retas Ocultas e Fronteira Não-Linear Final", fontsize=13, fontweight='bold')
    plt.xlabel("$x_0$", fontsize=12)
    plt.ylabel("$x_1$", fontsize=12)
    plt.legend(loc='upper right', framealpha=0.9, fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.savefig('graphics/fronteira_duas_luas.png', bbox_inches='tight')
    plt.savefig('graphics/fronteira_duas_luas.svg', bbox_inches='tight')
    plt.close()
    print("\n[OK] Gráficos salvos com sucesso na pasta 'graphics/'.")


main()