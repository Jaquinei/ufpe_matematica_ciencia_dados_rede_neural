import os
import warnings
import logging

# 1. Configurações globais de ambiente para suprimir logs do TensorFlow/oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# 2. Imports das bibliotecas principais
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD


def sigmoid(x):
    """Função de ativação Sigmoide: sigma(x) = 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-x))


def get_dataset(n_samples=100, noise=0.1, random_state=42):
    """
    Gera o dataset sintético Two Moons (Duas Luas).
    Retorna:
        X (np.ndarray): Matriz de atributos de shape (n_samples, 2).
        Y (np.ndarray): Vetor de rótulos binários {0, 1} de shape (n_samples,).
    """
    np.random.seed(random_state)
    X, Y = datasets.make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, Y


def plot_dataset(X, Y, output_dir):
    """
    Gera e salva o gráfico de dispersão 2D das Duas Luas no diretório de saída especificado.
    """
    os.makedirs(output_dir, exist_ok=True)
    color = ['blue' if k == 0 else 'red' for k in Y]

    plt.figure(figsize=(8, 6), dpi=150)
    plt.scatter(X[:, 0], X[:, 1], c=color, edgecolors='k', s=50)
    plt.title("Dataset Duas Luas (Two Moons)", fontsize=13, fontweight='bold')
    plt.xlabel("$x_0$", fontsize=12)
    plt.ylabel("$x_1$", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)

    filepath_svg = os.path.join(output_dir, 'duas_luas.svg')
    plt.savefig(filepath_svg, bbox_inches='tight')
    plt.close()


def plot_decision_boundary(model, X, Y, w0, b0, title, output_dir, filename_prefix='fronteira_duas_luas'):
    """
    Renderiza a fronteira de decisão não-linear aprendida pelo modelo Keras
    junto com as retas lineares dos neurônios da camada oculta (v_j = 0).

    Parâmetros:
        model: Modelo Keras treinado.
        X, Y: Dados de entrada e rótulos.
        w0: Matriz de pesos da camada oculta (shape: [n_hidden, 2]).
        b0: Vetor de bias da camada oculta (shape: [n_hidden,]).
        title: Título do gráfico.
        output_dir: Diretório de destino.
        filename_prefix: Prefixo do nome dos arquivos gerados.
    """
    os.makedirs(output_dir, exist_ok=True)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid, verbose=0).reshape(xx.shape)

    plt.figure(figsize=(10, 7), dpi=150)
    # Superfície de decisão suave e linha de contorno no limiar 0.5
    plt.contourf(xx, yy, Z, levels=50, cmap='RdBu_r', alpha=0.3)
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2.5)

    x_vals = np.linspace(x_min, x_max, 200)

    # Cores e estilos para as retas de cada neurônio oculto
    line_colors = ['darkorange', 'purple', 'teal', 'magenta', 'green', 'brown']
    n_hidden = w0.shape[0]

    for j in range(n_hidden):
        w0_j0 = w0[j, 0]
        w0_j1 = w0[j, 1]
        b0_j = b0[j]

        # Evita divisão por zero caso o peso seja nulo
        if abs(w0_j1) > 1e-7:
            reta_j = -(w0_j0 * x_vals + b0_j) / w0_j1
            c = line_colors[j % len(line_colors)]
            plt.plot(x_vals, reta_j, '--', color=c, linewidth=2.2,
                     label=rf'Reta Neurônio {j+1} ($v_{j} = 0$)')

    # Pontos das classes
    plt.scatter(X[Y == 0, 0], X[Y == 0, 1], c='blue', edgecolors='k', s=60, label='Classe 0 (Lua Azul)')
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], c='red', edgecolors='k', s=60, label='Classe 1 (Lua Vermelha)')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.xlabel("$x_0$", fontsize=12)
    plt.ylabel("$x_1$", fontsize=12)
    plt.legend(loc='upper right', framealpha=0.9, fontsize=9.5)
    plt.grid(True, linestyle=':', alpha=0.6)

    filepath_png = os.path.join(output_dir, f'{filename_prefix}.png')
    filepath_svg = os.path.join(output_dir, f'{filename_prefix}.svg')
    plt.savefig(filepath_png, bbox_inches='tight')
    plt.savefig(filepath_svg, bbox_inches='tight')
    plt.close()
    print(f"[OK] Gráficos da fronteira salvos em '{output_dir}'.")
