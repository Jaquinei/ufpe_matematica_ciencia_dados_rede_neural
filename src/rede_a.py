import os
import sys

# Garante que o diretório src esteja no path para importação
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    np, plt, Sequential, Dense, Input, SGD,
    sigmoid, get_dataset, plot_dataset, plot_decision_boundary
)


def run_neural_net(x, w0, b0, b1, w1):
    """
    Executa a inferência (Forward Pass) na Rede A (2 -> 2 -> 1).
    Retorna 1 se y2 >= 0.5, senão 0.
    """
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
    return 1 if y2 >= 0.5 else 0


def neural_net(x, d, w0, b0, b1, w1):
    """
    Executa o Forward Pass e deriva analiticamente todos os gradientes via Regra da Cadeia (Backward Pass)
    para um único exemplo de treino (x, d) na arquitetura 2 -> 2 -> 1.
    """
    # ==================== FORWARD PASS ====================
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

    # ==================== BACKWARD PASS ====================
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


def plot_neural_network_architecture(output_dir):
    """Gera o diagrama visual da arquitetura da Rede A (2 -> 2 -> 1)."""
    import matplotlib.patches as patches

    os.makedirs(output_dir, exist_ok=True)
    filename_png = os.path.join(output_dir, 'arquitetura_rede.png')
    filename_svg = os.path.join(output_dir, 'arquitetura_rede.svg')

    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=150)
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    input_coords = [(0.15, 0.70), (0.15, 0.35)]
    hidden_coords = [(0.50, 0.70), (0.50, 0.35)]
    output_coords = [(0.85, 0.525)]

    radius = 0.055

    weights_w0_labels = [
        [r'$w_{0[0,0]}$', r'$w_{0[0,1]}$'],
        [r'$w_{0[1,0]}$', r'$w_{0[1,1]}$']
    ]

    for i, (xi, yi) in enumerate(input_coords):
        for j, (xh, yh) in enumerate(hidden_coords):
            ax.annotate('', xy=(xh - radius, yh), xytext=(xi + radius, yi),
                        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.8, mutation_scale=15))
            t_pos = 0.35 if (i == j) else 0.25
            xt = xi + t_pos * (xh - xi)
            yt = yi + t_pos * (yh - yi) + (0.035 if i == 0 else -0.035)
            ax.text(xt, yt, weights_w0_labels[j][i], fontsize=11, color="#2c3e50",
                    ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.85))

    weights_w1_labels = [r'$w_{1[0]}$', r'$w_{1[1]}$']
    for j, (xh, yh) in enumerate(hidden_coords):
        xo, yo = output_coords[0]
        ax.annotate('', xy=(xo - radius, yo), xytext=(xh + radius, yh),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.8, mutation_scale=15))
        xt = xh + 0.35 * (xo - xh)
        yt = yh + 0.35 * (yo - yh) + (0.035 if j == 0 else -0.035)
        ax.text(xt, yt, weights_w1_labels[j], fontsize=11, color="#2c3e50",
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.85))

    xo, yo = output_coords[0]
    ax.annotate('', xy=(0.98, yo), xytext=(xo + radius, yo),
                arrowprops=dict(arrowstyle="->", color="#27ae60", lw=2.2, mutation_scale=18))
    ax.text(0.99, yo + 0.05, r'$\hat{y} \in \{0, 1\}$', fontsize=12, fontweight='bold', color='#27ae60', ha='left')
    ax.text(0.99, yo - 0.05, r'($y_2 \geq 0.5$)', fontsize=10, color='#555555', ha='left')

    for i, (x, y) in enumerate(input_coords):
        circle = patches.Circle((x, y), radius, facecolor='#3498db', edgecolor='#1d6fa5', lw=2, zorder=4)
        ax.add_patch(circle)
        ax.text(x, y, f'$x_{i}$', fontsize=14, fontweight='bold', color='white', ha='center', va='center', zorder=5)

    hidden_labels = [
        (r'$v_0$', r'$y_0 = \sigma(v_0)$', r'$b_{0[0]}$'),
        (r'$v_1$', r'$y_1 = \sigma(v_1)$', r'$b_{0[1]}$')
    ]
    for j, (x, y) in enumerate(hidden_coords):
        circle = patches.Circle((x, y), radius, facecolor='#9b59b6', edgecolor='#6c3483', lw=2, zorder=4)
        ax.add_patch(circle)
        ax.text(x, y, f'$h_{j}$', fontsize=14, fontweight='bold', color='white', ha='center', va='center', zorder=5)
        bias_y = y + 0.12 if j == 0 else y - 0.12
        ax.annotate('', xy=(x, y + (radius if j == 0 else -radius)), xytext=(x, bias_y),
                    arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.8, mutation_scale=14))
        ax.text(x, bias_y + (0.025 if j == 0 else -0.04), f'Bias {hidden_labels[j][2]}', fontsize=10,
                fontweight='bold', color='#d35400', ha='center')
        ax.text(x, y - (0.09 if j == 0 else -0.09), hidden_labels[j][1], fontsize=10,
                color='#6c3483', ha='center', fontweight='bold',
                bbox=dict(boxstyle='square,pad=0.2', facecolor='#f4ecf7', edgecolor='#d2b4de', lw=1))

    circle = patches.Circle((xo, yo), radius, facecolor='#2ecc71', edgecolor='#1e8449', lw=2, zorder=4)
    ax.add_patch(circle)
    ax.text(xo, yo, '$y_2$', fontsize=13, fontweight='bold', color='white', ha='center', va='center', zorder=5)
    ax.annotate('', xy=(xo, yo + radius), xytext=(xo, yo + 0.12),
                arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.8, mutation_scale=14))
    ax.text(xo, yo + 0.145, r'Bias $b_{1[0]}$', fontsize=10, fontweight='bold', color='#d35400', ha='center')
    ax.text(xo, yo - 0.09, r'$y_2 = \sigma(v_2)$', fontsize=10, color='#1e8449', fontweight='bold',
            bbox=dict(boxstyle='square,pad=0.2', facecolor='#eafaf1', edgecolor='#a9dfbf', lw=1))

    ax.text(0.15, 0.92, 'Camada de Entrada\n(2 Variáveis)', fontsize=12, fontweight='bold', ha='center', va='center', color='#2c3e50')
    ax.text(0.50, 0.92, 'Camada Oculta\n(2 Neurônios Sigmoide)', fontsize=12, fontweight='bold', ha='center', va='center', color='#2c3e50')
    ax.text(0.85, 0.92, 'Camada de Saída\n(1 Neurônio Sigmoide)', fontsize=12, fontweight='bold', ha='center', va='center', color='#2c3e50')

    formula_text = (
        r"Equações:  $v_0 = w_{0[0,0]}x_0 + w_{0[0,1]}x_1 + b_{0[0]}, \ y_0 = \sigma(v_0)$   |   "
        r"$v_1 = w_{0[1,0]}x_0 + w_{0[1,1]}x_1 + b_{0[1]}, \ y_1 = \sigma(v_1)$   |   "
        r"$v_2 = w_{1[0]}y_0 + w_{1[1]}y_1 + b_{1[0]}, \ y_2 = \sigma(v_2)$" + "\n"
        r"Ativação: $\sigma(v) = \frac{1}{1 + e^{-v}}$          "
        r"Função de Perda: $L = \frac{1}{2}(y_2 - d)^2$"
    )
    fig.text(0.5, 0.05, formula_text, fontsize=9.5, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', edgecolor='#bdc3c7', lw=1.2))

    plt.suptitle("Arquitetura da Rede A ($2 \\rightarrow 2 \\rightarrow 1$)", fontsize=15, fontweight='bold', y=0.98)
    plt.savefig(filename_png, bbox_inches='tight')
    plt.savefig(filename_svg, bbox_inches='tight')
    plt.close()
    print(f"[OK] Gráfico da arquitetura salvo em '{output_dir}'.")


def plot_computational_graph(output_dir):
    """Gera o grafo computacional detalhado (Forward em azul/preto, Backward em vermelho) da Rede A."""
    import matplotlib.patches as patches

    os.makedirs(output_dir, exist_ok=True)
    filename_png = os.path.join(output_dir, 'grafo_computacional.png')
    filename_svg = os.path.join(output_dir, 'grafo_computacional.svg')

    fig, ax = plt.subplots(figsize=(24, 10), dpi=200)
    ax.axis('off')
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 10)

    c_edge = '#2980b9'
    c_grad = '#c0392b'
    c_forward = '#2c3e50'
    c_op_bg = '#ffffff'
    c_op_border = '#34495e'
    c_act_bg = '#fcf3cf'
    c_act_border = '#f39c12'

    r_circle = 0.32

    def draw_circle_op(x, y, label):
        circle = patches.Circle((x, y), r_circle, facecolor=c_op_bg, edgecolor=c_op_border, lw=1.6, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, label, fontsize=14, fontweight='bold', color=c_op_border, ha='center', va='center', zorder=6)

    def draw_rect_act(x, y, label=r'$\sigma$ (Sigmoid)'):
        w, h = 1.3, 0.65
        rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle='round,pad=0.08',
                                      facecolor=c_act_bg, edgecolor=c_act_border, lw=1.6, zorder=5)
        ax.add_patch(rect)
        ax.text(x, y, label, fontsize=10, fontweight='bold', color='#7d6608', ha='center', va='center', zorder=6)

    def draw_arrow(x1, y1, x2, y2, label_fwd='', label_grad='', fwd_offset=(0, 0.16), grad_offset=(0, -0.22),
                   fwd_ha='center', grad_ha='center', curve=0):
        rad_str = f"arc3,rad={curve}" if curve != 0 else "arc3,rad=0"
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=c_edge, lw=1.5, mutation_scale=12,
                                    connectionstyle=rad_str), zorder=3)
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2
        if curve != 0:
            ym += curve * 0.8

        if label_fwd:
            ax.text(xm + fwd_offset[0], ym + fwd_offset[1], label_fwd, fontsize=9.5, fontweight='bold',
                    color=c_forward, ha=fwd_ha, va='center',
                    bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none', alpha=0.85))
        if label_grad:
            ax.text(xm + grad_offset[0], ym + grad_offset[1], label_grad, fontsize=8.5, fontweight='bold',
                    color=c_grad, ha=grad_ha, va='center',
                    bbox=dict(boxstyle='square,pad=0.1', facecolor='#fdedec', edgecolor='none', alpha=0.85))

    # Blocos de Operação
    draw_circle_op(3.0, 8.5, r'$\times$')   # mul00
    draw_circle_op(3.0, 6.5, r'$\times$')   # mul01
    draw_circle_op(4.5, 7.5, '+')           # add00
    draw_circle_op(6.0, 7.5, '+')           # add01
    draw_rect_act(8.0, 7.5, r'$\sigma(v_0)$')

    draw_circle_op(3.0, 4.0, r'$\times$')   # mul10
    draw_circle_op(3.0, 2.0, r'$\times$')   # mul11
    draw_circle_op(4.5, 3.0, '+')           # add10
    draw_circle_op(6.0, 3.0, '+')           # add11
    draw_rect_act(8.0, 3.0, r'$\sigma(v_1)$')

    draw_circle_op(11.0, 6.5, r'$\times$')  # mul20
    draw_circle_op(11.0, 4.0, r'$\times$')  # mul21
    draw_circle_op(13.0, 5.25, '+')         # add20
    draw_circle_op(14.8, 5.25, '+')         # add21
    draw_rect_act(17.0, 5.25, r'$\sigma(v_2)$')

    draw_circle_op(19.2, 5.25, '-')         # sub (e = y2 - d)
    draw_circle_op(21.2, 5.25, r'$(\cdot)^2$')  # sq
    draw_circle_op(23.0, 5.25, r'$\times$') # scale 1/2

    # Conexões Neurônio 0
    draw_arrow(0.8, 8.5, 3.0 - r_circle, 8.5, r'$w_{0[0,0]}$', r'$\frac{\partial L}{\partial w_{0[0,0]}}$')
    draw_arrow(0.8, 7.8, 3.0, 8.5 - r_circle, r'$x_0$', r'$\frac{\partial L}{\partial x_0}$')
    draw_arrow(0.8, 6.5, 3.0 - r_circle, 6.5, r'$w_{0[0,1]}$', r'$\frac{\partial L}{\partial w_{0[0,1]}}$')
    draw_arrow(0.8, 5.8, 3.0, 6.5 - r_circle, r'$x_1$', r'$\frac{\partial L}{\partial x_1}$')

    draw_arrow(3.0 + r_circle, 8.5, 4.5 - r_circle, 7.5 + 0.15, r'$s_{00}$', r'$\frac{\partial L}{\partial s_{00}}$')
    draw_arrow(3.0 + r_circle, 6.5, 4.5 - r_circle, 7.5 - 0.15, r'$s_{01}$', r'$\frac{\partial L}{\partial s_{01}}$')
    draw_arrow(4.5 + r_circle, 7.5, 6.0 - r_circle, 7.5, r'$s_{02}$', r'$\frac{\partial L}{\partial s_{02}}$')
    draw_arrow(6.0, 9.0, 6.0, 7.5 + r_circle, r'$b_{0[0]}$', r'$\frac{\partial L}{\partial b_{0[0]}}$', fwd_offset=(0.3, 0), grad_offset=(-0.35, 0))
    draw_arrow(6.0 + r_circle, 7.5, 8.0 - 0.65, 7.5, r'$v_0$', r'$\frac{\partial L}{\partial v_0}$')

    # Conexões Neurônio 1
    draw_arrow(0.8, 4.0, 3.0 - r_circle, 4.0, r'$w_{0[1,0]}$', r'$\frac{\partial L}{\partial w_{0[1,0]}}$')
    draw_arrow(0.8, 3.3, 3.0, 4.0 - r_circle, r'$x_0$', '')
    draw_arrow(0.8, 2.0, 3.0 - r_circle, 2.0, r'$w_{0[1,1]}$', r'$\frac{\partial L}{\partial w_{0[1,1]}}$')
    draw_arrow(0.8, 1.3, 3.0, 2.0 - r_circle, r'$x_1$', '')

    draw_arrow(3.0 + r_circle, 4.0, 4.5 - r_circle, 3.0 + 0.15, r'$s_{10}$', r'$\frac{\partial L}{\partial s_{10}}$')
    draw_arrow(3.0 + r_circle, 2.0, 4.5 - r_circle, 3.0 - 0.15, r'$s_{11}$', r'$\frac{\partial L}{\partial s_{11}}$')
    draw_arrow(4.5 + r_circle, 3.0, 6.0 - r_circle, 3.0, r'$s_{12}$', r'$\frac{\partial L}{\partial s_{12}}$')
    draw_arrow(6.0, 1.2, 6.0, 3.0 - r_circle, r'$b_{0[1]}$', r'$\frac{\partial L}{\partial b_{0[1]}}$', fwd_offset=(0.3, 0), grad_offset=(-0.35, 0))
    draw_arrow(6.0 + r_circle, 3.0, 8.0 - 0.65, 3.0, r'$v_1$', r'$\frac{\partial L}{\partial v_1}$')

    # Camada de Saída
    draw_arrow(8.0 + 0.65, 7.5, 11.0 - r_circle, 6.5, r'$y_0$', r'$\frac{\partial L}{\partial y_0}$')
    draw_arrow(9.5, 8.2, 11.0, 6.5 + r_circle, r'$w_{1[0]}$', r'$\frac{\partial L}{\partial w_{1[0]}}$', fwd_offset=(0.2, 0.15), grad_offset=(-0.25, -0.15))

    draw_arrow(8.0 + 0.65, 3.0, 11.0 - r_circle, 4.0, r'$y_1$', r'$\frac{\partial L}{\partial y_1}$')
    draw_arrow(9.5, 2.2, 11.0, 4.0 - r_circle, r'$w_{1[1]}$', r'$\frac{\partial L}{\partial w_{1[1]}}$', fwd_offset=(0.2, -0.15), grad_offset=(-0.25, 0.15))

    draw_arrow(11.0 + r_circle, 6.5, 13.0 - r_circle, 5.25 + 0.15, r'$s_{20}$', r'$\frac{\partial L}{\partial s_{20}}$')
    draw_arrow(11.0 + r_circle, 4.0, 13.0 - r_circle, 5.25 - 0.15, r'$s_{21}$', r'$\frac{\partial L}{\partial s_{21}}$')
    draw_arrow(13.0 + r_circle, 5.25, 14.8 - r_circle, 5.25, r'$s_{22}$', r'$\frac{\partial L}{\partial s_{22}}$')
    draw_arrow(14.8, 7.0, 14.8, 5.25 + r_circle, r'$b_{1[0]}$', r'$\frac{\partial L}{\partial b_{1[0]}}$', fwd_offset=(0.3, 0), grad_offset=(-0.35, 0))
    draw_arrow(14.8 + r_circle, 5.25, 17.0 - 0.65, 5.25, r'$v_2$', r'$\frac{\partial L}{\partial v_2}$')

    # Perda
    draw_arrow(17.0 + 0.65, 5.25, 19.2 - r_circle, 5.25, r'$y_2$', r'$\frac{\partial L}{\partial y_2}$')
    draw_arrow(19.2, 7.0, 19.2, 5.25 + r_circle, r'$d$', '', fwd_offset=(0.2, 0))
    draw_arrow(19.2 + r_circle, 5.25, 21.2 - r_circle, 5.25, r'$e$', r'$\frac{\partial L}{\partial e}$')
    draw_arrow(21.2 + r_circle, 5.25, 23.0 - r_circle, 5.25, r'$e^2$', r'$\frac{\partial L}{\partial (e^2)}$')
    draw_arrow(23.0, 6.8, 23.0, 5.25 + r_circle, r'$\frac{1}{2}$', '', fwd_offset=(0.2, 0))

    # Saída Final da Perda L
    ax.annotate('', xy=(23.9, 5.25), xytext=(23.0 + r_circle, 5.25),
                arrowprops=dict(arrowstyle="->", color=c_edge, lw=1.5, mutation_scale=12), zorder=3)
    ax.text(23.95, 5.25, r'$L$', fontsize=12, fontweight='bold', color='#16a085', va='center')
    ax.text(23.95, 4.80, r'$\frac{\partial L}{\partial L} = 1$', fontsize=9.5, fontweight='bold', color=c_grad, va='center')

    # Legendas explicativas
    legend_elements = [
        patches.Patch(facecolor='#ffffff', edgecolor='#34495e', label='Operações / Nós Forward'),
        patches.Patch(facecolor='#fcf3cf', edgecolor='#f39c12', label='Função de Ativação Sigmoide'),
        patches.Patch(color=c_edge, label='Forward Pass (Fluxo das Variáveis)'),
        patches.Patch(color=c_grad, label='Backward Pass (Gradientes Analíticos)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)

    plt.suptitle("Grafo Computacional Detalhado da Rede A ($2 \\rightarrow 2 \\rightarrow 1$)",
                 fontsize=15, fontweight='bold', y=0.96)
    plt.savefig(filename_png, bbox_inches='tight')
    plt.savefig(filename_svg, bbox_inches='tight')
    plt.close()
    print(f"[OK] Grafo computacional salvo em '{output_dir}'.")


def main():
    output_dir = os.path.join('graphics', 'rede_a')
    os.makedirs(output_dir, exist_ok=True)

    # 1. Carrega dados e salva plot inicial
    X, Y = get_dataset()
    plot_dataset(X, Y, output_dir)

    # 2. Inicialização dos Pesos
    np.random.seed(42)
    w0 = np.random.rand(2, 2)
    w1 = np.random.rand(2)
    b0 = np.random.rand(2)
    b1 = np.random.rand(1)

    w0_init = w0.copy()
    w1_init = w1.copy()
    b0_init = b0.copy()
    b1_init = b1.copy()

    taxa = 0.1

    print("=================================================================")
    print(" REDE A (2 -> 2 -> 1): 2 NEURÔNIOS OCULTOS COM SIGMOIDE")
    print("=================================================================")
    print(" 1. REDE MANUAL (BACKPROPAGATION ANALÍTICO)")
    print("=================================================================")

    acc_init = sum(1 for i in range(100) if run_neural_net(X[i], w0, b0, b1, w1) == Y[i])
    print(f"Acurácia inicial: {acc_init}%")

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

    acc_manual = sum(1 for i in range(100) if run_neural_net(X[i], w0, b0, b1, w1) == Y[i])
    print(f"Acurácia final (Manual): {acc_manual}%")

    print("\n=================================================================")
    print(" 2. REDE KERAS (ARQUITETURA 2 -> 2 -> 1 e MESMOS PESOS INICIAIS)")
    print("=================================================================")

    model = Sequential([
        Input(shape=(2,)),
        Dense(2, activation='sigmoid'),
        Dense(1, activation='sigmoid')
    ])

    model.layers[0].set_weights([w0_init.T, b0_init])
    model.layers[1].set_weights([w1_init.reshape(2, 1), b1_init])

    opt = SGD(learning_rate=taxa * 50)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    model.fit(X, Y, epochs=10000, batch_size=100, verbose=False)

    loss_k, acc_k = model.evaluate(X, Y, verbose=0)
    print(f"Acurácia final (Keras) : {acc_k * 100:.0f}%")
    print(f"Loss final (Keras MSE) : {loss_k:.6f}  (equiv. sum Loss manual: {loss_k * 50:.6f})")

    w_k0, b_k0 = model.layers[0].get_weights()
    w_k1, b_k1 = model.layers[1].get_weights()

    print("\n=================================================================")
    print(" 3. COMPARAÇÃO DOS PESOS FINAIS APRENDIDOS")
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

    # 4. Fronteira de decisão e retas dos neurônios
    title = "Rede A (2 -> 2 -> 1): 2 Retas Ocultas e Fronteira Não-Linear Final"
    plot_decision_boundary(model, X, Y, w0, b0, title, output_dir)

    # 5. Diagrama da Arquitetura
    plot_neural_network_architecture(output_dir)

    # 6. Grafo Computacional Detalhado
    plot_computational_graph(output_dir)


if __name__ == '__main__':
    main()
