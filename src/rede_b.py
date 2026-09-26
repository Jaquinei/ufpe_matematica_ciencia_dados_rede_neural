import os
import sys

# Garante que o diretório src esteja no path para importação
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    np, plt, Sequential, Dense, Input, SGD,
    sigmoid, relu, get_dataset, plot_dataset, plot_decision_boundary
)


def run_neural_net(x, w0, b0, b1, w1):
    """
    Executa a inferência (Forward Pass) na Rede B (2 -> 4 -> 1) com ReLU na camada oculta.
    Retorna 1 se y_out >= 0.5, senão 0.
    """
    # Camada oculta: 4 neurônios com ativação ReLU
    v = np.dot(w0, x) + b0      # shape (4,)
    y = relu(v)                 # shape (4,)

    # Camada de saída: 1 neurônio com ativação Sigmoide
    v_out = np.dot(w1, y) + b1[0]
    y_out = sigmoid(v_out)
    return 1 if y_out >= 0.5 else 0


def neural_net(x, d, w0, b0, b1, w1):
    """
    Executa o Forward Pass e deriva analiticamente os gradientes via Regra da Cadeia (Backward Pass)
    com ativação ReLU na camada oculta e Sigmoide na saída para a arquitetura 2 -> 4 -> 1.
    """
    # ==================== FORWARD PASS ====================
    # Camada Oculta (4 neurônios com ReLU)
    s0 = w0[:, 0] * x[0]        # shape (4,)
    s1 = w0[:, 1] * x[1]        # shape (4,)
    s_soma = s0 + s1            # shape (4,)
    v = s_soma + b0             # shape (4,)
    y = relu(v)                 # shape (4,)

    # Camada de Saída (1 neurônio com Sigmoide)
    s_out = y * w1              # shape (4,)
    v_out = np.sum(s_out) + b1[0]
    y_out = sigmoid(v_out)

    # Perda quadrática
    e = y_out - d
    L = 0.5 * (e ** 2)

    # ==================== BACKWARD PASS ====================
    grad_w0 = np.zeros(w0.shape)  # (4, 2)
    grad_w1 = np.zeros(w1.shape)  # (4,)
    grad_b0 = np.zeros(b0.shape)  # (4,)
    grad_b1 = np.zeros(b1.shape)  # (1,)

    grad_L = 1.0
    grad_e = grad_L * e

    grad_y_out = grad_e
    grad_v_out = grad_y_out * y_out * (1.0 - y_out)
    grad_b1[0] = grad_v_out

    # Gradientes dos pesos da camada de saída (w1)
    grad_w1 = grad_v_out * y

    # Gradientes propagados para a camada oculta
    grad_y = grad_v_out * w1

    # Derivada da função de ativação ReLU: 1 se v > 0, senão 0
    grad_v = grad_y * (v > 0).astype(float)

    # Gradientes dos biases da camada oculta (b0)
    grad_b0 = grad_v

    # Gradientes dos pesos da camada oculta (w0)
    grad_w0[:, 0] = grad_v * x[0]
    grad_w0[:, 1] = grad_v * x[1]

    return grad_w0, grad_b0, grad_w1, grad_b1, L


def plot_neural_network_architecture(output_dir):
    """Gera o diagrama visual da arquitetura da Rede B (2 -> 4 -> 1) com ReLU."""
    import matplotlib.patches as patches

    os.makedirs(output_dir, exist_ok=True)
    filename_png = os.path.join(output_dir, 'arquitetura_rede.png')
    filename_svg = os.path.join(output_dir, 'arquitetura_rede.svg')

    fig, ax = plt.subplots(figsize=(13, 8.5), dpi=150)
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    input_coords = [(0.15, 0.65), (0.15, 0.35)]
    hidden_coords = [(0.50, 0.80), (0.50, 0.60), (0.50, 0.40), (0.50, 0.20)]
    output_coords = [(0.85, 0.50)]

    radius = 0.045

    # Conexões Entrada -> Oculta
    for i, (xi, yi) in enumerate(input_coords):
        for j, (xh, yh) in enumerate(hidden_coords):
            ax.annotate('', xy=(xh - radius, yh), xytext=(xi + radius, yi),
                        arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.3, mutation_scale=12))

    # Conexões Oculta -> Saída
    for j, (xh, yh) in enumerate(hidden_coords):
        xo, yo = output_coords[0]
        ax.annotate('', xy=(xo - radius, yo), xytext=(xh + radius, yh),
                    arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.3, mutation_scale=12))

    # Saída
    xo, yo = output_coords[0]
    ax.annotate('', xy=(0.98, yo), xytext=(xo + radius, yo),
                arrowprops=dict(arrowstyle="->", color="#27ae60", lw=2.2, mutation_scale=18))
    ax.text(0.99, yo + 0.04, r'$\hat{y} \in \{0, 1\}$', fontsize=12, fontweight='bold', color='#27ae60', ha='left')
    ax.text(0.99, yo - 0.04, r'($y_{out} \geq 0.5$)', fontsize=10, color='#555555', ha='left')

    # Nós Entrada
    for i, (x, y) in enumerate(input_coords):
        circle = patches.Circle((x, y), radius, facecolor='#3498db', edgecolor='#1d6fa5', lw=2, zorder=4)
        ax.add_patch(circle)
        ax.text(x, y, f'$x_{i}$', fontsize=13, fontweight='bold', color='white', ha='center', va='center', zorder=5)

    # Nós Ocultos (com ReLU)
    for j, (x, y) in enumerate(hidden_coords):
        circle = patches.Circle((x, y), radius, facecolor='#9b59b6', edgecolor='#6c3483', lw=2, zorder=4)
        ax.add_patch(circle)
        ax.text(x, y, f'$h_{j}$', fontsize=12, fontweight='bold', color='white', ha='center', va='center', zorder=5)
        ax.text(x + 0.065, y, rf'$v_{j} \rightarrow y_{j}=\mathrm{{ReLU}}(v_{j})$', fontsize=9.5, color='#6c3483',
                fontweight='bold', va='center', bbox=dict(boxstyle='square,pad=0.15', facecolor='#f4ecf7', edgecolor='#d2b4de', lw=0.8))

    # Nó Saída (com Sigmoide)
    circle = patches.Circle((xo, yo), radius, facecolor='#2ecc71', edgecolor='#1e8449', lw=2, zorder=4)
    ax.add_patch(circle)
    ax.text(xo, yo, '$y_{out}$', fontsize=11, fontweight='bold', color='white', ha='center', va='center', zorder=5)
    ax.text(xo, yo - 0.08, r'$y_{out} = \sigma(v_{out})$', fontsize=10, color='#1e8449', fontweight='bold', ha='center',
            bbox=dict(boxstyle='square,pad=0.2', facecolor='#eafaf1', edgecolor='#a9dfbf', lw=1))

    # Títulos das Colunas
    ax.text(0.15, 0.93, 'Camada de Entrada\n(2 Variáveis: $x_0, x_1$)', fontsize=11, fontweight='bold', ha='center', va='center', color='#2c3e50')
    ax.text(0.50, 0.93, 'Camada Oculta\n(4 Neurônios ReLU)', fontsize=11, fontweight='bold', ha='center', va='center', color='#2c3e50')
    ax.text(0.85, 0.93, 'Camada de Saída\n(1 Neurônio Sigmoide)', fontsize=11, fontweight='bold', ha='center', va='center', color='#2c3e50')

    formula_text = (
        r"Equações Matriciais:" + "\n"
        r"Camada Oculta: $\mathbf{v} = \mathbf{W}_0 \mathbf{x} + \mathbf{b}_0, \quad \mathbf{y} = \mathrm{ReLU}(\mathbf{v}) = \max(0, \mathbf{v}) \quad (\mathbf{W}_0 \in \mathbb{R}^{4 \times 2}, \ \mathbf{b}_0 \in \mathbb{R}^4)$" + "\n"
        r"Camada de Saída: $v_{out} = \mathbf{w}_1^T \mathbf{y} + b_1, \quad y_{out} = \sigma(v_{out}) \quad (\mathbf{w}_1 \in \mathbb{R}^4, \ b_1 \in \mathbb{R})$" + "\n"
        r"Função de Perda Quadrática: $L = \frac{1}{2}(y_{out} - d)^2$"
    )
    fig.text(0.5, 0.06, formula_text, fontsize=9.5, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#bdc3c7', lw=1.2))

    plt.suptitle("Arquitetura da Rede B ($2 \\rightarrow 4 \\rightarrow 1$ - ReLU Oculta)", fontsize=15, fontweight='bold', y=0.98)
    plt.savefig(filename_png, bbox_inches='tight')
    plt.savefig(filename_svg, bbox_inches='tight')
    plt.close()
    print(f"[OK] Gráfico da arquitetura salvo em '{output_dir}'.")


def plot_computational_graph(output_dir):
    """Gera o grafo computacional detalhado (Forward em azul/preto, Backward em vermelho) da Rede B com ReLU."""
    import matplotlib.patches as patches

    os.makedirs(output_dir, exist_ok=True)
    filename_png = os.path.join(output_dir, 'grafo_computacional.png')
    filename_svg = os.path.join(output_dir, 'grafo_computacional.svg')

    fig, ax = plt.subplots(figsize=(24, 12), dpi=200)
    ax.axis('off')
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 12)

    c_edge = '#2980b9'
    c_grad = '#c0392b'
    c_forward = '#2c3e50'
    c_op_bg = '#ffffff'
    c_op_border = '#34495e'
    c_act_bg = '#d5f5e3'
    c_act_border = '#27ae60'
    c_sig_bg = '#fcf3cf'
    c_sig_border = '#f39c12'

    r_circle = 0.28

    def draw_circle_op(x, y, label):
        circle = patches.Circle((x, y), r_circle, facecolor=c_op_bg, edgecolor=c_op_border, lw=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, label, fontsize=12, fontweight='bold', color=c_op_border, ha='center', va='center', zorder=6)

    def draw_rect_act(x, y, label, is_relu=True):
        w, h = 1.3, 0.55
        bg = c_act_bg if is_relu else c_sig_bg
        border = c_act_border if is_relu else c_sig_border
        txt_color = '#1e8449' if is_relu else '#7d6608'
        rect = patches.FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle='round,pad=0.06',
                                      facecolor=bg, edgecolor=border, lw=1.5, zorder=5)
        ax.add_patch(rect)
        ax.text(x, y, label, fontsize=9.5, fontweight='bold', color=txt_color, ha='center', va='center', zorder=6)

    def draw_arrow(x1, y1, x2, y2, label_fwd='', label_grad='', fwd_offset=(0, 0.15), grad_offset=(0, -0.20),
                   fwd_ha='center', grad_ha='center', curve=0):
        rad_str = f"arc3,rad={curve}" if curve != 0 else "arc3,rad=0"
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=c_edge, lw=1.3, mutation_scale=11,
                                    connectionstyle=rad_str), zorder=3)
        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2
        if curve != 0:
            ym += curve * 0.8

        if label_fwd:
            ax.text(xm + fwd_offset[0], ym + fwd_offset[1], label_fwd, fontsize=8.5, fontweight='bold',
                    color=c_forward, ha=fwd_ha, va='center',
                    bbox=dict(boxstyle='square,pad=0.08', facecolor='white', edgecolor='none', alpha=0.85))
        if label_grad:
            ax.text(xm + grad_offset[0], ym + grad_offset[1], label_grad, fontsize=7.5, fontweight='bold',
                    color=c_grad, ha=grad_ha, va='center',
                    bbox=dict(boxstyle='square,pad=0.08', facecolor='#fdedec', edgecolor='none', alpha=0.85))

    y_centers = [10.2, 7.6, 5.0, 2.4]

    for j, yc in enumerate(y_centers):
        draw_circle_op(2.8, yc + 0.6, r'$\times$')
        draw_circle_op(2.8, yc - 0.6, r'$\times$')
        draw_circle_op(4.5, yc, '+')
        draw_circle_op(6.0, yc, '+')
        draw_rect_act(7.8, yc, rf'$\mathrm{{ReLU}}(v_{j})$', is_relu=True)

        draw_arrow(0.8, yc + 0.6, 2.8 - r_circle, yc + 0.6, rf'$w_{{0[{j},0]}}$', rf'$\frac{{\partial L}}{{\partial w_{{0[{j},0]}}}}$')
        draw_arrow(0.8, yc - 0.6, 2.8 - r_circle, yc - 0.6, rf'$w_{{0[{j},1]}}$', rf'$\frac{{\partial L}}{{\partial w_{{0[{j},1]}}}}$')

        draw_arrow(2.8 + r_circle, yc + 0.6, 4.5 - r_circle, yc + 0.15, rf'$s_{{{j}0}}$', '')
        draw_arrow(2.8 + r_circle, yc - 0.6, 4.5 - r_circle, yc - 0.15, rf'$s_{{{j}1}}$', '')
        draw_arrow(4.5 + r_circle, yc, 6.0 - r_circle, yc, rf'$s_{{{j}\Sigma}}$', '')
        draw_arrow(6.0, yc + 0.9, 6.0, yc + r_circle, rf'$b_{{0[{j}]}}$', rf'$\frac{{\partial L}}{{\partial b_{{0[{j}]}}}}$', fwd_offset=(0.25, 0), grad_offset=(-0.3, 0))
        draw_arrow(6.0 + r_circle, yc, 7.8 - 0.65, yc, rf'$v_{j}$', rf'$\frac{{\partial L}}{{\partial v_{j}}}$')

        draw_circle_op(10.8, yc, r'$\times$')
        draw_arrow(7.8 + 0.65, yc, 10.8 - r_circle, yc, rf'$y_{j}$', rf'$\frac{{\partial L}}{{\partial y_{j}}}$')
        draw_arrow(9.6, yc + 0.6, 10.8, yc + r_circle, rf'$w_{{1[{j}]}}$', rf'$\frac{{\partial L}}{{\partial w_{{1[{j}]}}}}$', fwd_offset=(0.15, 0.12), grad_offset=(-0.2, -0.12))

    # Junção da Camada de Saída
    draw_circle_op(13.2, 6.3, r'$\sum$')
    draw_circle_op(15.0, 6.3, '+')
    draw_rect_act(17.0, 6.3, r'$\sigma(v_{out})$', is_relu=False)

    for j, yc in enumerate(y_centers):
        draw_arrow(10.8 + r_circle, yc, 13.2 - r_circle, 6.3 + (1.5 - j) * 0.15, rf'$s_{{out[{j}]}}$', '')

    draw_arrow(13.2 + r_circle, 6.3, 15.0 - r_circle, 6.3, r'$s_{out\Sigma}$', '')
    draw_arrow(15.0, 7.6, 15.0, 6.3 + r_circle, r'$b_{1[0]}$', r'$\frac{\partial L}{\partial b_1}$', fwd_offset=(0.25, 0), grad_offset=(-0.3, 0))
    draw_arrow(15.0 + r_circle, 6.3, 17.0 - 0.65, 6.3, r'$v_{out}$', r'$\frac{\partial L}{\partial v_{out}}$')

    # Cálculo da Perda
    draw_circle_op(19.2, 6.3, '-')
    draw_circle_op(21.0, 6.3, r'$(\cdot)^2$')
    draw_circle_op(22.7, 6.3, r'$\times$')

    draw_arrow(17.0 + 0.65, 6.3, 19.2 - r_circle, 6.3, r'$y_{out}$', r'$\frac{\partial L}{\partial y_{out}}$')
    draw_arrow(19.2, 7.6, 19.2, 6.3 + r_circle, r'$d$', '', fwd_offset=(0.2, 0))
    draw_arrow(19.2 + r_circle, 6.3, 21.0 - r_circle, 6.3, r'$e$', r'$\frac{\partial L}{\partial e}$')
    draw_arrow(21.0 + r_circle, 6.3, 22.7 - r_circle, 6.3, r'$e^2$', r'$\frac{\partial L}{\partial (e^2)}$')
    draw_arrow(22.7, 7.6, 22.7, 6.3 + r_circle, r'$\frac{1}{2}$', '', fwd_offset=(0.2, 0))

    ax.annotate('', xy=(23.8, 6.3), xytext=(22.7 + r_circle, 6.3),
                arrowprops=dict(arrowstyle="->", color=c_edge, lw=1.5, mutation_scale=12), zorder=3)
    ax.text(23.85, 6.3, r'$L$', fontsize=12, fontweight='bold', color='#16a085', va='center')
    ax.text(23.85, 5.9, r'$\frac{\partial L}{\partial L} = 1$', fontsize=9, fontweight='bold', color=c_grad, va='center')

    legend_elements = [
        patches.Patch(facecolor='#ffffff', edgecolor='#34495e', label='Operações / Nós Forward'),
        patches.Patch(facecolor='#d5f5e3', edgecolor='#27ae60', label='Ativação Camada Oculta: ReLU'),
        patches.Patch(facecolor='#fcf3cf', edgecolor='#f39c12', label='Ativação Camada Saída: Sigmoide'),
        patches.Patch(color=c_edge, label='Forward Pass (Fluxo das Variáveis)'),
        patches.Patch(color=c_grad, label='Backward Pass (Gradientes Analíticos)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)

    plt.suptitle("Grafo Computacional Detalhado da Rede B ($2 \\rightarrow 4 \\rightarrow 1$ - ReLU Oculta)",
                 fontsize=15, fontweight='bold', y=0.97)
    plt.savefig(filename_png, bbox_inches='tight')
    plt.savefig(filename_svg, bbox_inches='tight')
    plt.close()
    print(f"[OK] Grafo computacional salvo em '{output_dir}'.")


def main():
    output_dir = os.path.join('graphics', 'rede_b')
    os.makedirs(output_dir, exist_ok=True)

    # 1. Carrega dados e salva plot inicial
    X, Y = get_dataset()
    plot_dataset(X, Y, output_dir)

    # 2. Inicialização dos Pesos (4 neurônios ocultos)
    np.random.seed(42)
    w0 = np.random.randn(4, 2) * np.sqrt(2.0 / 2.0)  # Inicialização He para ReLU
    w1 = np.random.randn(4) * np.sqrt(2.0 / 4.0)
    b0 = np.zeros(4)
    b1 = np.zeros(1)

    w0_init = w0.copy()
    w1_init = w1.copy()
    b0_init = b0.copy()
    b1_init = b1.copy()

    taxa = 0.05

    print("=================================================================")
    print(" REDE B (2 -> 4 -> 1): 4 NEURÔNIOS OCULTOS COM RELU")
    print("=================================================================")
    print(" 1. REDE MANUAL (BACKPROPAGATION ANALÍTICO COM RELU)")
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
    print(" 2. REDE KERAS (ARQUITETURA 2 -> 4 -> 1 COM RELU E MESMOS PESOS)")
    print("=================================================================")

    model = Sequential([
        Input(shape=(2,)),
        Dense(4, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.layers[0].set_weights([w0_init.T, b0_init])
    model.layers[1].set_weights([w1_init.reshape(4, 1), b1_init])

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

    # 4. Fronteira de decisão e retas dos 4 neurônios
    title = "Rede B (2 -> 4 -> 1): 4 Retas Ocultas e Fronteira Não-Linear (ReLU)"
    plot_decision_boundary(model, X, Y, w0, b0, title, output_dir)

    # 5. Diagrama da Arquitetura
    plot_neural_network_architecture(output_dir)

    # 6. Grafo Computacional Detalhado
    plot_computational_graph(output_dir)


if __name__ == '__main__':
    main()
