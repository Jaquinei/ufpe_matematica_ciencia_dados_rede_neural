# UFPE - Matemática para Ciência de Dados: Implementação de Redes Neurais (MLP)

Projeto desenvolvido para a disciplina de **Matemática para Ciência de Dados** (UFPE). O objetivo é demonstrar a fundamentação matemática, implementação manual do algoritmo de *Backpropagation* (regra da cadeia passo a passo) e comparação determinística com o framework **Keras/TensorFlow** no problema clássico de classificação não-linear das **Duas Luas (*make_moons*)**.

O projeto conta com duas arquiteturas de rede para análise e comparação geométrica:
- **Rede A ($2 \rightarrow 2 \rightarrow 1$):** Arquitetura básica com 2 neurônios na camada oculta.
- **Rede B ($2 \rightarrow 4 \rightarrow 1$):** Arquitetura aprimorada com 4 neurônios na camada oculta, contornando a não-linearidade das luas e atingindo acurácia próxima a 100%.

---

## Visão Geral do Projeto

O projeto aborda a resolução do problema de classificação binária gerado via `sklearn.datasets.make_moons` (100 amostras com ruído 0.1).

Para cada arquitetura, foram desenvolvidas duas implementações:
1. **Rede Neural Manual (NumPy Puro):**
   - Cálculo explícito do *Forward Pass* e derivação analítica de todos os gradientes no *Backpropagation* usando a regra da cadeia.
   - Atualização dos pesos e biases via Gradiente Descendente.
   - Função de perda quadrática: $L = \frac{1}{2}(y - d)^2$.
2. **Rede Neural Equivalente em Keras:**
   - Implementação utilizando `keras.models.Sequential` e camadas `Dense`.
   - Inicialização com os mesmos pesos da rede manual e taxa de aprendizado equivalente para validação cruzada determinística.

---

## Estrutura Modular do Projeto

O código-fonte foi estruturado seguindo o princípio de responsabilidade única e DRY (*Don't Repeat Yourself*):

- **`src/utils.py`**: Centraliza as configurações do ambiente (supressão de logs do TensorFlow/oneDNN), importações centrais, definição da função de ativação Sigmoide, geração consistente do dataset e rotinas genéricas de plotagem (dataset e fronteira de decisão com as retas ocultas).
- **`src/rede_a.py`**: Contém a derivação analítica escalar passo a passo para a topologia $2 \rightarrow 2 \rightarrow 1$, seu loop de treinamento e a geração de seus diagramas de arquitetura e grafo computacional.
- **`src/rede_b.py`**: Contém a formulação matricial/vetorial para a topologia $2 \rightarrow 4 \rightarrow 1$, seu loop de treinamento e a geração de seus diagramas de arquitetura e grafo computacional.
- **`src/main.py`**: Orquestrador central que executa o pipeline de ambas as redes em sequência.

---

## Especificações e Comparação entre Rede A e Rede B

| Especificação | Rede A (Básica) | Rede B (Aprimorada) |
| :--- | :--- | :--- |
| **Arquivo Fonte** | `src/rede_a.py` | `src/rede_b.py` |
| **Topologia** | **$2 \rightarrow 2 \rightarrow 1$** | **$2 \rightarrow 4 \rightarrow 1$** |
| **Neurônios Ocultos** | 2 neurônios | 4 neurônios |
| **Ativação Camada Oculta** | Sigmoide | Sigmoide |
| **Ativação Camada de Saída** | Sigmoide | Sigmoide |
| **Total de Parâmetros** | 9 parâmetros | 17 parâmetros |
| **Retas de Partição Ocultas** | 2 retas no plano 2D | 4 retas no plano 2D |
| **Acurácia Final Obtida** | $\approx 85\% - 90\%$ | **$\approx 98\% - 100\%$** |
| **Convergência Keras vs. Manual** | Pesos e perda idênticos | Pesos e perda idênticos |

---

## Visualizações e Resultados Gerados

Os gráficos são organizados em pastas dedicadas para cada rede:

### 1. Rede A ($2 \rightarrow 2 \rightarrow 1$) - Diretório `graphics/rede_a/`

#### Arquitetura da Rede A
![Arquitetura Rede A](graphics/rede_a/arquitetura_rede.png)

#### Grafo Computacional da Rede A (Forward & Backward Pass)
![Grafo Computacional Rede A](graphics/rede_a/grafo_computacional.png)

#### Fronteira de Decisão e Retas da Rede A
Visualização das **2 retas lineares** dos neurônios da camada oculta ($v_0 = 0$ e $v_1 = 0$) e da fronteira resultante:
![Fronteira Rede A](graphics/rede_a/fronteira_duas_luas.png)

---

### 2. Rede B ($2 \rightarrow 4 \rightarrow 1$) - Diretório `graphics/rede_b/`

#### Arquitetura da Rede B
![Arquitetura Rede B](graphics/rede_b/arquitetura_rede.png)

#### Grafo Computacional da Rede B (Forward & Backward Pass)
![Grafo Computacional Rede B](graphics/rede_b/grafo_computacional.png)

#### Fronteira de Decisão e Retas da Rede B
Visualização das **4 retas lineares** dos neurônios da camada oculta ($v_0, v_1, v_2, v_3 = 0$) envolvendo as luas com precisão superior:
![Fronteira Rede B](graphics/rede_b/fronteira_duas_luas.png)

---

## Estrutura de Diretórios

```text
├── graphics/
│   ├── rede_a/
│   │   ├── arquitetura_rede.png      # Diagrama da arquitetura Rede A (PNG)
│   │   ├── arquitetura_rede.svg      # Diagrama da arquitetura Rede A (SVG)
│   │   ├── grafo_computacional.png   # Grafo computacional Rede A (PNG)
│   │   ├── grafo_computacional.svg   # Grafo computacional Rede A (SVG)
│   │   ├── duas_luas.svg             # Gráfico do dataset de entrada
│   │   ├── fronteira_duas_luas.png   # Fronteira de decisão e 2 retas (PNG)
│   │   └── fronteira_duas_luas.svg   # Fronteira de decisão e 2 retas (SVG)
│   └── rede_b/
│       ├── arquitetura_rede.png      # Diagrama da arquitetura Rede B (PNG)
│       ├── arquitetura_rede.svg      # Diagrama da arquitetura Rede B (SVG)
│       ├── grafo_computacional.png   # Grafo computacional Rede B (PNG)
│       ├── grafo_computacional.svg   # Grafo computacional Rede B (SVG)
│       ├── duas_luas.svg             # Gráfico do dataset de entrada
│       ├── fronteira_duas_luas.png   # Fronteira de decisão e 4 retas (PNG)
│       └── fronteira_duas_luas.svg   # Fronteira de decisão e 4 retas (SVG)
├── src/
│   ├── main.py                       # Orquestrador principal (executa Rede A e Rede B)
│   ├── rede_a.py                     # Implementação e execução da Rede A (2 -> 2 -> 1)
│   ├── rede_b.py                     # Implementação e execução da Rede B (2 -> 4 -> 1)
│   └── utils.py                      # Utilitários compartilhados (dataset, ativações, plots e ambiente)
├── requirements.txt                  # Dependências do projeto (scikit-learn, numpy, matplotlib, tensorflow)
├── run.ps1                           # Script de execução automatizada para Windows (PowerShell)
├── run.bat                           # Script de execução automatizada para Windows (CMD)
├── run.sh                            # Script de execução automatizada para Linux/macOS
└── README.md                         # Documentação completa do projeto
```

---

## Pré-requisitos

- **Python 3.9 ou superior**
- Gerenciador de pacotes `pip`

Verifique suas versões com:
```bash
python --version
pip --version
```

---

## Como Executar

### 1. Criando e Ativando o Ambiente Virtual (`venv`)

#### Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(Caso o PowerShell bloqueie a execução de scripts: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

#### Windows (Command Prompt):
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

#### Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 2. Instalando as Dependências

Com o ambiente virtual ativado:
```bash
pip install -r requirements.txt
```

---

### 3. Executando as Aplicações

#### Executando ambas as redes sequencialmente (Recomendado):
```bash
python src/main.py
```
*(Ou através dos scripts `.\run.ps1`, `run.bat` ou `./run.sh`)*

#### Ou executando uma rede individualmente:
- **Rede A ($2 \rightarrow 2 \rightarrow 1$):**
  ```bash
  python src/rede_a.py
  ```
- **Rede B ($2 \rightarrow 4 \rightarrow 1$):**
  ```bash
  python src/rede_b.py
  ```

---

## Autor

- **Jaquinei de Oliveira** — UFPE
