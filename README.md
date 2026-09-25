# UFPE - Matemática para Ciência de Dados: Implementação de Rede Neural (MLP)

Projeto desenvolvido para a disciplina de **Matemática para Ciência de Dados** (UFPE). O objetivo é demonstrar a fundamentação matemática, implementação manual do algoritmo de *Backpropagation* e comparação determinística com o framework **Keras/TensorFlow** no problema clássico de classificação não-linear das **Duas Luas (*make_moons*)**.

---

## Visão Geral do Projeto

O projeto aborda a resolução de um problema de classificação binária não-linearmente separável gerado via `sklearn.datasets.make_moons`. 

Para resolver o problema, foram desenvolvidas duas abordagens:
1. **Rede Neural Manual (NumPy Puro):**
   - Arquitetura $2 \rightarrow 2 \rightarrow 1$ (2 entradas, 2 neurônios ocultos com ativação Sigmoide e 1 neurônio de saída com ativação Sigmoide).
   - Cálculo explícito do *Forward Pass* e derivação analítica passo a passo do *Backpropagation* usando a regra da cadeia para atualização dos pesos e biases via Gradiente Descendente.
   - Função de perda quadrática: $L = \frac{1}{2}(y - d)^2$.
2. **Rede Neural Equivalente em Keras:**
   - Implementação da mesma topologia utilizando `keras.models.Sequential` e camadas `Dense`.
   - Inicialização com os mesmos pesos e taxa de aprendizado equivalente para validação cruzada exata dos pesos e perda calculados.

---

## Visualizações e Resultados Gerados

A aplicação gera automaticamente gráficos para demonstrar a distribuição dos dados, a geometria dos neurônios e a convergência da rede:

### 1. Distribuição dos Dados (*Dataset Duas Luas*)
Visualização da distribuição espacial dos pontos das duas classes geradas pelo `make_moons`:

![Dataset Duas Luas](graphics/duas_luas.svg)

---

### 2. Fronteira de Decisão e Retas dos Neurônios Ocultos
Visualização das **duas retas lineares** dos neurônios da camada oculta ($v_0 = 0$ e $v_1 = 0$) e da **fronteira de decisão não-linear final** obtida pela combinação na camada de saída:

![Fronteira de Decisão e Retas Ocultas](graphics/fronteira_duas_luas.png)

---

## Estrutura do Projeto

```text
├── graphics/
│   ├── duas_luas.svg            # Gráfico do dataset de entrada
│   ├── fronteira_duas_luas.png  # Gráfico da fronteira de decisão (PNG)
│   └── fronteira_duas_luas.svg  # Gráfico da fronteira de decisão (SVG)
├── src/
│   └── main.py                  # Implementação da rede manual, modelo Keras e gráficos
├── requirements.txt             # Dependências do projeto (scikit-learn, numpy, matplotlib, tensorflow)
├── run.ps1                      # Script de execução automatizada para Windows (PowerShell)
├── run.bat                  # Script de execução automatizada para Windows (CMD)
├── run.sh                   # Script de execução automatizada para Linux/macOS
└── README.md                # Documentação do projeto
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

##  Como Executar

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

### 3. Executando a Aplicação

#### Utilizando os scripts automatizados:
- **Windows (PowerShell):** `.\run.ps1`
- **Windows (CMD):** `run.bat`
- **Linux/macOS:** `./run.sh`

#### Ou executando diretamente via Python:
```bash
python src/main.py
```

---

##  Autor

- **Jaquinei de Oliveira** — UFPE
