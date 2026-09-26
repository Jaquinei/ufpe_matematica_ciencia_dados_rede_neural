import os
import sys

# Garante que o diretório src esteja no sys.path para importação dos módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import rede_a
import rede_b


def main():
    print("=" * 70)
    print("   UFPE - MATEMÁTICA PARA CIÊNCIA DE DADOS: EXECUÇÃO DE REDES NEURAIS")
    print("=" * 70)

    print("\n" + "=" * 70)
    print(">>> [1/2] EXECUTANDO REDE A (2 -> 2 -> 1): 2 NEURÔNIOS OCULTOS")
    print("=" * 70 + "\n")
    rede_a.main()

    print("\n\n" + "=" * 70)
    print(">>> [2/2] EXECUTANDO REDE B (2 -> 4 -> 1): 4 NEURÔNIOS OCULTOS")
    print("=" * 70 + "\n")
    rede_b.main()

    print("\n" + "=" * 70)
    print(" [CONCLUÍDO] Ambas as redes foram treinadas e validadas com sucesso!")
    print(" Gráficos salvos em 'graphics/rede_a/' e 'graphics/rede_b/'.")
    print("=" * 70)


if __name__ == '__main__':
    main()