import os
import sys

# Garante que o diretório src esteja no sys.path para importação dos módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import rede_a
import rede_b


def run_rede_a():
    print("\n" + "=" * 70)
    print(">>> EXECUTANDO REDE REFERÊNCIA (REDE A: 2 -> 2 -> 1)")
    print("=" * 70 + "\n")
    rede_a.main()
    print("\n" + "=" * 70)
    print("[CONCLUÍDO] Rede A executada com sucesso!")
    print("Gráficos salvos em 'graphics/rede_a/'.")
    print("=" * 70)


def run_rede_b():
    print("\n" + "=" * 70)
    print(">>> EXECUTANDO REDE ALTERADA (REDE B: 2 -> 4 -> 1)")
    print("=" * 70 + "\n")
    rede_b.main()
    print("\n" + "=" * 70)
    print("[CONCLUÍDO] Rede B executada com sucesso!")
    print("Gráficos salvos em 'graphics/rede_b/'.")
    print("=" * 70)


def run_todas():
    print("\n" + "=" * 70)
    print(">>> EXECUTANDO TODAS AS REDES SEQUENCIALMENTE")
    print("=" * 70)
    run_rede_a()
    run_rede_b()
    print("\n" + "=" * 70)
    print("[CONCLUÍDO] Ambas as redes foram executadas com sucesso!")
    print("=" * 70)


def show_menu():
    print("\n" + "=" * 70)
    print("   UFPE - MATEMÁTICA PARA CIÊNCIA DE DADOS: REDES NEURAIS")
    print("=" * 70)
    print("  [1] Rodar Rede Referência (Rede A: 2 -> 2 -> 1)")
    print("  [2] Rodar Rede Alterada (Rede B: 2 -> 4 -> 1)")
    print("  [3] Rodar Todas as Redes (Rede A e Rede B)")
    print("  [0] Sair do Programa")
    print("=" * 70)


def main():
    while True:
        show_menu()
        opcao = input("Escolha uma opção [0-3]: ").strip()

        if opcao == '1':
            run_rede_a()
        elif opcao == '2':
            run_rede_b()
        elif opcao == '3':
            run_todas()
        elif opcao == '0':
            print("\nEncerrando o programa.")
            break
        else:
            print("\n[Opção inválida] Por favor, digite 1, 2, 3 ou 0.")


if __name__ == '__main__':
    main()