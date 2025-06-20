import sys

from utils.game_modes import (
    start_game_against_minimax,
    start_train_network,
    start_game_against_network
    )

class TicTacToeCLI:
    """
    CLI para:
      1) jogo contra Minimax
      2) treinamento via AG
      3) jogo contra rede treinada

    Convenção de peças (alinhada ao treinamento):
       +1  → X
       -1  → O
    """

    def start(self):
        while True:
            print(f"""{5* '='} MENU {5 * '='}\n1 - Jogar contra Minimax (modo humano)\n2 - Treinar IA com AG
3 - Jogar contra IA treinada\n0 - Sair""")
            cmd = input("Escolha: ").strip()
            if cmd == "1":
                start_game_against_minimax()
            elif cmd == "2":
                start_train_network()
            elif cmd == "3":
                start_game_against_network()
            elif cmd == "0":
                print("Até logo!")
                sys.exit(0)
            else:
                print("Opção inválida. Tente novamente.")