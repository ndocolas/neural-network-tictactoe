import sys
import os
import numpy as np

from adapters.minimax_player import MinimaxPlayer
from services.tic_tac_toe_simulator import TicTacToeSimulator
from entities.neural_network import NeuralNetwork
from usecases.genetic_algorithm import GeneticAlgorithm

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

    def __init__(self):
        self.simulator = TicTacToeSimulator(None, None)

    # ---------- helpers ----------
    @staticmethod
    def clear_screen():
        os.system("cls" if os.name == "nt" else "clear")

    @staticmethod
    def render_board(board: np.ndarray):
        def cell_str(v: int) -> str:
            if v == +1:
                return " X "
            if v == -1:
                return " O "
            return "   "

        rows = ["|".join(cell_str(board[i, j]) for j in range(3)) for i in range(3)]
        print("\n---+---+---\n".join(rows))

    def human_move(self, board: np.ndarray) -> tuple[int, int]:
        occupied = {i + 1 for i, v in enumerate(board.flatten()) if v != 0}
        while True:
            choice = input("Escolha uma posição (1-9): ").strip()
            if choice.isdigit():
                pos = int(choice)
                if 1 <= pos <= 9 and pos not in occupied:
                    idx = pos - 1
                    return idx // 3, idx % 3
            print("Jogada inválida. Tente novamente.")

    # ---------- menu ----------
    def start(self):
        while True:
            print(
                """
=== MENU ===
1 - Jogar contra Minimax (modo humano)
2 - Treinar IA com AG
3 - Jogar contra IA treinada
0 - Sair
"""
            )
            cmd = input("Escolha: ").strip()
            if cmd == "1":
                self.start_game_against_minimax()
            elif cmd == "2":
                self.start_train_network()
            elif cmd == "3":
                self.start_game_against_network()
            elif cmd == "0":
                print("Até logo!")
                sys.exit(0)
            else:
                print("Opção inválida. Tente novamente.")

    # ---------- modo 1 ----------
    def start_game_against_minimax(self):
        player = MinimaxPlayer()
        board = np.zeros((3, 3), dtype=int)

        turn = -1  # HUMANO começa como O (-1)

        while True:
            self.clear_screen()
            self.render_board(board)

            if turn == +1:  # jogada do Minimax (X)
                r, c = player.move(board.tolist())
            else:           # jogada do humano (O)
                r, c = self.human_move(board)

            board[r, c] = turn
            winner = self.simulator.check_winner(board)
            if winner is not None:
                self.clear_screen()
                self.render_board(board)

                if winner == +1:
                    print("\nMinimax (X) vence!")
                elif winner == -1:
                    print("\nVocê (O) vence!")
                else:
                    print("\nEmpate!")

                input("\nPressione Enter para voltar ao menu...")
                return

            turn *= -1

    # ---------- modo 2 ----------
    def start_train_network(self):
        gens = int(input("Quantas gerações deseja treinar? ").strip())
        ga = GeneticAlgorithm(population_size=200, generations=gens, n_games=10)

        print("\nIniciando treinamento...\n")
        best = ga.evolve(verbose=True)
        np.save("rnn.npy", best)
        print("\nTreino concluído. Melhor weights_vector salvo em 'rnn.npy'.")
        input("\nPressione Enter para voltar ao menu...")

    # ---------- modo 3 ----------
    def start_game_against_network(self):
        path = input("Caminho do weights_vector [rnn.npy]: ").strip() or "rnn"
        weights_vector = np.load(f"{path}.npy")

        nn = NeuralNetwork(9, 9, 9, weights_vector)
        board = np.zeros((3, 3), dtype=int)

        turn = +1  # Rede Neural começa como X (+1)

        while True:
            self.clear_screen()
            self.render_board(board)

            if turn == +1:  # jogada da IA (X)
                idx = nn.predict(board.flatten())
                r, c = divmod(idx, 3)
            else:           # jogada do humano (O)
                r, c = self.human_move(board)

            board[r, c] = turn
            winner = self.simulator.check_winner(board)
            if winner is not None:
                self.clear_screen()
                self.render_board(board)

                if winner == +1:
                    print("\nRede Neural (X) vence!")
                elif winner == -1:
                    print("\nVocê (O) vence!")
                else:
                    print("\nEmpate!")

                input("\nPressione Enter para voltar ao menu...")
                return

            turn *= -1


# ---------- execução direta ----------
if __name__ == "__main__":
    TicTacToeCLI().start()
