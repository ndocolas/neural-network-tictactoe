from .utils import check_winner
from adapters.minimax_player import MinimaxPlayer
from usecases.genetic_algorithm import GeneticAlgorithm
from entities.neural_network import NeuralNetwork
import numpy as np
import os

def start_game_against_minimax():
        player = MinimaxPlayer()
        board = np.zeros((3, 3), dtype=int)

        turn = -1  # HUMANO começa como O (-1)

        while True:
            clear_screen()
            render_board(board)

            if turn == +1:  # jogada do Minimax (X)
                r, c = player.move(board.tolist())
            else:           # jogada do humano (O)
                r, c = human_move(board)

            board[r, c] = turn
            winner = check_winner(board)
            if winner is not None:
                clear_screen()
                render_board(board)
                clear_screen()

                if winner == +1:
                    print("\nMinimax (X) vence!")
                elif winner == -1:
                    print("\nVocê (O) vence!")
                else:
                    print("\nEmpate!")

                return

            turn *= -1


def start_train_network():
    gens = int(input("Quantas gerações deseja treinar? ").strip())
    games = int(input("Quantos jogos deseja jogar? ").strip())
    pop_size = int(input("Qual tamanho da populacao? ").strip())
    ga = GeneticAlgorithm(population_size=pop_size, generations=gens, n_games=games)

    print("\nIniciando treinamento...\n")
    best = ga.evolve(verbose=True)
    np.save("rnn.npy", best)
    print("\nTreino concluído. Melhor weights_vector salvo em 'rnn.npy'.")


def start_game_against_network():
        path = input("Caminho do weights_vector [rnn.npy]: ").strip() or "rnn"
        weights_vector = np.load(f"{path}.npy")

        nn = NeuralNetwork(9, 9, 9, weights_vector)
        board = np.zeros((3, 3), dtype=int)

        turn = -1  # Rede Neural começa como X (+1)

        while True:
            clear_screen()
            render_board(board)

            if turn == +1:  # jogada da IA (X)
                idx = nn.predict(board.flatten())
                r, c = divmod(idx, 3)
            else:           # jogada do humano (O)
                r, c = human_move(board)

            board[r, c] = turn
            winner = check_winner(board)
            if winner is not None:
                clear_screen()
                render_board(board)

                if winner == +1:
                    print("\nRede Neural (X) vence!")
                elif winner == -1:
                    print("\nVocê (O) vence!")
                else:
                    print("\nEmpate!")

                return

            turn *= -1


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def render_board(board: np.ndarray):
    def cell_str(v: int) -> str:
        if v == +1:
            return " X "
        if v == -1:
            return " O "
        return "   "
    
    rows = ["|".join(cell_str(board[i, j]) for j in range(3)) for i in range(3)]
    print("\n---+---+---\n".join(rows))

def human_move(board: np.ndarray) -> tuple[int, int]:
    occupied = {i + 1 for i, v in enumerate(board.flatten()) if v != 0}
    while True:
        choice = input("Escolha uma posição (1-9): ").strip()
        if choice.isdigit():
            pos = int(choice)
            if 1 <= pos <= 9 and pos not in occupied:
                idx = pos - 1
                return idx // 3, idx % 3
        print("Jogada inválida. Tente novamente.")