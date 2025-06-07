import sys
import numpy as np

from adapters.minimax_player import MinimaxPlayer
from services.tic_tac_toe_simulator import TicTacToeSimulator
from entities.neural_network import NeuralNetwork
from usecases.genetic_algorithm import GeneticAlgorithm

class TicTacToeCLI:
    """
    Interface de linha de comando para:
      - jogo contra Minimax
      - treino via AG
      - jogo contra rede treinada
    """

    def __init__(self):
        # vamos usar um simulador apenas para checar vencedores
        self.simulator = TicTacToeSimulator(None, None)

    def start(self):
        """
        Menu principal, repassa para os métodos especializados.
        """
        while True:
            print("""
=== MENU ===
1 - Jogar contra Minimax (modo humano)
2 - Treinar IA com AG
3 - Jogar contra IA treinada
0 - Sair
""")

            choice = input("Escolha: ").strip()
            if choice == "1":
                self.start_game_against_minimax()
            elif choice == "2":
                self.start_train_network()
            elif choice == "3":
                self.start_game_against_network()
            elif choice == "0":
                print("Até logo!")
                sys.exit(0)
            else:
                print("Opção inválida. Tente novamente.")

    def start_game_against_minimax(self):
        """
        Inicia o jogo humano (X) vs Minimax (O).
        """
        player = MinimaxPlayer()
        board = np.zeros((3, 3), dtype=int)
        turn = +1  # Minimax começa como +1 (O)

        while True:
            print(board)
            if turn == +1:
                r, c = player.move(board.tolist())
                print(f"Minimax (O) jogou em {(r, c)}")
            else:
                r, c = self.human_move(board)
            board[r, c] = turn

            winner = self.simulator.check_winner(board)
            if winner is not None:
                print(board)
                if winner == +1:
                    print("Minimax (O) vence!")
                elif winner == -1:
                    print("Você (X) vence!")
                else:
                    print("Empate!")
                return

            turn *= -1

    def start_train_network(self):
        """
        Inicia o treino do AG com progresso visível no terminal.
        """
        gens = int(input("Quantas gerações deseja treinar? "))
        ga = GeneticAlgorithm(
            input_size=9,
            hidden_size=9,
            output_size=9,
            population_size=100,
            generations=gens,
            # elitism=2,
            tournament_size=3,
            crossover_rate=0.7,
            mutation_rate=0.1,
            mutation_scale=0.5,
        )
        print("\nIniciando treinamento...\n")
        best_genome = ga.evolve(verbose=True)
        np.save("rnn.npy", best_genome)
        print("Treino concluído. Melhor genome salvo em 'best_genome.npy'\n")


    def start_game_against_network(self):
        """
        Inicia o jogo humano (X) vs Rede Neural (O) usando genome salvo.
        """
        path = input("Caminho do genome [best_genome.npy]: ").strip() or "rnn"
        genome = np.load(f"{path}.npy")
        nn = NeuralNetwork(input_size=9, hidden_size=9, output_size=9, genome=genome)
        board = np.zeros((3, 3), dtype=int)
        turn = +1  # Rede começa como +1 (O)

        while True:
            print(board)
            if turn == +1:
                idx = nn.predict(board.flatten())
                r, c = divmod(idx, 3)
                print(f"NN (O) jogou em {(r, c)}")
            else:
                r, c = self.human_move(board)
            board[r, c] = turn

            winner = self.simulator.check_winner(board)
            if winner is not None:
                print(board)
                if winner == +1:
                    print("NN (O) vence!")
                elif winner == -1:
                    print("Você (X) vence!")
                else:
                    print("Empate!")
                return

            turn *= -1

    @staticmethod
    def human_move(board: np.ndarray) -> tuple[int, int]:
        """
        Solicita ao humano uma jogada válida.
        """
        free = [(r, c) for r in range(3) for c in range(3) if board[r, c] == 0]
        while True:
            s = input(f"Sua jogada (células livres: {free}) como 'linha,coluna': ")
            try:
                r, c = map(int, s.strip().split(","))
                if (r, c) in free:
                    return r, c
            except:
                pass
            print("Jogada inválida. Tente novamente.")
