import random
from pathlib import Path
import numpy as np

from adapters.minimax_trainer import MinimaxTrainer
from entities.neural_network import NeuralNetwork
from utils import WIN_LINES

class ScoreEvaluator:
    """
    Calcula a pontuação média de uma rede em `n_games` partidas
    contra o Minimax.
    """

    # ----- Pontos / penalidades -----
    RIGHT_PLACE = 3
    WRONG_PLACE = 20
    WIN_POINTS  = 40
    LOSE_POINTS = 40
    DRAW_POINTS = 20
    HELP = True

    def __init__(self, input_size: int, hidden_size: int, output_size: int, n_games: int):
        self.in_size = input_size
        self.h_size = hidden_size
        self.o_size = output_size
        self.n_games = n_games

    def evaluate(self, weights_vector: np.ndarray) -> float:
        """
        Devolve a média de pontos em `n_games`.
        Usa sempre a mesma instância de rede para velocidade.
        """
        net = NeuralNetwork(self.in_size, self.h_size, self.o_size, weights_vector)

        total = 0.0
        for game in range(self.n_games):
            # 80 % das partidas contra Minimax com profundidade média (p=0.5),
            # 20 % com profundidade máxima (p=1.0)
            p_minimax = 0.5 if game < round(self.n_games * 0.8) else 1.0
            help = True if game < round(self.n_games * 0.3) else False
            total += self._play_one(net, p_minimax, help)
        return total / self.n_games

    @staticmethod
    def _check(board: np.ndarray):
        """Retorna +1 se X venceu, -1 se O venceu, 0 se empate, None se em jogo."""
        for line in WIN_LINES:
            s = sum(board[r, c] for r, c in line)
            if s == +3:
                return +1
            if s == -3:
                return -1
        return 0 if not (board == 0).any() else None


    def _play_one(self, AI: NeuralNetwork, p_minimax: float, help: bool) -> float:
        board = np.zeros((3, 3), dtype=int)
        minimax = MinimaxTrainer(p_minimax)

        r, c = random.choice(np.argwhere(board == 0))

        board[r, c] = -1

        score = 0.0
        turn = +1

        while True:
            if turn == +1:  # ----- TURNO DA REDE -----
                idx = AI.predict(board.flatten(), help)
                r, c = divmod(idx, 3)

                if board[r, c] != 0:
                    # célula ocupada → penaliza e rede tenta de novo
                    score -= self.WRONG_PLACE

                    free = np.argwhere(board == 0)
                    if free.size == 0:
                        return score  # tabuleiro cheio
                    r, c = random.choice(free)
                else:
                    score += self.RIGHT_PLACE  # jogada válida

                board[r, c] = +1
            else:            # ----- TURNO DO MINIMAX -----
                r, c = minimax.move(board.tolist())
                board[r, c] = -1

            # Verifica se terminou
            outcome = self._check(board)
            if outcome is not None:
                if outcome == +1:
                    score += self.WIN_POINTS
                elif outcome == 0:
                    score += self.DRAW_POINTS
                else:
                    score -= self.LOSE_POINTS
                return score

            turn *= -1  # alterna turno
