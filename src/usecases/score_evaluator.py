from adapters.minimax_trainer import MinimaxTrainer
from entities.neural_network import NeuralNetwork
from utils.utils import check_winner
import numpy as np
import random

class ScoreEvaluator:
    """
    Mede o desempenho médio de uma rede em `n_games` contra o Minimax.
    """

    RIGHT_PLACE = 3
    WRONG_PLACE = 20
    WIN_POINTS  = 40
    LOSE_POINTS = 40
    DRAW_POINTS = 20

    def __init__(self, input_size: int, hidden_size: int, output_size: int, n_games: int):
        self.in_size = input_size
        self.h_size = hidden_size
        self.o_size = output_size
        self.n_games = n_games

    def evaluate(self, weights_vector: np.ndarray) -> float:
        """Retorna a média de pontos obtidos em `n_games`."""
        ai = NeuralNetwork(self.in_size, self.h_size, self.o_size, weights_vector)
        total = 0.0

        for g in range(self.n_games):
            # 80 % p=0.5 (médio); 20 % p=1.0 (difícil)
            p_minimax = 0.5 if g < int(self.n_games * 0.80) else 1.0
            mask_invalid = g < int(self.n_games * 0.30)
            total += self._play_one(ai, p_minimax, mask_invalid)

        return total / self.n_games

    def _play_one(self, ai: NeuralNetwork, p_minimax: float, mask_invalid: bool) -> float:

        board = np.zeros((3, 3), dtype=int)
        minimax = MinimaxTrainer(p_minimax)
        score = 0.0
        turn = -1  # Minimax (-1) começa, conforme enunciado


        while True:
            if turn == +1:  # ---------- TURNO DA REDE ----------
                idx = ai.predict(board.flatten(), mask_invalid)

                r, c = divmod(idx, 3)

                if board[r, c] != 0:
                    score -= self.WRONG_PLACE
                    free = np.argwhere(board == 0)
                    if free.size == 0:
                        return score  # tabuleiro cheio
                    r, c = random.choice(free)
                else:
                    score += self.RIGHT_PLACE

                board[r, c] = +1

            else:           # ---------- TURNO DO MINIMAX ----------
                r, c = minimax.move(board.tolist())
                board[r, c] = -1

            # Verifica término
            outcome = check_winner(board)
            if outcome is not None:
                if outcome == +1:
                    score += self.WIN_POINTS
                elif outcome == 0:
                    score += self.DRAW_POINTS
                else:
                    score -= self.LOSE_POINTS
                return score

            turn *= -1  # alterna turno
