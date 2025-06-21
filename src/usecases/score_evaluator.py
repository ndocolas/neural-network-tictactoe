from adapters.minimax_trainer import MinimaxTrainer
from entities.neural_network import NeuralNetwork
from utils.utils import check_winner
import numpy as np


class ScoreEvaluator:
    """
    Mede o desempenho médio de uma rede em `n_games` contra o Minimax.
    Pontuação:
      + Jogada válida            → +10
      + Vitória                  → +40
      + Empate                   → +20
      - Jogada em célula ocupada → -15
      - Derrota                  → -25
    """

    RIGHT_PLACE = 10
    WIN_POINTS  = 40
    DRAW_POINTS = 20

    WRONG_PLACE = 15
    LOSE_POINTS = 25

    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int, n_games: int):
        self.in_size = input_size
        self.h_size = hidden_size
        self.o_size = output_size
        self.n_games = n_games

    # ------------------------------------------------------------------ #
    def evaluate(self, weights_vector: np.ndarray) -> float:
        """Retorna a média de pontos em `n_games`."""
        ai = NeuralNetwork(self.in_size, self.h_size, self.o_size, weights_vector)
        total = 0.0

        for g in range(self.n_games):
            p_minimax   = 0.5 if g < int(self.n_games * 0.80) else 1.0
            mask_invalid = g < int(self.n_games * 0.30)  # “rodinhas” só no início
            total += self._play_one(ai, p_minimax, mask_invalid)

        return total / self.n_games

    # ------------------------------------------------------------------ #
    def _play_one(self, ai: NeuralNetwork, p_minimax: float,
                  mask_invalid: bool) -> float:
        board   = np.zeros((3, 3), dtype=int)
        minimax = MinimaxTrainer(p_minimax)
        score   = 0.0
        turn    = -1  # Minimax (-1) começa

        while True:
            # ------------------ Lance do turno atual -------------------
            if turn == +1:                         # ----- RN -----
                idx = ai.predict(board.flatten(), mask_invalid)

                # Tabuleiro cheio → predict devolve -1 (empate imediato)
                if idx == -1:
                    return score + self.DRAW_POINTS

                if not 0 <= idx < 9:               # índice fora do range
                    return score - self.LOSE_POINTS

                r, c = divmod(idx, 3)

                if board[r, c] != 0:               # célula ocupada
                    return score - self.WRONG_PLACE

                board[r, c] = +1
                score += self.RIGHT_PLACE

            else:                                  # ----- Minimax -----
                r, c = minimax.move(board.tolist())

                # Minimax devolve (-1, -1) → tabuleiro cheio → empate
                if (r, c) == (-1, -1):
                    return score + self.DRAW_POINTS

                board[r, c] = -1

            # ------------------ Checa término -------------------------
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
