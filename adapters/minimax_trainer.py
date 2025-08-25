from minimax.minimax import minimax
from typing import List, Tuple
import numpy as np
import random

class MinimaxTrainer:
    """
    Adapter usado no treino.

    • p_minimax  = probabilidade de jogar pelo Minimax perfeito
      - 1.0 → sempre Minimax
      - 0.5 → 50 % Minimax, 50 % aleatório
    O jogador controlado aqui é sempre -1 (O).
    """

    def __init__(self, p_minimax: float = 1.0):
        if not 0.0 <= p_minimax <= 1.0:
            raise ValueError("p_minimax deve estar entre 0.0 e 1.0")

        self.p_minimax = p_minimax

    def move(self, board: List[List[int]]) -> Tuple[int, int]:
        """
        Decide a jogada para o player -1 (O).
        Retorna (r, c).  Se não houver células livres, devolve (-1, -1).
        """
        board_arr = np.asarray(board, dtype=int)
        free = [tuple(pos) for pos in np.argwhere(board_arr == 0)]
        if not free:
            return -1, -1

        if len(free) == 9:
            return random.choice(free)

        use_minimax = random.random() <= self.p_minimax
        if use_minimax:
            inv_board = (-board_arr).tolist()
            r, c = minimax(inv_board)
            return r, c

        return random.choice(free)
