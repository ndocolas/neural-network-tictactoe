import random
import numpy as np
from minimax.minimax import minimax

class MinimaxTrainer:
    """
    Adapter para uso durante o treino:
      - p_minimax = 1.0 → sempre usa Minimax ótimo para X (–1)
      - p_minimax = 0.5 → 50% Minimax, 50% aleatório
    """
    def __init__(self, p_minimax: float = 1.0):
        if not 0.0 <= p_minimax <= 1.0:
            raise ValueError("p_minimax deve estar entre 0.0 e 1.0")
        self.p_minimax = p_minimax

    def move(self, board: list[list[int]]) -> tuple[int,int]:
        """
        Se random ≤ p_minimax, inverte board, chama minimax para +1,
        e devolve esse movimento para o player –1; caso contrário,
        joga aleatório entre células livres.
        """
        flat = np.array(board).flatten()
        free = [(i//3, i%3) for i,v in enumerate(flat) if v == 0]

        if random.random() <= self.p_minimax and free:
            inv = [[-cell for cell in row] for row in board]
            r, c = minimax(inv)
            return (r, c)
        return random.choice(free) if free else (-1, -1)
