import numpy as np

WIN_LINES = [
    [(0, 0), (0, 1), (0, 2)],
    [(1, 0), (1, 1), (1, 2)],
    [(2, 0), (2, 1), (2, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 1), (1, 1), (2, 1)],
    [(0, 2), (1, 2), (2, 2)],
    [(0, 0), (1, 1), (2, 2)],
    [(2, 0), (1, 1), (0, 2)],
]

def check_winner(board: np.ndarray) -> int:
    """
    Retorna +1 se O vencer; -1 se X vencer; 0 empate; None caso em curso.
    """
    for line in WIN_LINES:
        s = sum(board[r, c] for r, c in line)
        if s == +3: return +1
        if s == -3: return -1
    if not (board == 0).any(): return 0
    return None