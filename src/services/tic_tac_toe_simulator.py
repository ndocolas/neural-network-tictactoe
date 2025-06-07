import numpy as np
from entities.neural_network import NeuralNetwork
from adapters.minimax_player import MinimaxPlayer

class TicTacToeSimulator:
    """
    Simula um jogo entre NeuralNetwork (O = +1) e MinimaxPlayer (X = -1).
    Responsabilidade única: controlar fluxo do jogo e verificar vencedor.
    """
    WIN_LINES = [
        [(0,0),(0,1),(0,2)], [(1,0),(1,1),(1,2)], [(2,0),(2,1),(2,2)],
        [(0,0),(1,0),(2,0)], [(0,1),(1,1),(2,1)], [(0,2),(1,2),(2,2)],
        [(0,0),(1,1),(2,2)], [(2,0),(1,1),(0,2)]
    ]

    def __init__(self, nn: NeuralNetwork, minimax_player: MinimaxPlayer):
        self.nn = nn
        self.minimax = minimax_player

    def check_winner(self, board: np.ndarray) -> int:
        """
        Retorna +1 se O vencer; -1 se X vencer; 0 empate; None caso em curso.
        """
        for line in self.WIN_LINES:
            s = sum(board[r, c] for r, c in line)
            if s == +3: return +1
            if s == -3: return -1
        if not (board == 0).any(): return 0
        return None

    def play(self) -> int:
        """
        Joga partida completa. NN começa (+1).
        Retorna +1 vitória NN; 0 empate; -1 derrota.
        """
        board = np.zeros((3,3), dtype=int)
        turn = +1
        while True:
            if turn == +1:
                idx = self.nn.predict(board.flatten())
                r, c = divmod(idx, 3)
            else:
                r, c = self.minimax.move(board.tolist())

            if board[r, c] != 0:
                return -turn  # jogada inválida é penalizada como perda
            board[r, c] = turn

            result = self.check_winner(board)
            if result is not None:
                return result
            turn *= -1
