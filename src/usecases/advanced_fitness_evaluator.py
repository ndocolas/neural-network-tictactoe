import numpy as np
import random
from adapters.minimax_trainer import MinimaxTrainer
from entities.neural_network import NeuralNetwork

class AdvancedFitnessEvaluator:
    """
    Avalia um vetor de pesos em `n_games` partidas.
    80 % dos jogos contra Minimax p=0.5, 20 % contra p=1.0.
    Minimax sempre faz o 1º lance com peso center>corners>edges.
    """

    # ----- Pontuações (fáceis de tunar) -----
    WIN_POINTS        = 40
    LOSE_POINTS       = 40
    DRAW_POINTS       = 10

    RIGHT_PLACE       = 3
    WRONG_PLACE       = 15
    
    BLOCK_BONUS       = 5
    MISS_PENALTY      = 5
    RAGE_QUIT_LIMIT   = 20        # inválidas totais → encerra jogo
    RAGE_QUIT_PENALTY = 50

    WIN_LINES = [
        [(0,0),(0,1),(0,2)], [(1,0),(1,1),(1,2)], [(2,0),(2,1),(2,2)],
        [(0,0),(1,0),(2,0)], [(0,1),(1,1),(2,1)], [(0,2),(1,2),(2,2)],
        [(0,0),(1,1),(2,2)], [(2,0),(1,1),(0,2)]
    ]

    def __init__(self, input_size:int, hidden_size:int, output_size:int, n_games:int):
        self.in_size = input_size
        self.h_size  = hidden_size
        self.o_size  = output_size
        self.n_games = n_games

    # ---------- API pública ----------
    def evaluate(self, weights_vector: np.ndarray, gen_number: int) -> float:
        """
        Fitness médio em `n_games`.  Reusa uma instância da rede.
        """
        net = NeuralNetwork(self.in_size, self.h_size, self.o_size, weights_vector)
        games_p50 = int(round(self.n_games * 0.8))

        total = 0.0
        for g in range(self.n_games):
            p_minimax = 0.5 if g < games_p50 else 1.0
            total += self._play_one(net, p_minimax, gen_number)
        return total / self.n_games

    # ---------- helpers ----------
    @staticmethod
    def _immediate_wins(board: np.ndarray, player: int):
        moves = []
        for r, c in np.argwhere(board == 0):
            board[r, c] = player
            if AdvancedFitnessEvaluator._check(board) == player:
                moves.append((r, c))
            board[r, c] = 0
        return moves

    @staticmethod
    def _check(board: np.ndarray):
        for line in AdvancedFitnessEvaluator.WIN_LINES:
            s = sum(board[r, c] for r, c in line)
            if s == +3: return +1
            if s == -3: return -1
        return 0 if not (board == 0).any() else None

    @staticmethod
    def _weighted_first_move(board: np.ndarray):
        """center > corners > edges"""
        center  = [(1,1)]
        corners = [(0,0),(0,2),(2,0),(2,2)]
        edges   = [(0,1),(1,0),(1,2),(2,1)]
        for group in (center, corners, edges):
            free = [p for p in group if board[p]==0]
            if free: return random.choice(free)

    # ---------- partida ----------
    def _play_one(self, net, p_minimax: float, gen: int) -> float:
        board = np.zeros((3,3), dtype=int)
        trainer = MinimaxTrainer(p_minimax)
    
        r,c = self._weighted_first_move(board) if random.random() > 0.5 else tuple(random.choice(np.argwhere(board == 0)))
        board[r, c] = -1

        turn           = +1
        score          = 0.0
        invalid_total  = 0

        while True:
            if turn == +1:
                win_moves   = self._immediate_wins(board, +1)
                block_moves = self._immediate_wins(board, -1)
                idx = net.predict(board.flatten())
                r, c = divmod(idx, 3)

                attempts = 0

                while board[r, c] != 0:
                    score -= self.WRONG_PLACE
                    invalid_total += 1

                    if invalid_total > self.RAGE_QUIT_LIMIT:
                        return score - self.RAGE_QUIT_PENALTY
                    
                    idx = net.predict(board.flatten())
                    r, c = divmod(idx, 3)
                    attempts += 1

                    if attempts >= 10:
                        score -= 3 * self.WRONG_PLACE
                        free = np.argwhere(board == 0)
                        if free.size == 0:
                            return score
                        r, c = random.choice(free)
                        break
                    
                # jogada válida
                score += self.RIGHT_PLACE
                if (r, c) in block_moves and (r, c) not in win_moves:
                    score += self.BLOCK_BONUS
                if win_moves   and (r, c) not in win_moves:
                    score -= self.MISS_PENALTY
                if block_moves and (r, c) not in block_moves:
                    score -= self.MISS_PENALTY
                board[r, c] = +1
            else:
                r, c = trainer.move(board.tolist())
                board[r, c] = -1

            outcome = self._check(board)

            if outcome is not None:
                if   outcome == +1: score += self.WIN_POINTS
                elif outcome == 0 : score += self.DRAW_POINTS
                else              : score -= self.LOSE_POINTS
                return score

            turn *= -1
