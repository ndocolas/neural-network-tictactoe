import numpy as np
import random

from adapters.minimax_trainer import MinimaxTrainer
from entities.neural_network import NeuralNetwork
from entities.chromosome import Chromosome

class AdvancedFitnessEvaluator:
    """
    Responsabilidade única: avaliar um Chromosome via reforço detalhado.
    - +1 ponto por jogada válida
    - -5 pontos por jogada inválida e fim imediato
    - +10 pontos por vitória, plus bônus por vencer rápido
    - +0.5 pontos por empate
    - -10 pontos por derrota, plus penalidade extra por perder rápido
    """

    WIN_LINES = [
        [(0,0),(0,1),(0,2)], [(1,0),(1,1),(1,2)], [(2,0),(2,1),(2,2)],
        [(0,0),(1,0),(2,0)], [(0,1),(1,1),(2,1)], [(0,2),(1,2),(2,2)],
        [(0,0),(1,1),(2,2)], [(2,0),(1,1),(0,2)]
    ]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_games: int = 5
    ):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_games     = n_games

    def evaluate(self, chrom: Chromosome, p_minimax: float) -> float:
        """
        Roda n_games partidas e retorna o fitness médio.
        """
        total = 0.0
        for _ in range(self.n_games):
            total += self._eval_one(chrom.genome, p_minimax)
        return total / self.n_games

    def _eval_one(self, genome: np.ndarray, p_minimax: float) -> float:
        board = np.zeros((3,3), dtype=int)
        trainer = MinimaxTrainer(p_minimax)
        nn = NeuralNetwork(self.input_size, self.hidden_size, self.output_size, genome)
        turn = +1   # +1 = NN, -1 = adversário
        moves_made = 0
        score = 0.0
        max_moves = 9

        while True:
            if turn == +1:
                # jogada da rede
                idx = nn.predict(board.flatten())
                r, c = divmod(idx, 3)
                moves_made += 1

                if board[r, c] != 0:
                    # jogada inválida: penalidade e fim imediato
                    score -= 5.0
                    return score

                # jogada válida
                score += 1.0
            else:
                # jogada do adversário (Minimax ou aleatório)
                r, c = trainer.move(board.tolist())

            board[r, c] = turn

            outcome = self._check_winner(board)
            if outcome is not None:
                # aplicando recompensa final
                if outcome == +1:
                    # vitória: base +10 + bônus por rapidez
                    score += 10.0 + (max_moves - moves_made)
                elif outcome == 0:
                    # empate
                    score += 0.5
                else:
                    # derrota: base -10 - penalidade por rapidez
                    score -= 10.0 + (max_moves - moves_made)
                return score

            turn *= -1

    def _check_winner(self, board: np.ndarray) -> int | None:
        """
        Retorna +1 se O vence, -1 se X vence, 0 empate, None se continuar.
        """
        for line in self.WIN_LINES:
            s = sum(board[r, c] for r, c in line)
            if s == +3: return +1
            if s == -3: return -1
        if not (board == 0).any():
            return 0
        return None
