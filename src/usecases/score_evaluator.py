import random
from pathlib import Path
import numpy as np

from src.adapters.minimax_trainer import MinimaxTrainer
from src.entities.neural_network import NeuralNetwork
from src.utils import WIN_LINES

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
        try:
            net = NeuralNetwork(self.in_size, self.h_size, self.o_size, weights_vector)
        except Exception as e:
            print(f"Erro ao criar a rede neural: {e}")
            return 0.0

        total = 0.0
        for game in range(self.n_games):
            # 80 % das partidas contra Minimax com profundidade média (p=0.5),
            # 20 % com profundidade máxima (p=1.0)
            p_minimax = 0.5 if game < round(self.n_games * 0.8) else 1.0
            help_mode = game < round(self.n_games * 0.3)  # Ajuda nos primeiros 30% dos jogos
            game_score = self._play_one(net, p_minimax, help_mode)
            total += game_score
            print(f"  Jogo {game+1}/{self.n_games} - Score: {game_score}")
        
        avg_score = total / self.n_games
        print(f"  Média de score: {avg_score}")
        return avg_score

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
        
        # O jogo começa com o Minimax fazendo a primeira jogada
        r, c = random.choice(np.argwhere(board == 0))
        board[r, c] = -1  # -1 representa o Minimax (O)
        
        score = 0.0
        turn = +1  # +1 é a rede neural (X), -1 é o Minimax (O)

        while True:
            if turn == +1:  # ----- TURNO DA REDE -----
                try:
                    # Obtém a previsão da rede neural
                    board_flat = board.flatten()
                    idx = AI.predict(board_flat, help)
                    r, c = divmod(idx, 3)
                    
                    if board[r, c] != 0:
                        # célula ocupada → penaliza e rede tenta de novo
                        score -= self.WRONG_PLACE
                        free = np.argwhere(board == 0)
                        if free.size == 0:
                            return score  # tabuleiro cheio
                        r, c = random.choice(free)
                        print(f"    Rede jogou em célula ocupada! Penalidade de {self.WRONG_PLACE}")
                    else:
                        score += self.RIGHT_PLACE  # jogada válida
                        print(f"    Rede jogou em ({r}, {c}) - Válido")
                    
                    board[r, c] = +1  # +1 representa a rede neural (X)
                    
                except Exception as e:
                    print(f"    Erro na previsão da rede: {e}")
                    free = np.argwhere(board == 0)
                    if free.size > 0:
                        r, c = random.choice(free)
                        board[r, c] = +1
                    else:
                        return score  # tabuleiro cheio
            else:            # ----- TURNO DO MINIMAX -----
                try:
                    r, c = minimax.move(board.tolist())
                    board[r, c] = -1  # -1 representa o Minimax (O)
                    print(f"    Minimax jogou em ({r}, {c})")
                except Exception as e:
                    print(f"    Erro no movimento do Minimax: {e}")
                    free = np.argwhere(board == 0)
                    if free.size > 0:
                        r, c = random.choice(free)
                        board[r, c] = -1
                    else:
                        return score  # tabuleiro cheio

            # Verifica se terminou
            outcome = self._check(board)
            if outcome is not None:
                if outcome == +1:  # Rede venceu
                    score += self.WIN_POINTS
                    print(f"    FIM: Rede venceu! +{self.WIN_POINTS} pontos")
                elif outcome == 0:  # Empate
                    score += self.DRAW_POINTS
                    print(f"    FIM: Empate! +{self.DRAW_POINTS} pontos")
                else:  # Minimax venceu
                    score -= self.LOSE_POINTS
                    print(f"    FIM: Minimax venceu! -{self.LOSE_POINTS} pontos")
                print(f"    Score final do jogo: {score}")
                return score

            turn *= -1  # alterna turno
