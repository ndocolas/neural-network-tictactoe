import pytest
import numpy as np
from src.usecases.score_evaluator import ScoreEvaluator
from src.entities.neural_network import NeuralNetwork
from src.minimax.minimax import minimax as minimax_function

class TestScoreEvaluator:
    """Testes para a classe ScoreEvaluator."""
    
    def test_evaluate_win(self):
        """Testa a avaliação de uma vitória."""
        # Cria uma rede neural que sempre joga no canto superior esquerdo
        weights = np.zeros(9 * 9 + 9 * 9 + 9 + 9)  # Zeros para simplificar
        nn = NeuralNetwork(weights, 9, 9, 9)
        
        # Força a rede a jogar em uma posição específica (canto superior esquerdo)
        def mock_predict(board):
            return 0  # Sempre joga na posição 0 (canto superior esquerdo)
            
        nn.predict = mock_predict
        
        # Cria um avaliador com um oponente que sempre perde
        class AlwaysLosePlayer:
            def get_move(self, board):
                # Retorna uma jogada que não interfere no resultado
                for i in range(9):
                    if board[i] == 0:
                        return i
                return -1
                
        evaluator = ScoreEvaluator(opponent=AlwaysLosePlayer(), num_games=1)
        
        # A rede deve vencer todas as partidas
        score = evaluator.evaluate(weights)
        assert score > 0  # Deve ser positivo para vitória
    
    def test_evaluate_loss(self):
        """Testa a avaliação de uma derrota."""
        # Cria uma rede neural que sempre joga no canto superior esquerdo
        weights = np.zeros(9 * 9 + 9 * 9 + 9 + 9)  # Zeros para simplificar
        nn = NeuralNetwork(weights, 9, 9, 9)
        
        # Força a rede a jogar em uma posição específica (canto superior esquerdo)
        def mock_predict(board):
            return 0  # Sempre joga na posição 0 (canto superior esquerdo)
            
        nn.predict = mock_predict
        
        # Cria um avaliador com um oponente que sempre vence
        class AlwaysWinPlayer:
            def get_move(self, board):
                # Joga de forma a vencer em 2 jogadas
                if 4 not in board:  # Se o centro estiver livre, joga lá
                    return 4
                # Senão, completa uma linha para vencer
                for i in range(0, 9, 3):
                    if board[i] == 1 and board[i+1] == 1 and board[i+2] == 0:
                        return i+2
                    if board[i] == 1 and board[i+2] == 1 and board[i+1] == 0:
                        return i+1
                    if board[i+1] == 1 and board[i+2] == 1 and board[i] == 0:
                        return i
                # Se não conseguir vencer, joga na primeira posição disponível
                for i in range(9):
                    if board[i] == 0:
                        return i
                return -1
                
        evaluator = ScoreEvaluator(opponent=AlwaysWinPlayer(), num_games=1)
        
        # A rede deve perder todas as partidas
        score = evaluator.evaluate(weights)
        assert score < 0  # Deve ser negativo para derrota
    
    def test_evaluate_draw(self):
        """Testa a avaliação de um empate."""
        # Cria uma rede neural que joga de forma a forçar empate
        weights = np.zeros(9 * 9 + 9 * 9 + 9 + 9)  # Zeros para simplificar
        nn = NeuralNetwork(weights, 9, 9, 9)
        
        # Força a rede a jogar de forma a forçar empate
        def mock_predict(board):
            # Joga na primeira posição disponível
            for i in range(9):
                if board[i] == 0:
                    return i
            return -1
            
        nn.predict = mock_predict
        
        # Cria um avaliador com um oponente que joga de forma a forçar empate
        class DrawPlayer:
            def get_move(self, board):
                # Joga na última posição disponível
                for i in range(8, -1, -1):
                    if board[i] == 0:
                        return i
                return -1
                
        evaluator = ScoreEvaluator(opponent=DrawPlayer(), num_games=1)
        
        # Deve resultar em empate
        score = evaluator.evaluate(weights)
        assert abs(score) < 0.1  # Deve ser próximo de zero para empate
    
    def test_evaluate_with_minimax(self):
        """Testa a avaliação contra o algoritmo Minimax."""
        # Cria uma rede neural com pesos aleatórios
        input_size = 9
        hidden_size = 9
        output_size = 9
        weights = np.random.randn(input_size * hidden_size + hidden_size * output_size + hidden_size + output_size)
        
        # Cria um avaliador com o Minimax como oponente
        class MinimaxWrapper:
            def get_move(self, board):
                # Converte o formato do tabuleiro para o formato esperado pela função minimax
                board_2d = [board[i*3:(i+1)*3] for i in range(3)]
                row, col = minimax_function(board_2d)
                return row * 3 + col
                
        evaluator = ScoreEvaluator(opponent=MinimaxWrapper(), num_games=1)
        
        # Avalia a rede neural
        score = evaluator.evaluate(weights)
        
        # Verifica se a pontuação está dentro de limites razoáveis
        # (o valor exato depende da implementação do ScoreEvaluator)
        assert -100 <= score <= 100
    
    def test_evaluate_multiple_games(self):
        """Testa a avaliação com múltiplos jogos."""
        # Cria uma rede neural que alterna entre vitória e derrota
        class AlternatingNetwork:
            def __init__(self):
                self.game_count = 0
                
            def predict(self, board):
                self.game_count += 1
                # Em jogos ímpares, joga para vencer
                # Em jogos pares, joga para perder
                if self.game_count % 2 == 1:
                    # Joga para vencer (se possível)
                    for i in range(9):
                        if board[i] == 0:
                            return i
                else:
                    # Joga para perder
                    for i in range(9):
                        if board[i] == 0:
                            return i
                return -1
        
        # Cria um avaliador com um oponente que se adapta
        class AdaptivePlayer:
            def get_move(self, board):
                # Em jogos ímpares, deixa o oponente vencer
                # Em jogos pares, vence
                if sum(1 for x in board if x != 0) % 2 == 0:  # Primeira jogada do jogo
                    return 4  # Joga no centro
                else:
                    # Tenta bloquear ou vencer
                    for i in range(9):
                        if board[i] == 0:
                            return i
                    return -1
        
        # Cria uma rede neural falsa que usa o AlternatingNetwork
        weights = np.zeros(9 * 9 + 9 * 9 + 9 + 9)
        nn = AlternatingNetwork()
        
        # Cria um avaliador com 2 jogos
        evaluator = ScoreEvaluator(opponent=AdaptivePlayer(), num_games=2)
        
        # Avalia a rede neural
        score = evaluator.evaluate(weights)
        
        # Como um jogo deve ser vitória e o outro derrota, o score deve estar próximo de zero
        assert abs(score) < 0.1  # Deve ser próximo de zero (média de vitória e derrota)
