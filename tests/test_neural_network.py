import pytest
import numpy as np
from src.entities.neural_network import NeuralNetwork

def test_neural_network_initialization():
    """Testa a inicialização da rede neural."""
    # Cria uma rede neural com pesos específicos para teste
    input_size = 9  # 9 células do tabuleiro
    hidden_size = 9  # 9 neurônios na camada oculta
    output_size = 9  # 9 possíveis jogadas
    
    # Pesos específicos para teste
    weights = np.random.randn(input_size * hidden_size + hidden_size * output_size)
    nn = NeuralNetwork(weights, input_size, hidden_size, output_size)
    
    # Verifica se as camadas foram criadas corretamente
    assert nn.weights_1.shape == (input_size, hidden_size)
    assert nn.weights_2.shape == (hidden_size, output_size)
    assert nn.bias_1.shape == (1, hidden_size)
    assert nn.bias_2.shape == (1, output_size)

def test_neural_network_predict():
    """Testa a previsão da rede neural."""
    # Cria uma rede neural com pesos específicos para teste
    input_size = 9
    hidden_size = 9
    output_size = 9
    
    # Pesos específicos para teste (todos 1 para previsão determinística)
    weights = np.ones(input_size * hidden_size + hidden_size * output_size)
    nn = NeuralNetwork(weights, input_size, hidden_size, output_size)
    
    # Tabuleiro de teste (vazio)
    board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # Faz a previsão
    move = nn.predict(board)
    
    # Verifica se a jogada é válida (entre 0 e 8)
    assert 0 <= move <= 8

def test_neural_network_predict_with_invalid_moves():
    """Testa se a rede neural ignora jogadas inválidas."""
    input_size = 9
    hidden_size = 9
    output_size = 9
    
    # Pesos que favorecem a primeira saída (índice 0)
    weights = np.zeros(input_size * hidden_size + hidden_size * output_size)
    nn = NeuralNetwork(weights, input_size, hidden_size, output_size)
    
    # Tabuleiro onde a primeira posição (índice 0) já está ocupada
    board = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # Faz a previsão
    move = nn.predict(board)
    
    # A rede não deve escolher a posição 0 (já ocupada)
    assert move != 0
    assert 1 <= move <= 8

def test_neural_network_predict_with_full_board():
    """Testa o comportamento da rede neural com tabuleiro cheio."""
    input_size = 9
    hidden_size = 9
    output_size = 9
    
    # Pesos aleatórios
    weights = np.random.randn(input_size * hidden_size + hidden_size * output_size)
    nn = NeuralNetwork(weights, input_size, hidden_size, output_size)
    
    # Tabuleiro cheio (sem jogadas válidas)
    board = [1, -1, 1, -1, 1, -1, 1, -1, 1]
    
    # Faz a previsão (deve retornar -1 quando não há jogadas válidas)
    move = nn.predict(board)
    assert move == -1

def test_neural_network_predict_with_one_valid_move():
    """Testa se a rede neural escolhe a única jogada válida disponível."""
    input_size = 9
    hidden_size = 9
    output_size = 9
    
    # Pesos aleatórios
    weights = np.random.randn(input_size * hidden_size + hidden_size * output_size)
    nn = NeuralNetwork(weights, input_size, hidden_size, output_size)
    
    # Tabuleiro com apenas uma jogada válida (posição 4)
    board = [1, -1, 1, -1, 0, -1, 1, -1, 1]
    
    # Faz a previsão
    move = nn.predict(board)
    
    # A única jogada válida é a posição 4
    assert move == 4
