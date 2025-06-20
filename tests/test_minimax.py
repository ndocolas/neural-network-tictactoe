import pytest
import numpy as np
import sys
import os

# Adiciona o diretório raiz ao path do Python para garantir que as importações funcionem
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.minimax.minimax import minimax as minimax_function

def test_minimax_blocks_winning_move():
    """Testa se o Minimax bloqueia uma jogada vencedora do oponente."""
    # Tabuleiro onde o jogador (O) está prestes a vencer na próxima jogada
    board = [
        [1, 0, -1],
        [0, 1, 0],
        [0, 0, -1]
    ]
    row, col = minimax_function(board)
    
    # A melhor jogada deve ser na posição (2, 0) para bloquear a vitória do jogador
    assert (row, col) == (2, 0), f"O Minimax deveria ter jogado em (2, 0), mas jogou em ({row}, {col})"

def test_minimax_wins_if_possible():
    """Testa se o Minimax faz uma jogada vencedora quando possível."""
    # Tabuleiro onde a IA (X) pode vencer na próxima jogada
    board = [
        [1, 0, -1],
        [0, 1, 0],
        [-1, 0, 0]
    ]
    row, col = minimax_function(board)
    
    # A melhor jogada deve ser na posição (1, 2) para vencer o jogo
    assert (row, col) == (1, 2), f"O Minimax deveria ter jogado em (1, 2) para vencer, mas jogou em ({row}, {col})"

def test_minimax_creates_fork():
    """Testa se o Minimax cria uma bifurcação quando possível."""
    # Tabuleiro onde a IA pode criar uma bifurcação
    board = [
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 0]
    ]
    row, col = minimax_function(board)
    
    # As melhores jogadas para criar uma bifurcação são os cantos
    best_moves = [(0, 2), (2, 0), (2, 2)]
    assert (row, col) in best_moves, f"O Minimax deveria ter jogado em um dos cantos para criar uma bifurcação, mas jogou em ({row}, {col})"

def test_minimax_blocks_fork():
    """Testa se o Minimax bloqueia uma bifurcação do oponente."""
    # Tabuleiro onde o jogador pode criar uma bifurcação na próxima jogada
    board = [
        [0, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ]
    row, col = minimax_function(board)
    
    # A melhor jogada deve ser em uma aresta para evitar a bifurcação
    best_moves = [(0, 1), (1, 0), (1, 2), (2, 1)]
    assert (row, col) in best_moves, f"O Minimax deveria ter jogado em uma aresta para bloquear a bifurcação, mas jogou em ({row}, {col})"

def test_minimax_center_control():
    """Testa se o Minimax ocupa o centro quando disponível."""
    # Tabuleiro vazio
    board = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    row, col = minimax_function(board)
    
    # A melhor jogada deve ser no centro
    assert (row, col) == (1, 1), f"O Minimax deveria ter jogado no centro (1, 1), mas jogou em ({row}, {col})"

def test_minimax_corner_control():
    """Testa se o Minimax ocupa um canto quando o centro está ocupado."""
    # Tabuleiro com o centro ocupado pelo oponente
    board = [
        [0, 0, 0],
        [0, -1, 0],
        [0, 0, 0]
    ]
    row, col = minimax_function(board)
    
    # A melhor jogada deve ser em um canto
    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    assert (row, col) in corners, f"O Minimax deveria ter jogado em um canto, mas jogou em ({row}, {col})"
