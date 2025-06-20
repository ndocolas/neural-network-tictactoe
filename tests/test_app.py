import pytest
import json
import numpy as np
from app import app as flask_app

@pytest.fixture
def app():
    """Cria uma instância do aplicativo Flask para teste."""
    flask_app.config['TESTING'] = True
    return flask_app

@pytest.fixture
def client(app):
    """Cria um cliente de teste para o aplicativo Flask."""
    return app.test_client()

def test_start_game(client):
    """Testa a rota de início de jogo."""
    # Testa com modo minimax
    response = client.post('/api/start', json={'mode': 'minimax'})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'board' in data
    assert 'current_player' in data
    assert data['current_player'] == -1  # Jogador começa
    
    # Testa com modo neural_network
    response = client.post('/api/start', json={'mode': 'neural_network'})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'board' in data
    
    # Testa com modo inválido
    response = client.post('/api/start', json={'mode': 'invalid_mode'})
    assert response.status_code == 400

def test_player_move(client):
    """Testa a rota de jogada do jogador."""
    # Inicia um novo jogo
    client.post('/api/start', json={'mode': 'minimax'})
    
    # Faz uma jogada válida
    response = client.post('/api/move/player', json={'row': 0, 'col': 0})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'board' in data
    assert data['board'][0][0] == -1  # Jogador é -1
    
    # Tenta fazer uma jogada inválida (mesma posição)
    response = client.post('/api/move/player', json={'row': 0, 'col': 0})
    assert response.status_code == 400
    
    # Tenta fazer uma jogada com parâmetros inválidos
    response = client.post('/api/move/player', json={'row': 5, 'col': 5})
    assert response.status_code == 400

def test_ai_move(client):
    """Testa a rota de jogada da IA."""
    # Inicia um novo jogo
    client.post('/api/start', json={'mode': 'minimax'})
    
    # Faz uma jogada do jogador para que a IA possa jogar
    client.post('/api/move/player', json={'row': 0, 'col': 0})
    
    # Faz a IA jogar
    response = client.post('/api/move/ai')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'board' in data
    assert 'row' in data
    assert 'col' in data
    assert data['board'][data['row']][data['col']] == 1  # IA é 1

def test_minimax_move(client):
    """Testa a rota específica de jogada do Minimax."""
    # Cria um tabuleiro onde o Minimax deve bloquear o jogador
    board = [
        [-1, -1, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
    
    response = client.post('/api/move/minimax', json={'board': board})
    assert response.status_code == 200
    data = json.loads(response.data)
    
    # O Minimax deve jogar na posição (0, 2) para bloquear o jogador
    assert (data['row'], data['col']) == (0, 2)

def test_neural_network_move(client):
    """Testa a rota específica de jogada da rede neural."""
    # Cria um tabuleiro simples
    board = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    
    response = client.post('/api/move/neural-network', json={'board': board})
    
    # Pode retornar 200 (sucesso) ou 503 (serviço indisponível se o modelo não estiver treinado)
    if response.status_code == 200:
        data = json.loads(response.data)
        assert 'row' in data
        assert 'col' in data
        assert 0 <= data['row'] <= 2
        assert 0 <= data['col'] <= 2
        assert board[data['row']][data['col']] == 0  # Posição deve estar vazia
    else:
        assert response.status_code == 503

def test_reset_game(client):
    """Testa a rota de reset do jogo."""
    # Inicia um jogo e faz algumas jogadas
    client.post('/api/start', json={'mode': 'minimax'})
    client.post('/api/move/player', json={'row': 0, 'col': 0})
    
    # Reseta o jogo
    response = client.post('/api/reset')
    assert response.status_code == 200
    data = json.loads(response.data)
    
    # Verifica se o tabuleiro foi reiniciado
    assert data['board'] == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    assert data['current_player'] == -1  # Jogador começa

def test_train_endpoint(client):
    """Testa a rota de treinamento da rede neural."""
    # Configuração de treinamento mínima para teste
    train_data = {
        'generations': 1,
        'games_per_generation': 1,
        'population_size': 2
    }
    
    # Inicia o treinamento
    response = client.post('/api/train', json=train_data)
    assert response.status_code in [200, 503]  # 503 se já estiver treinando
    
    if response.status_code == 200:
        # Verifica o status do treinamento
        response = client.get('/api/train/status')
        assert response.status_code == 200
        # A resposta deve ser um stream de eventos
        assert 'text/event-stream' in response.content_type

def test_static_files(client):
    """Testa se os arquivos estáticos estão sendo servidos corretamente."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Jogo da Velha com IA' in response.data
    
    # Testa outros arquivos estáticos
    response = client.get('/static/css/style.css')
    assert response.status_code == 200
    assert b'body' in response.data
    
    response = client.get('/static/js/game.js')
    assert response.status_code == 200
    assert b'initializeGame' in response.data
    
    response = client.get('/static/js/training.js')
    assert response.status_code == 200
    assert b'startTraining' in response.data
