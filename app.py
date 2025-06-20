from flask import Flask, render_template, jsonify, request, send_from_directory, Response, stream_with_context
import numpy as np
import json
import os
import sys
import time
import random
from pathlib import Path
from functools import wraps
from typing import List, Tuple, Optional, Dict, Any

# Adiciona o diretório src ao path para importar os módulos
sys_path = str(Path(__file__).parent / 'src')
if sys_path not in sys.path:
    sys.path.append(sys_path)

# Configura o caminho para os templates
app = Flask(__name__, template_folder='templates', static_folder='static')

# Tipos de jogadores
PLAYER = -1
AI = 1
EMPTY = 0

# Estado do jogo
game_state = {
    'board': [[EMPTY] * 3 for _ in range(3)],
    'current_player': PLAYER,
    'game_mode': 'minimax',  # 'minimax' ou 'neural_network'
    'game_over': False,
    'winner': None
}

from threading import Lock

# Lock para evitar condições de corrida
game_lock = Lock()

class TrainingManager:
    """Classe para gerenciar o estado do treinamento."""
    
    def __init__(self):
        self.training_in_progress = False
        self.training_progress = 0.0
        self.training_generation = 0
        self.training_best_fitness = 0.0
        self.training_cancel = False
        self.training_total_generations = 0
        self.training_start_time = 0
        self.training_stats = {}
        self.model = None
        self.MODEL_LOADED = False

# Instância global do gerenciador de treinamento
training_manager = TrainingManager()

# Importa o minimax e a rede neural
try:
    from minimax import minimax
    from entities.neural_network import NeuralNetwork
    
    # Tenta importar o treinamento genético, mas não falha se não estiver disponível
    try:
        from usecases import train_neural_network
    except ImportError as e:
        print(f"Aviso: Módulo de treinamento genético não disponível: {e}")
        train_neural_network = None
    
    # Cria o diretório de modelos se não existir
    os.makedirs('models', exist_ok=True)
    
    # Tenta carregar o modelo treinado
    try:
        # Caminho para o modelo salvo
        model_path = os.path.join('models', 'rnn.npy')
        if os.path.exists(model_path):
            # Carrega os pesos salvos
            weights = np.load(model_path)
            # Inicializa a rede neural com os pesos carregados
            input_size = 9
            hidden_size = 9  
            output_size = 9
            training_manager.model = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size, weights_vector=weights)
            training_manager.MODEL_LOADED = True
            print("Modelo carregado com sucesso!")
            print(f"Modelo carregado de: {os.path.abspath(model_path)}")
    except Exception as e:
        print(f"Aviso: Não foi possível carregar o modelo treinado: {e}")
        training_manager.MODEL_LOADED = False
        training_manager.model = None
    
    # Se não conseguiu carregar o modelo, define um modelo vazio
    if not hasattr(training_manager, 'model') or training_manager.model is None:
        print("Iniciando sem modelo pré-treinado. Um novo modelo será criado quando necessário.")
        training_manager.MODEL_LOADED = False
        training_manager.model = None
        
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    minimax = None
    train_neural_network = None
    NeuralNetwork = None
    training_manager.MODEL_LOADED = False
    training_manager.model = None

# Tipos de jogadores
PLAYER = -1
AI = 1
EMPTY = 0

# Estado do jogo
game_state = {
    'board': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    'current_player': PLAYER,
    'game_mode': 'minimax',
    'game_over': False,
    'winner': None,
    'training': False,
    'training_progress': 0
}

# Lock para operações atômicas
from threading import Lock
game_lock = Lock()

# Dados de treinamento
training_data = {
    'status': 'idle',
    'progress': 0,
    'best_fitness': 0,
    'current_generation': 0,
    'total_generations': 0
}

def check_winner(board: List[List[int]]) -> Optional[int]:
    """
    Verifica se há um vencedor no tabuleiro.
    Retorna 1 se a IA venceu, -1 se o jogador venceu, 0 para empate ou None se o jogo não acabou.
    """
    # Verifica linhas e colunas
    for i in range(3):
        # Linhas
        if board[i][0] == board[i][1] == board[i][2] != EMPTY:
            return board[i][0]
        # Colunas
        if board[0][i] == board[1][i] == board[2][i] != EMPTY:
            return board[0][i]
    
    # Verifica diagonais
    if board[0][0] == board[1][1] == board[2][2] != EMPTY:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != EMPTY:
        return board[0][2]
    
    # Verifica empate
    if all(cell != EMPTY for row in board for cell in row):
        return 0
    
    # Jogo não acabou
    return None

def get_available_moves(board: List[List[int]]) -> List[Tuple[int, int]]:
    """Retorna uma lista de tuplas (linha, coluna) com as jogadas disponíveis."""
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == EMPTY]

def make_random_move(board: List[List[int]]) -> Tuple[int, int]:
    """Faz uma jogada aleatória em uma posição vazia."""
    available = get_available_moves(board)
    return random.choice(available) if available else (None, None)

class TrainingManager:
    """Classe para gerenciar o estado do treinamento."""
    
    def __init__(self):
        self.training_in_progress = False
        self.training_progress = 0.0
        self.training_generation = 0
        self.training_best_fitness = 0.0
        self.training_cancel = False
        self.training_total_generations = 0
        self.training_start_time = 0
        self.training_stats = {}
        self.model = None
        self.MODEL_LOADED = False

# Instância global do gerenciador de treinamento
training_manager = TrainingManager()

def train_ai(generations, games, population_size):
    """Função para treinar a IA em segundo plano."""
    global training_manager
    
    # Inicializa o gerenciador de treinamento
    training_manager.training_in_progress = True
    training_manager.training_progress = 0.0
    training_manager.training_generation = 0
    training_manager.training_best_fitness = 0.0
    training_manager.training_cancel = False
    training_manager.training_total_generations = generations
    training_manager.training_start_time = time.time()
    
    # Inicializa estatísticas
    training_manager.training_stats = {
        'generations_completed': 0,
        'best_fitness': 0.0,
        'start_time': training_manager.training_start_time,
        'end_time': 0,
        'total_generations': generations,
        'games_per_generation': games,
        'population_size': population_size,
        'status': 'training'
    }
    
    print(f"Iniciando treinamento com {generations} gerações, {games} jogos/geração e população de {population_size}")
    
    # Função de callback para atualizar o progresso
    def on_generation(gen, best_individual, stats=None):
        try:
            # Atualiza as variáveis de progresso
            current_gen = gen + 1  # +1 porque começa em 0
            training_manager.training_generation = current_gen
            training_manager.training_progress = current_gen / generations
            
            # Obtém o fitness do melhor indivíduo
            # O Chromosome armazena o score diretamente no atributo 'score'
            current_fitness = getattr(best_individual, 'score', 0.0)
            
            # Atualiza o melhor fitness se for maior
            if current_fitness > training_manager.training_best_fitness:
                training_manager.training_best_fitness = current_fitness
            
            # Prepara estatísticas básicas
            stats_update = {
                'generations_completed': current_gen,
                'best_fitness': float(training_manager.training_best_fitness),
                'current_fitness': float(current_fitness),
                'progress': float(training_manager.training_progress),
                'elapsed_time': time.time() - training_manager.training_start_time,
                'last_update': time.time(),
                'status': 'training'
            }
            
            # Adiciona estatísticas adicionais se disponíveis
            if stats is not None:
                if isinstance(stats, dict):
                    stats_update.update(stats)
                else:
                    # Tenta converter stats para dicionário se for um objeto
                    try:
                        stats_dict = {k: float(v) if hasattr(v, '__float__') else str(v) 
                                  for k, v in vars(stats).items() if not k.startswith('_')}
                        stats_update.update(stats_dict)
                    except Exception as e:
                        print(f"Erro ao processar estatísticas: {e}")
            
            # Atualiza as estatísticas no gerenciador
            training_manager.training_stats.update(stats_update)
            
            # Log de progresso
            print(f"Geração {current_gen}/{generations} - Fitness: {current_fitness:.4f} (Melhor: {training_manager.training_best_fitness:.4f})")
            
            # Verifica se o treinamento foi cancelado
            if training_manager.training_cancel:
                raise Exception("Treinamento cancelado pelo usuário")
                
        except Exception as e:
            print(f"Erro no callback de geração: {e}")
            raise
    
    try:
        # Inicia o treinamento
        best_individual = train_neural_network(
            generations=generations,
            population_size=population_size,
            games_per_generation=games,
            callback=on_generation
        )
        
        # Atualiza o modelo global
        training_manager.MODEL_LOADED = True
        training_manager.training_in_progress = False
        training_manager.training_stats.update({
            'end_time': time.time(),
            'status': 'completed',
            'progress': 1.0
        })
        
        # Atualiza o modelo com o melhor indivíduo
        from entities.neural_network import NeuralNetwork
        
        # Cria uma nova instância do modelo com os pesos do melhor indivíduo
        # Usando a arquitetura 9-9-9 conforme definido no GeneticAlgorithm
        input_size = 9
        hidden_size = 9  # Camada oculta com 9 neurônios (mesmo que entrada/saída)
        output_size = 9
        
        try:
            # O best_individual já é o vetor de pesos
            training_manager.model = NeuralNetwork(input_size, hidden_size, output_size, best_individual)
            print(f"Modelo criado com sucesso com {len(best_individual)} pesos")
        except Exception as e:
            print(f"Erro ao criar o modelo: {str(e)}")
            raise
        
        # Salva o modelo treinado
        os.makedirs('models', exist_ok=True)
        model_path = os.path.join('models', 'rnn.npy')
        
        try:
            # Garante que o diretório existe
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Salva os pesos do modelo
            np.save(model_path, best_individual.weights_vector if hasattr(best_individual, 'weights_vector') else best_individual)
            
            # Atualiza o gerenciador de treinamento
            training_manager.model = training_manager.model  # Já foi atualizado anteriormente
            training_manager.MODEL_LOADED = True
            training_manager.training_in_progress = False
            training_manager.training_progress = 1.0
            
            # Atualiza estatísticas
            training_manager.training_stats.update({
                'end_time': time.time(),
                'status': 'completed',
                'progress': 1.0,
                'best_fitness': float(training_manager.training_best_fitness),
                'model_saved': True,
                'model_path': os.path.abspath(model_path)
            })
            
            print(f"\nTreinamento concluído em {time.time() - training_manager.training_start_time:.2f} segundos")
            print(f"Melhor fitness alcançado: {training_manager.training_best_fitness:.4f}")
            print(f"Modelo salvo em: {os.path.abspath(model_path)}")
            
            return best_individual
            
        except Exception as save_error:
            error_msg = f"Erro ao salvar o modelo: {str(save_error)}"
            print(f"\n{'='*50}")
            print(error_msg)
            print(f"{'='*50}\n")
            
            # Atualiza estatísticas de erro
            training_manager.training_stats.update({
                'end_time': time.time(),
                'status': 'error',
                'error': str(save_error),
                'error_type': type(save_error).__name__,
                'model_saved': False
            })
            
            # Propaga o erro para tratamento externo
            raise Exception(f"Erro ao salvar o modelo: {str(save_error)}") from save_error
        
    except Exception as e:
        error_msg = f"Erro durante o treinamento: {str(e)}"
        print("\n" + "="*50)
        print(error_msg)
        print("="*50 + "\n")
        
        # Atualiza estatísticas de erro
        training_manager.training_in_progress = False
        training_manager.training_stats.update({
            'end_time': time.time(),
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        })
        raise

@app.route('/')
def index():
    """Rota principal que exibe a interface de treinamento."""
    return render_template('index.html')

@app.route('/api/train', methods=['GET', 'POST'])
def train():
    global training_manager
    
    if training_manager.training_in_progress:
        return jsonify({
            'status': 'error',
            'message': 'Já existe um treinamento em andamento',
            'training_in_progress': True
        }), 400
    
    # Obtém os parâmetros da requisição
    if request.method == 'POST':
        data = request.get_json() or {}
    else:  # GET request
        data = request.args
    
    generations = int(data.get('generations', 20))
    games_per_gen = int(data.get('games', 20))
    population_size = int(data.get('population', 20))
    
    # Reinicializa o estado do gerenciador de treinamento
    training_manager.training_in_progress = True
    training_manager.training_progress = 0.0
    training_manager.training_generation = 0
    training_manager.training_total_generations = generations
    training_manager.training_best_fitness = 0.0
    training_manager.training_cancel = False
    training_manager.training_start_time = time.time()
    training_manager.MODEL_LOADED = False
    training_manager.model = None
    
    # Inicializa estatísticas
    training_manager.training_stats = {
        'generations_completed': 0,
        'best_fitness': 0.0,
        'start_time': training_manager.training_start_time,
        'end_time': 0,
        'total_generations': generations,
        'games_per_generation': games_per_gen,
        'population_size': population_size,
        'status': 'initializing',
        'progress': 0.0,
        'elapsed_time': 0.0,
        'current_fitness': 0.0
    }
    
    # Inicia o treinamento em uma thread separada
    import threading
    training_thread = threading.Thread(
        target=train_ai,
        args=(generations, games_per_gen, population_size)
    )
    training_thread.daemon = True
    training_thread.start()
    
    return jsonify({
        'status': 'started',
        'message': 'Treinamento iniciado',
        'generations': generations,
        'games_per_generation': games_per_gen,
        'population_size': population_size,
        'start_time': training_manager.training_start_time
    })

@app.route('/api/train/status')
def training_status():
    """Endpoint para obter atualizações de status do treinamento via Server-Sent Events."""
    global training_manager
    
    def generate():
        client_id = f"client-{time.time()}"
        print(f"Novo cliente conectado ao stream de status: {client_id}")
        
        try:
            while True:
                try:
                    # Obtém o status atual do treinamento
                    stats = training_manager.training_stats
                    current_time = time.time()
                    
                    # Verifica se o treinamento está em andamento
                    if not training_manager.training_in_progress:
                        if training_manager.training_progress >= 1.0 or 'status' in stats and stats['status'] == 'completed':
                            # Treinamento concluído com sucesso
                            data = {
                                'status': 'completed',
                                'progress': 1.0,
                                'generation': training_manager.training_generation,
                                'total_generations': training_manager.training_total_generations,
                                'best_fitness': float(training_manager.training_best_fitness) if hasattr(training_manager, 'training_best_fitness') else 0.0,
                                'current_fitness': float(stats.get('current_fitness', 0.0)),
                                'elapsed_time': current_time - training_manager.training_start_time,
                                'stats': stats,
                                'message': 'Treinamento concluído com sucesso!',
                                'model_loaded': training_manager.MODEL_LOADED,
                                'model_available': training_manager.model is not None
                            }
                            print(f"Enviando dados de conclusão: {data}")
                            yield f"data: {json.dumps(data, default=str)}\n\n"
                            break
                        elif training_manager.training_cancel or ('status' in stats and stats['status'] == 'cancelled'):
                            # Treinamento cancelado
                            data = {
                                'status': 'cancelled',
                                'message': 'Treinamento cancelado pelo usuário',
                                'elapsed_time': current_time - training_manager.training_start_time,
                                'stats': stats,
                                'progress': training_manager.training_progress,
                                'best_fitness': float(training_manager.training_best_fitness) if hasattr(training_manager, 'training_best_fitness') else 0.0,
                                'model_loaded': training_manager.MODEL_LOADED
                            }
                            print(f"Enviando dados de cancelamento: {data}")
                            yield f"data: {json.dumps(data, default=str)}\n\n"
                            break
                        elif 'status' in stats and stats['status'] == 'error':
                            # Erro no treinamento
                            data = {
                                'status': 'error',
                                'message': stats.get('error', 'Erro desconhecido durante o treinamento'),
                                'elapsed_time': current_time - training_manager.training_start_time,
                                'stats': stats,
                                'progress': training_manager.training_progress,
                                'best_fitness': float(training_manager.training_best_fitness) if hasattr(training_manager, 'training_best_fitness') else 0.0,
                                'model_loaded': training_manager.MODEL_LOADED
                            }
                            print(f"Enviando dados de erro: {data}")
                            yield f"data: {json.dumps(data, default=str)}\n\n"
                            break
                    
                    # Treinamento em andamento
                    elapsed_time = current_time - training_manager.training_start_time
                    remaining_time = 0
                    progress = training_manager.training_progress
                    
                    # Calcula o tempo restante com base no progresso atual
                    if progress > 0 and progress < 1.0:
                        remaining_time = (elapsed_time / progress) * (1 - progress)
                    
                    # Prepara os dados de status
                    data = {
                        'status': 'training',
                        'progress': float(progress) if progress is not None else 0.0,
                        'generation': int(training_manager.training_generation) if hasattr(training_manager, 'training_generation') else 0,
                        'total_generations': int(training_manager.training_total_generations) if hasattr(training_manager, 'training_total_generations') else 0,
                        'best_fitness': float(training_manager.training_best_fitness) if hasattr(training_manager, 'training_best_fitness') else 0.0,
                        'current_fitness': float(stats.get('current_fitness', 0.0)) if stats else 0.0,
                        'elapsed_time': float(elapsed_time) if elapsed_time is not None else 0.0,
                        'remaining_time': float(remaining_time) if remaining_time and remaining_time > 0 else None,
                        'stats': stats,
                        'message': f'Em andamento: Geração {training_manager.training_generation}/{training_manager.training_total_generations} - Fitness: {training_manager.training_best_fitness:.4f}',
                        'model_loaded': training_manager.MODEL_LOADED,
                        'model_available': training_manager.model is not None
                    }
                    
                    # Envia os dados para o cliente
                    yield f"data: {json.dumps(data, default=str)}\n\n"
                    
                    # Pequena pausa para evitar sobrecarga
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Erro ao gerar dados de status: {str(e)}")
                    yield f"data: {json.dumps({'status': 'error', 'message': f'Erro interno: {str(e)}'})}\n\n"
                    time.sleep(5)  # Pausa antes de tentar novamente
        
        except GeneratorExit:
            print(f"Cliente desconectado: {client_id}")
        except Exception as e:
            print(f"Erro no stream de status para {client_id}: {str(e)}")
            yield f"data: {json.dumps({'status': 'error', 'message': f'Erro no stream de status: {str(e)}'})}\n\n"
    
    # Configura os cabeçalhos para Server-Sent Events
    response = Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'  # Desativa buffering no Nginx
        }
    )
    return response

@app.route('/api/train/cancel', methods=['POST'])
def cancel_train():
    global training_manager
    
    if not training_manager.training_in_progress:
        return jsonify({'status': 'error', 'message': 'Nenhum treinamento em andamento'}), 400
    
    training_manager.training_cancel = True
    
    # Atualiza as estatísticas para refletir o cancelamento
    if 'status' in training_manager.training_stats:
        training_manager.training_stats['status'] = 'cancelled'
    
    return jsonify({
        'status': 'success',
        'message': 'Solicitação de cancelamento do treinamento enviada',
        'cancelled': True
    })

@app.route('/api/move/neural-network', methods=['POST'])
def neural_network_move():
    """Rota específica para obter a jogada da rede neural."""
    global training_manager
    
    data = request.json
    board = data.get('board', [[EMPTY]*3, [EMPTY]*3, [EMPTY]*3])
    
    # Tenta carregar o modelo se não estiver carregado
    if not training_manager.MODEL_LOADED or training_manager.model is None:
        model_path = os.path.join('models', 'rnn.npy')
        if os.path.exists(model_path):
            try:
                # Carrega os pesos do modelo
                weights = np.load(model_path)
                
                # Cria uma nova instância da rede neural com os pesos carregados
                input_size = 9  # 3x3 tabuleiro
                hidden_size = 9  # Mesmo tamanho da camada oculta usado no treinamento
                output_size = 9  # 9 possíveis jogadas
                
                training_manager.model = NeuralNetwork(input_size, hidden_size, output_size, weights)
                training_manager.MODEL_LOADED = True
                print("Modelo carregado com sucesso do arquivo:", model_path)
            except Exception as e:
                print(f"Erro ao carregar o modelo: {str(e)}")
                return jsonify({
                    'error': 'Erro ao carregar o modelo treinado',
                    'status': 'model_load_error',
                    'details': str(e)
                }), 500
        else:
            return jsonify({
                'error': 'Nenhum modelo treinado encontrado. Por favor, treine um modelo primeiro.',
                'status': 'model_not_found'
            }), 404
    
    try:
        # Converte o tabuleiro para o formato esperado pela rede neural (1 para IA, -1 para jogador, 0 para vazio)
        board_flat = []
        for row in board:
            for cell in row:
                if cell == 1:  # IA
                    board_flat.append(1)
                elif cell == -1:  # Jogador
                    board_flat.append(-1)
                else:  # Vazio
                    board_flat.append(0)
        
        # Converte para numpy array
        board_flat = np.array(board_flat, dtype=np.float32)
        
        # Obtém a jogada da rede neural
        move = training_manager.model.predict(board_flat)
        
        # Se a jogada for inválida, faz uma aleatória
        if move == -1 or (0 <= move < 9 and board_flat[move] != 0):
            available = [(i, j) for i in range(3) for j in range(3) if board[i][j] == EMPTY]
            if not available:
                return jsonify({
                    'error': 'Nenhuma jogada disponível',
                    'status': 'no_moves_available'
                }), 400
            row, col = random.choice(available)
        else:
            # Converte o movimento linear para coordenadas (linha, coluna)
            row, col = move // 3, move % 3
        
        return jsonify({
            'status': 'success',
            'row': row,
            'col': col,
            'move_type': 'neural_network'
        })
        
    except Exception as e:
        import traceback
        error_msg = f"Erro na jogada da rede neural: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        
        # Em caso de erro, tenta fazer uma jogada aleatória
        try:
            available = [(i, j) for i in range(3) for j in range(3) if board[i][j] == EMPTY]
            if available:
                row, col = random.choice(available)
                return jsonify({
                    'status': 'success',
                    'row': row,
                    'col': col,
                    'move_type': 'random_fallback',
                    'warning': 'Falha na rede neural, usando jogada aleatória'
                })
        except:
            pass
            
        return jsonify({
            'error': 'Erro ao processar a jogada da rede neural',
            'details': str(e),
            'status': 'neural_network_error'
        }), 500

# Rota para servir arquivos estáticos
@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve arquivos estáticos."""
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    # Cria o diretório para salvar os modelos, se não existir
    os.makedirs('models', exist_ok=True)
    try:
        app.run(debug=True, port=5001)
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print("Port 5001 is already in use. Trying port 5002...")
            app.run(debug=True, port=5002)
        else:
            raise
