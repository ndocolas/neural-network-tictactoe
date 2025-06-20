import numpy as np
from src.entities.layer import Layer

class NeuralNetwork:
    """
    MLP de duas camadas (oculta + saída).
    Constrói-se diretamente de um único vetor de pesos.
    """
    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int, weights_vector: np.ndarray):
        
        expected_len = hidden_size * (input_size + 1) + output_size * (hidden_size + 1)
        if weights_vector.size != expected_len:
            raise ValueError(
                f"weights_vector size={weights_vector.size} incompatible "
                f"with network topology (expected {expected_len})"
            )
        
        hidden_end = hidden_size * (input_size + 1)
        hidden_w = weights_vector[:hidden_end].reshape(hidden_size, input_size + 1)
        output_w = weights_vector[hidden_end:].reshape(output_size, hidden_size + 1)

        self.hidden_layer = Layer.from_weights_matrix(hidden_w)
        self.output_layer = Layer.from_weights_matrix(output_w)

    def predict(self, board: np.ndarray, mask_invalid: bool = True) -> int:
        """
        Retorna o índice da jogada escolhida (0-8).
        Se `mask_invalid` == True, nunca devolve célula ocupada;
        se o tabuleiro estiver cheio, devolve -1.
        """
        try:
            if board.shape != (9,):
                raise ValueError(f"board must have shape (9,), got {board.shape}")

            # Converte para float e normaliza se necessário
            board_float = board.astype(float)
            
            # Verifica se a entrada contém valores não numéricos
            if not np.all(np.isfinite(board_float)):
                print(f"  AVISO: Entrada contém valores não numéricos: {board_float}")
                board_float = np.nan_to_num(board_float, nan=0.0, posinf=1.0, neginf=-1.0)

            # Propagação direta
            h_out = self.hidden_layer.forward(board_float)
            o_out = self.output_layer.forward(h_out)
            
            # Verifica se as saídas são válidas
            if not np.all(np.isfinite(o_out)):
                print(f"  AVISO: Saída da rede contém valores não numéricos: {o_out}")
                o_out = np.nan_to_num(o_out, nan=0.0, posinf=1.0, neginf=-1.0)

            # Impede jogada em célula ocupada
            if mask_invalid:
                invalid = board != 0
                o_out_masked = o_out.copy()
                o_out_masked[invalid] = -np.inf
                
                if np.all(np.isneginf(o_out_masked)):
                    print("  AVISO: Todas as jogadas são inválidas (tabuleiro cheio?)")
                    return -1
                
                chosen_idx = int(np.argmax(o_out_masked))
                print(f"  Rede escolheu posição {chosen_idx} (valor: {o_out[chosen_idx]:.4f})")
                return chosen_idx
            else:
                chosen_idx = int(np.argmax(o_out))
                print(f"  Rede escolheu posição {chosen_idx} sem máscara (valor: {o_out[chosen_idx]:.4f})")
                return chosen_idx
                
        except Exception as e:
            print(f"  ERRO em predict: {str(e)}")
            # Em caso de erro, retorna uma jogada aleatória válida
            valid_moves = [i for i, val in enumerate(board) if val == 0]
            if valid_moves:
                return np.random.choice(valid_moves)
            return -1
        
    def save(self, filename: str) -> None:
        """
        Salva os pesos da rede neural em um arquivo.
        
        Args:
            filename: Nome do arquivo para salvar os pesos
        """
        # Obtém os pesos da camada oculta (incluindo bias)
        hidden_weights = np.array([neuron.weights for neuron in self.hidden_layer.neurons])
        # Obtém os pesos da camada de saída (incluindo bias)
        output_weights = np.array([neuron.weights for neuron in self.output_layer.neurons])
        # Concatena os pesos em um único vetor
        weights = np.concatenate([hidden_weights.flatten(), output_weights.flatten()])
        # Salva os pesos no arquivo
        np.save(filename, weights)
        
    @classmethod
    def load(cls, filename: str, input_size: int = 9, hidden_size: int = 18, output_size: int = 9):
        """
        Carrega uma rede neural a partir de um arquivo.
        
        Args:
            filename: Nome do arquivo com os pesos
            input_size: Tamanho da camada de entrada
            hidden_size: Tamanho da camada oculta
            output_size: Tamanho da camada de saída
            
        Returns:
            Instância de NeuralNetwork com os pesos carregados
        """
        # Carrega os pesos do arquivo
        weights = np.load(filename)
        # Cria uma nova instância da rede neural com os pesos carregados
        return cls(input_size, hidden_size, output_size, weights)
