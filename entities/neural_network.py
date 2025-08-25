from entities.layer import Layer
import numpy as np

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
        if board.shape != (9,):
            raise ValueError("board must have shape (9,)")

        # Propagação direta
        h_out = self.hidden_layer.forward(board.astype(float))
        o_out = self.output_layer.forward(h_out)

        # Impede jogada em célula ocupada
        if mask_invalid:
            invalid = board != 0
            o_out = o_out.copy()
            o_out[invalid] = -np.inf
            
            if np.all(np.isneginf(o_out)):
                return -1

        return int(np.argmax(o_out))
