import numpy as np
from entities.layer import Layer

class NeuralNetwork:
    """
    MLP de duas camadas (oculta + saída).
    Responsabilidade única: mapear weights_vector → arquitetura e fazer predição.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, weights_vector: np.ndarray):
        # separa weights_vector em duas matrizes de pesos
        hidden_end = hidden_size * (input_size + 1)
        hidden_w = weights_vector[:hidden_end].reshape(hidden_size, input_size + 1)
        output_w = weights_vector[hidden_end:].reshape(output_size, hidden_size + 1)

        self.hidden_layer = Layer.from_weights_matrix(hidden_w)
        self.output_layer = Layer.from_weights_matrix(output_w)

    def predict(self, board: np.ndarray) -> int:
        """
        Dado um vetor de board (9 células), retorna índice da jogada.
        Escolhe máximo entre células livres.
        """
        h_out = self.hidden_layer.forward(board)
        o_out = self.output_layer.forward(h_out)
        
        return int(np.argmax(o_out))
