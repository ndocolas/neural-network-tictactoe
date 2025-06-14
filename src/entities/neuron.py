import numpy as np
from math import exp

class Neuron:
    """
    Representa um neurônio com pesos reais (incluindo bias).
    Responsabilidade única: soma ponderada + ativação logística.
    """
    def __init__(self, weights: np.ndarray):
        """
        :param weights: vetor de tamanho (n_inputs + 1), 
                        último elemento é o peso do bias.
        """
        self.weights = weights

    def activate(self, x: float) -> float:
        """Função logística 1 / (1 + e⁻ˣ)."""
        return 1.0 / (1.0 + exp(-x))

    def forward(self, inputs: np.ndarray) -> float:
        """
        Propaga inputs pelo neurônio.
        :param inputs: array de shape (n_inputs,)
        :return: saída ativada
        """
        extended = np.append(inputs, 1.0)
        z = float(np.dot(self.weights, extended))
        return self.activate(z)
