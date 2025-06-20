import numpy as np
from src.entities.neuron import Neuron

class Layer:
    """
    Camada de neurônios.
    Responsabilidade única: agrupar neurônios e encaminhar forward.
    """
    def __init__(self, neurons: list[Neuron]):
        self.neurons = neurons

    @classmethod
    def from_weights_matrix(cls, weight_matrix: np.ndarray):
        """
        Constrói camada a partir de matriz de pesos.
        :param weight_matrix: shape (n_neurons, n_inputs + 1)
        """
        return cls([Neuron(w.copy()) for w in weight_matrix])

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Propaga inputs em todos os neurônios.
        :param inputs: array (n_inputs,)
        :return: array (n_neurons,)
        """
        return np.array([n.forward(inputs) for n in self.neurons])
